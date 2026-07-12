# Refactor Plan

*v3 — feasibility-checked against the codebase (2026-07-05); naming
revised after review. See [Revision notes](#revision-notes).*

## Goal

Restructure `src/` around a clean separation between a **nondimensional
algorithm core** (plain `SVector{3, Float64}`, no `Unitful`/`Meshes` types)
and a **unitful public API** (`Point{𝕄,C}`, `AbstractSpacing`,
`PointCloud`). The developer thinks in numbers; the user thinks in
dimensions. As a side effect, the four overlong files are split by
concern into focused, single-responsibility modules.

Public API is unchanged — no user-visible behavior changes.

## Why

The codebase already has a nondimensional core — `geometric_utils.jl`
states it in its header comment, and every algorithmic hot loop
(`repel`, `bridson`, gradient limiter, octree queries) already works
with `SVector{3, Float64}` internally. The problem is that the boundary
is **implicit and leaky**: 46 `ustrip` calls are scattered across 12
files, with 5 variants of the same point-to-`SVector` conversion, and
the stripping happens inside hot loops rather than at a single clean
boundary. The worst case is `TriangleOctree`: it stores the `SimpleMesh`
and re-extracts vertices through `Meshes.to` on *every query* — coverage
from a full test run shows ~4.7 × 10⁹ hits on that conversion line.

Consolidating the conversion into one place:
- removes all 46 `ustrip` scatterings from algorithm code
- makes the core testable with plain `rand(SVector{3})` and `h = c -> 0.1`
- eliminates the 5 duplicated conversion variants
- moves per-query unit conversions out of hot loops (triangle cache —
  see Phase 3 — should be a measurable speedup on signed-distance and
  classification workloads)
- lets each file focus on one concern, bringing overlong files into the
  100–350 line range

## Verified Current State

Numbers checked against the tree on 2026-07-05:

- `src/` total: 6653 lines across 32 files
- 4 files account for 2946 lines (44%): `octree.jl` (944),
  `spatial_octree.jl` (799), `triangle_octree.jl` (616), `repel.jl` (587)
- 46 `ustrip` calls across 12 files (repel.jl ×9; metrics, io,
  spacings, octree.jl ×5 each; isinside, spacing_guidance ×4; …)
- 5 variants of the same point→SVector conversion (`_raw_point` in
  repel.jl, `_extract_vertex` in triangle_octree.jl, inline
  `Float64.(ustrip.(to(p)))` in spacings.jl/octree.jl/metrics.jl)
- Box-geometry helpers split across 2 files (`_box_corners` in
  `spatial_octree.jl`, `_box_face_centers`/`_box_edge_midpoints` in
  `spacing_criterion.jl`)
- Leaf classification (`LEAF_*` constants, `classify_leaves!`,
  `_classify_leaf_conservative`) lives in `spatial_octree.jl`, **not**
  `triangle_octree.jl` — the split in Phase 3 draws from both files
- Tests reference only 4 internal names (`WhatsThePoint._angle` ×18,
  `_near_duplicate_keep_mask` ×3, `_compute_normal` ×3,
  `_estimate_volume_points` ×1) — moving internals into a submodule
  is cheap on the test side
- `src/reconstruction.jl` is a 14-line stub that is **not included** in
  the module — dead code, delete in Phase 1
- Stray `.cov` coverage artifacts sit in `src/octree/` — housekeeping,
  already gitignored; remove from the working tree

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Public API layer (src/, module WhatsThePoint)         │
│  PointCloud, PointBoundary, AbstractSpacing            │
│  spacing(p::Point) -> Unitful.Length                   │
│  discretize, repel, isinside, compute_normals          │
│  TriangleOctree (wraps mesh + core index)              │
└──────────────────┬─────────────────────────────────────┘
                   │ to_numerical / from_numerical /
                   │ numerical_spacing / to_numerical_triangles
                   │ (src/numerical.jl — the ONLY place ustrip appears)
┌──────────────────▼─────────────────────────────────────┐
│  Core algorithm layer (src/numerics/, module Numerics) │
│  all functions: SVector{3, Float64} in, Float64 out    │
│  spacing is a (SVector{3,Float64}) -> Float64 callable │
│  no Meshes.jl, no Unitful — enforced by the submodule  │
│  KDTree, SpatialOctree, triangle index live here       │
└────────────────────────────────────────────────────────┘
```

### The core is a real submodule, not a directory convention

`WhatsThePoint.jl` does `using Meshes` and `using Unitful` at module
top, so in a flat module every included file silently sees both — the
"no Unitful in core" rule would be grep-enforced convention only, and
would erode. Instead:

```julia
# src/numerics/Numerics.jl
module Numerics

using StaticArrays, LinearAlgebra, Statistics, Random
using NearestNeighbors: KDTree, knn, knn!
using OhMyThreads: tmap, tmapreduce
using ChunkSplitters

include("geometric_utils.jl")
include("spatial_octree.jl")
# ...

end # module
```

The API layer does `using .Numerics: find_leaf, box_bounds, ...` with an
explicit import list. Inside `Numerics`, any accidental `Meshes.x` or
`ustrip` is an `UndefVarError` at compile/test time — the boundary is
compiler-enforced, not reviewed-for.

Naming: `Numerics` says exactly what the layer is — the numerical
routines of the package, operating on plain floats. Two tempting
alternatives are ruled out: **`Core`** shadows Julia's `Base.Core` and
breaks lowered code and macros in confusing ways, and **`Kernels`** (or
any use of the word "kernel") collides with RBF kernels in
RadialBasisFunctions.jl, which WhatsThePoint interfaces with — to this
package's audience, "kernel" means the basis function, so the word is
reserved for that meaning and must not appear in module, file, function
names, or prose. The directory is `src/numerics/`, matching the module.
If the core ever grows into its own registered package (the `MakieCore`
pattern), the submodule boundary is exactly the cut line.

Fallback if submodule churn is unwanted: keep the flat module and add a
purity-guard `@testitem` that scans `src/numerics/*.jl` for
`ustrip|Unitful|Meshes\.|\bPoint\(`. Weaker (regex vs compiler), but
better than nothing. The plan below assumes the submodule.

### The conversion boundary (`src/numerical.jl`)

Four functions, the only place `ustrip` appears in the package:

```julia
# Point{M,C} -> SVector{3, Float64} (strip units in len_unit, promote to Float64)
to_numerical(p, len_unit) = SVector{3, Float64}(ustrip.(len_unit, to(p)))

# Vector{SVector{3,Float64}} -> Vector{Point} (reattach units)
from_numerical(xs, len_unit) = [Point((x .* len_unit)...) for x in xs]

# AbstractSpacing -> (SVector{3,Float64}) -> Float64, unit-consistent
numerical_spacing(s, len_unit) =
    c -> Float64(ustrip(len_unit, s(Point((c .* len_unit)...))))
numerical_spacing(s::ConstantSpacing, len_unit) =   # avoid Point round-trip
    let h = Float64(ustrip(len_unit, s.Δx)); _ -> h end

# SimpleMesh -> Vector{NTuple{3, SVector{3, Float64}}} (Phase 3)
to_numerical_triangles(mesh, len_unit) = ...
```

Two correctness notes baked into the design (both latent issues in the
current code, made explicit here):

1. **Unit consistency.** A blind strip (`ustrip(spacing(Point(c...)))`,
   as the current `_spacing_value` does) takes whatever unit the spacing
   returns. If the cloud is in `m` and the user writes
   `ConstantSpacing(5mm)`, coordinates and spacing disagree by 1000×.
   Every strip must go through `ustrip(len_unit, x)` so mixed units
   convert instead of silently corrupting. `len_unit` is derived once
   per entry point from the cloud/boundary (`unit(lentype(...))`) and
   threaded through.
2. **CRS assumption.** `from_numerical` reconstructs Cartesian points.
   This matches current behavior (`Point(sv .* len_unit...)` in
   repel.jl), but the assumption now lives in one documented function
   instead of five call sites.

`to_numerical`, `from_numerical`, `numerical_spacing` are internal — not
exported, not in `docs/src/api.md`.

### Spacing inside the core

Core functions never see `AbstractSpacing`. They take `h`, any callable
`(SVector{3,Float64}) -> Float64`. The API layer wraps user spacings
with `numerical_spacing` at each entry point (`discretize`, `repel`,
`TriangleOctree` construction). Consequences:

- `SpacingCriterion` stores the plain callable, not the user's spacing
  object; its API-facing constructor does the wrapping. `_spacing_value`
  disappears (it *was* the wrapping, done per-call).
- `_LeafSpacing` splits: the leaf-field lookup (`find_leaf` + fallback
  callable) is a core callable struct `LeafField`; the
  `<: AbstractSpacing` subtype with the `(::Point)` method is a thin
  API-layer wrapper around it (needed because `_apply_gradient_limit`'s
  result is user-visible via `repel`).

## Target Directory Structure

```
src/
├── WhatsThePoint.jl             # module, includes, exports
├── numerical.jl                 # THE conversion boundary
├── utils.jl
├── geometry.jl                  # unitful Point geometry (centroid, boundingbox)
├── points.jl
├── shadow.jl
├── topology.jl
├── surface.jl
├── volume.jl
├── boundary.jl
├── cloud.jl
├── normals.jl                   # unitful wrapper → Numerics PCA
├── neighbors.jl
├── surface_operations.jl
├── surface_sampling.jl
├── triangle_octree.jl           # public TriangleOctree wrapper (mesh + core index)
├── isinside.jl                  # unitful wrapper → Numerics
├── metrics.jl                   # unitful wrapper + display → Numerics
├── io.jl
├── repel.jl                     # thin unitful wrapper → Numerics       (~100)
├── repel_forces.jl
├── discretization/
│   ├── discretization.jl
│   ├── spacings.jl
│   ├── spacing_guidance.jl
│   └── algorithms/
│       ├── fornberg_flyer.jl
│       ├── octree.jl            # Octree struct + thin wrapper          (~290)
│       ├── slak_kosec.jl
│       └── vandersande_fornberg.jl
└── numerics/                    # ── module Numerics ──
    ├── Numerics.jl              # submodule definition, includes
    ├── traits.jl                # octree trait definitions
    ├── geometric_utils.jl       # triangle-box tests, closest point,
    │                            #   + box corners/face-centers/edge-midpoints (~460)
    ├── spatial_octree.jl        # struct, constructors, subdivide!, add_box!  (~300)
    ├── octree_traversal.jl      # find_leaf, box_center/bounds/size, neighbors(~250)
    ├── octree_balance.jl        # 2:1 balancing                               (~150)
    ├── classification.jl        # LEAF_* constants, classify_leaves!,
    │                            #   _classify_leaf_conservative (from BOTH
    │                            #   spatial_octree.jl and triangle_octree.jl) (~200)
    ├── triangle_index.jl        # core triangle-octree: tree + plain-SVector
    │                            #   triangle cache, construction               (~250)
    ├── pseudonormals.jl         # MeshPseudonormals, angle-weighted build     (~150)
    ├── signed_distance.jl       # signed distance, nearest-triangle queries   (~150)
    ├── spacing_criterion.jl     # SpacingCriterion (plain h), build_node_octree (~170)
    ├── bridson.jl               # _BridsonGrid, _generate_bridson,
    │                            #   _bridson_separated, _bridson_h_min        (~280)
    ├── spacing_field.jl         # LeafField, _leaf_spacing_field,
    │                            #   _gradient_limit_field, _apply_gradient_limit (~200)
    ├── leaf_sampling.jl         # LeafGroup, _collect_weighted_leaves,
    │                            #   _allocate_counts_by_volume, _rand_point_in_box,
    │                            #   _generate_points_in_box, _generate_from_leaves,
    │                            #   _non_exterior_leaves, _estimate_volume_points (~230)
    ├── isinside.jl              # winding number (2D), sign queries (3D) on SVectors
    ├── normals.jl               # PCA normal computation on SVectors
    ├── metrics.jl               # dnn_cv, closest_pair, distribution stats
    │                            #   (shared by repel and public metrics — ONE copy)
    └── repel/
        ├── relax.jl             # _relax! main loop                           (~250)
        ├── constraint.jl        # _constrain_octree, _project_to_boundary,
        │                        #   _deposit_escaped!                         (~150)
        └── culling.jl           # _cull, _near_duplicate_keep_mask, _maybe_kick! (~120)
```

Structure notes:
- No separate `box_geometry.jl` — three tiny box helpers don't justify
  a file; they join `geometric_utils.jl`, which is already the "pure
  geometry predicates" home.
- One `numerics/metrics.jl` only — dnn-CV/closest-pair must not exist
  in both a shared file and a repel-local file; repel imports the
  shared implementations.
- `triangle_index.jl` holds only the mesh-free index; the public
  `TriangleOctree` (which owns the `SimpleMesh` and the
  `Manifold`/`CRS` parameters) stays in the API layer as
  `src/triangle_octree.jl`. See Phase 3.
- `src/reconstruction.jl` deleted (dead stub, never included).

Optional, decide at Phase 2: inside `Numerics` the `_` privacy prefix is
redundant (the module boundary marks privacy). Dropping it
(`Numerics.relax!` vs `Numerics._relax!`) reads better but adds rename
churn; either way, be consistent across all core files.

## Execution Plan

Five phases, each a separate PR. Each phase is independently testable
and leaves the package in a working state. Phases 4 and 5 depend on 2;
3 depends on 2; 1 stands alone.

### Phase 1 — Conversion boundary + dedup + housekeeping

**Created:** `src/numerical.jl` (`to_numerical`, `from_numerical`,
`numerical_spacing` — unit-aware `ustrip(len_unit, ...)` form).

**Modified:**
- Replace the 5 inline conversion variants (`_raw_point`,
  `_extract_vertex` call sites outside triangle construction, inline
  `Float64.(ustrip.(to(p)))`) with `to_numerical`; replace
  `Point(sv .* len_unit...)` reconstructions with `from_numerical`.
- `SpacingCriterion` stores a plain `(SVector)->Float64` callable;
  API-facing constructors wrap via `numerical_spacing`. Delete
  `_spacing_value`.
- Consolidate `_box_face_centers`/`_box_edge_midpoints` from
  `spacing_criterion.jl` into `src/octree/geometric_utils.jl` next to
  `_box_corners` (moved there from `spatial_octree.jl`).
- Dedup `max_points = @something(max_points, 10_000_000)` + early-stop
  `@warn` shared by slak_kosec.jl / vandersande_fornberg.jl into a
  helper in `discretization/discretization.jl`.
- Delete `src/reconstruction.jl`; remove stray `.cov` files.
- `src/WhatsThePoint.jl` — `include("numerical.jl")` early.

**No algorithm code moves.** This establishes the boundary later phases
route through.

**Watch for:** the unit-aware `numerical_spacing` is *stricter* than the
old blind strip — if any test relied on mixed-unit slop, it was wrong
before; fix the test, not the boundary (cf. the project convention that
scenario consistency beats making a failing test pass).

**Verify:** standard checks (below) + grep `ustrip` outside
`numerical.jl` — only legitimately-unitful API code (`io.jl`,
`metrics.jl` display, `spacing_guidance.jl`) remains.

### Phase 2 — `Numerics` submodule + move pure octree code

**Created:** `src/numerics/Numerics.jl` — submodule with explicit
imports (StaticArrays, LinearAlgebra, NearestNeighbors, OhMyThreads,
ChunkSplitters, Statistics, Random — and **not** Meshes, Unitful,
CoordRefSystems).

**Moved (no logic change):**
- `src/octree/traits.jl` → `src/numerics/traits.jl`
- `src/octree/geometric_utils.jl` → `src/numerics/geometric_utils.jl`
- `src/octree/spacing_criterion.jl` → `src/numerics/spacing_criterion.jl`
  (already unit-free after Phase 1)

**Split (moved + divided, no logic change):**
- `src/octree/spatial_octree.jl` (799) →
  - `numerics/spatial_octree.jl` (~300): struct, constructors,
    `subdivide!`, `add_box!`, `num_elements`, `bounding_box`
  - `numerics/octree_traversal.jl` (~250): `find_leaf`, `box_center`,
    `box_bounds`, `box_size`, `is_leaf`, `has_children`,
    `find_neighbor`, `find_boxes_at_coords`, `all_leaves`, `all_boxes`,
    `any_leaf_overlapping`
  - `numerics/octree_balance.jl` (~150): `needs_balancing`,
    `balance_octree!`
  - `numerics/classification.jl` (~150 from this file): `LEAF_*`
    constants, `_CLASSIFICATION_INSET`/`_CLASSIFY_TOLERANCE_*`,
    `classify_leaves!`, `_classify_leaf_conservative`

API layer adds `using .Numerics: ...` for the names it re-exports or
uses. Update the 4 internal test references
(`WhatsThePoint._estimate_volume_points` →
`WhatsThePoint.Numerics...`, etc.) as their functions move (this phase
and later ones).

**Verify:** standard checks + the submodule now hard-fails on any
`Meshes.`/`ustrip` reference in `numerics/`.

### Phase 3 — TriangleOctree: triangle cache + split

The one phase with a real (internal) logic change, so it gets its own
PR. `TriangleOctree{M,C,T}` currently stores `mesh::SimpleMesh{M,C}`
and every query re-extracts vertices via `Meshes.vertices`/`Meshes.to`
— ~4.7 × 10⁹ conversions in a test run. A "rename-only" move to a
Meshes-free core is impossible; instead:

1. Add `to_numerical_triangles(mesh, len_unit)` to `numerical.jl`.
2. Core struct `TriangleIndex{T}` in `numerics/triangle_index.jl`: the
   `SpatialOctree`, `triangles::Vector{NTuple{3, SVector{3, T}}}`,
   `MeshPseudonormals`, optional leaf classification, mesh bbox. All
   construction logic that operates on plain triangles moves here.
3. Public `TriangleOctree{M,C,T}` in `src/triangle_octree.jl` wraps
   `TriangleIndex` + the `SimpleMesh` (kept for I/O, visualization,
   `num_triangles`). Its constructors load/validate the mesh, call
   `to_numerical_triangles`, and delegate. `Meshes.*` appears only in
   this wrapper.
4. Query paths (`signed_distance.jl`, `classification.jl`,
   `spacing_criterion.jl`, `isinside`) read the triangle cache — no
   per-query unit conversion.

**Split targets:** `pseudonormals.jl` (~150), `signed_distance.jl`
(~150), `triangle_index.jl` (~250), remainder of classification into
`numerics/classification.jl`.

**Cost:** ~72 bytes/triangle of cache (≈1.8 MB for the 25k-triangle
test bifurcation) — negligible next to the query savings.

**Verify:** standard checks + benchmark `isinside`/`repel(cloud,
spacing, octree)` on `bifurcation.stl` before/after — expect neutral or
faster; regression here means the cache is being bypassed somewhere.

### Phase 4 — Split the Octree discretization algorithm

`src/discretization/algorithms/octree.jl` (944) →

- `numerics/bridson.jl` (~280): `_BridsonGrid`, `_grid_cell`,
  `_grid_insert!`, `_bridson_separated`, `_bridson_inside`,
  `_bridson_h_min`, `_generate_bridson`
- `numerics/spacing_field.jl` (~200): `LeafField` (plain-number
  successor of `_LeafSpacing`), `_leaf_spacing_field`,
  `_gradient_limit_field`, `_apply_gradient_limit`
- `numerics/leaf_sampling.jl` (~230): `LeafGroup`,
  `_collect_weighted_leaves`, `_allocate_counts_by_volume`,
  `_rand_point_in_box`, `_generate_points_in_box`,
  `_generate_from_leaves`, `_non_exterior_leaves`,
  `_estimate_volume_points`
- `discretization/algorithms/octree.jl` (~290, stays): `Octree` struct
  + constructors (parameterized on `Manifold`/`CRS` — API layer),
  `_auto_min_ratio` (takes `SimpleMesh`), `_extract_min_spacing`
  (dispatches on spacing types), the `AbstractSpacing` wrapper around
  `LeafField`, and the thin `_discretize_volume` wrapper.

Every function in the current file is assigned above.

**The wrapper pattern:**
```julia
function _discretize_volume(cloud::PointCloud{𝔼{3}}, spacing, alg::Octree; max_points)
    len_unit = unit(lentype(cloud))
    h = numerical_spacing(spacing, len_unit)       # (SVector)->Float64
    seeds = [to_numerical(p, len_unit) for p in points(boundary(cloud))]
    xs = Numerics.generate(alg_params(alg), h, seeds; max_points)  # pure math
    return PointVolume(from_numerical(xs, len_unit))
end
```
(`alg_params` extracts the plain-number fields of `Octree` so the core
never sees the `{M,C}`-parameterized struct.)

**Verify:** standard checks.

### Phase 5 — Split repel + move isinside/normals/metrics core

`src/repel.jl` (587) →
- `numerics/repel/relax.jl` (~250): `_relax!`, `_safe_direction`
- `numerics/repel/constraint.jl` (~150): `_constrain_octree`,
  `_project_to_boundary`, `_deposit_escaped!`
- `numerics/repel/culling.jl` (~120): `_cull`,
  `_near_duplicate_keep_mask`, `_maybe_kick!`
- `numerics/metrics.jl`: `_dnn_cv`, `_closest_pair` (shared — see below)
- `src/repel.jl` (~100, stays): both public `repel` methods,
  `_reconstruct_cloud`, unit derivation, force-model plumbing

`_constrain_octree`/`_deposit_escaped!` currently build unitful
`Point`s mid-loop (repel.jl:430,462) to run octree projections — after
Phase 3 the projections take plain SVectors, so these become pure and
the `Point` round-trip disappears.

**Core extraction (wrapper stays, math moves):**
- `src/isinside.jl` → winding number / sign queries on SVectors →
  `numerics/isinside.jl`; `Point`/`PointBoundary` dispatch stays.
- `src/normals.jl` → PCA on SVectors → `numerics/normals.jl`; MST/DFS
  orientation and `Vec` handling stay in the wrapper.
- `src/metrics.jl` → statistics on plain distances →
  `numerics/metrics.jl` (merging with repel's
  `_dnn_cv`/`_closest_pair` — after this phase there is exactly one
  dnn-CV implementation in the package); unitful display/formatting
  stays in `src/metrics.jl`.

**Verify:** standard checks + `repel` seeded-gate quality numbers
unchanged (CV within noise of the 0.044 reference).

## Consolidated Duplications

| Duplication | Current locations | Resolved in |
|---|---|---|
| point→SVector conversion (5 variants) | repel.jl, octree.jl, spacings.jl, spacing_guidance.jl, triangle_octree.jl | Phase 1 → `numerical.jl` |
| `_box_corners` vs `_box_face_centers`/`_box_edge_midpoints` | spatial_octree.jl, spacing_criterion.jl | Phase 1 → consolidated in geometric_utils.jl (moved to numerics in Phase 2) |
| `Point(pt...)` reconstruction loops | repel.jl (×2), octree.jl (×2) | Phase 1 → `from_numerical` |
| `max_points` default + early-stop `@warn` | slak_kosec.jl, vandersande_fornberg.jl | Phase 1 → helper in `discretization/` |
| `_spacing_value(T, spacing, c)` per-call wrapping | spacing_criterion.jl, octree.jl (×6) | Phase 1 → `numerical_spacing` at entry points |
| dnn-CV / closest-pair statistics | repel.jl (`_dnn_cv`, `_closest_pair`), metrics.jl | Phase 5 → single `numerics/metrics.jl` |
| per-query mesh vertex extraction | triangle_octree.jl hot paths | Phase 3 → triangle cache |

## Risks and Decisions

- **Submodule vs flat directory** — submodule chosen; it is the only
  enforcement mechanism the compiler checks. Named `Numerics`: not
  `Core` (shadows `Base.Core`), and never anything with "kernel" in it
  (reserved for RBF kernels via RadialBasisFunctions.jl). Fallback:
  flat + purity-guard `@testitem`.
- **Unit-aware `numerical_spacing`** — behavior-identical when spacing
  and cloud units agree (all current tests); *corrects* silent
  1000×-style errors when they don't. Strictly better, but call it out
  in the PR.
- **`TriangleOctree` field change** (Phase 3) is internal but visible
  to anyone poking at `octree.mesh`/struct internals; the public
  constructor and query API are unchanged. Memory +72 B/triangle.
- **CRS**: `from_numerical` assumes Cartesian output, same as today —
  documented in `numerical.jl` rather than implicit in five call sites.
- **Docs**: `docs/src/api.md` references may need path/name updates per
  phase; the docs build in the verification list catches breakage.

## Revision notes

### v2 — feasibility corrections from v1

1. Phase 2's "rename only" could not deliver a Meshes-free
   `triangle_octree.jl` — the struct stores `SimpleMesh{M,C}` and
   queries hit `Meshes.to` per call. Now Phase 3, with an explicit
   triangle cache and an API-layer wrapper.
2. The v1 spacing-conversion draft blind-stripped units — now
   `ustrip(len_unit, …)`.
3. "No Unitful in core" was unenforceable in the flat module — core is
   now a real submodule.
4. v1's target tree duplicated dnn-CV/closest-pair across two core
   metrics files — merged.
5. `traits.jl` was moved in the text but missing from the tree.
6. Box-helper destination contradicted itself (geometric_utils.jl vs a
   separate box file) — consolidated into geometric_utils.jl.
7. Classification code was attributed to triangle_octree.jl but mostly
   lives in spatial_octree.jl — split description fixed.
8. ~150 lines of octree.jl (sampling group) were unassigned — full
   function inventory now mapped.
9. `_LeafSpacing`'s `<: AbstractSpacing` + `(::Point)` method is API
   coupling — split into core `LeafField` + API wrapper.
10. Dead `src/reconstruction.jl` and stray `.cov` files — housekeeping
    added to Phase 1.

### v3 — naming

- v2's boundary file `raw.jl` said nothing about what is raw or which
  direction the conversion flows — renamed `numerical.jl`, functions
  `to_numerical` / `from_numerical` / `numerical_spacing` /
  `to_numerical_triangles`.
- v2's submodule name `Kernels` collides with RBF kernels in
  RadialBasisFunctions.jl — renamed `Numerics` (`src/numerics/`). The
  word "kernel" is banned from this refactor's names and prose.

## Verification (every phase)

- `julia --project -e 'using Pkg; Pkg.test()'` — full suite, ~5 min
- Runic on all touched files — CI enforces Runic 1.x
- After Phase 2: the `Numerics` submodule compiles with no
  Meshes/Unitful/CoordRefSystems imports (compiler-enforced; the greps
  `grep -rn "ustrip\|Unitful\|Meshes\." src/numerics/` stay as a cheap
  belt-and-braces CI check)
- After Phase 3: `isinside`/`repel` benchmark on `bifurcation.stl`
  neutral or faster
- After Phase 5: repel seeded-gate CV within noise of 0.044 reference
- `julia --project=docs docs/make.jl` — no broken cross-references
- Public API surface unchanged across all phases (exports diff empty)
