# Simplification Plan

*v3 — 2026-07-12, revised for the `polish_before_JCon` track. Supersedes v1's
"delete SlakKosec" direction: **all algorithms stay**, none of them is a
silent default, and Octree becomes the friction-free golden path. Ease of
reading and maintainability are the paramount criteria; every edit below is
sized to be small and pointed. v3 promotes the `TriangleIndex{T}` triangle
cache from the deferred appendix to the top of Tier A — it is the one
measurable performance lever and it should land before the JCon demo. The
rest of the deep architectural program (Numerics submodule, strip-once
boundary) stays deferred to the post-JCon appendix.*

## Goal

Make the package effortless for an outside reader in three senses:

1. **As a user** — one obvious way to go from a mesh file to a good point
   cloud, with the Octree/bridson pipeline as the default of that path and
   every other algorithm available as an explicit, documented opt-in.
2. **As a reader of the code** — files that can be read top-to-bottom, one
   concern per file, one implementation per concept, no duplicated
   conversion idioms.
3. **As a maintainer** — conventions written down (CLAUDE.md), behavior
   changes separated from moves, every PR independently green.

## Principles

- **Golden path, not gatekeeping.** Octree is the silent default everywhere
  a mesh is reachable; VanDerSandeFornberg / FornbergFlyer remain first-class
  citizens you *choose*. SlakKosec stays in the tree but is **known broken**
  (wrong output) — never selected implicitly, documented as under repair.
  Nothing is deleted.
- **Explicit over implicit — for the *user's* choices.** The library never
  swaps a user-passed algorithm for another (today's 2D method does exactly
  that); where the user expresses no choice, Octree is picked silently.
  Errors that name the fix.
- **Units at the door.** Every user-facing length — spacing, `at_wall`,
  `layer_thickness`, import scale — is a `Unitful.Length`; a bare number is
  rejected with an error that names the fix (`0.5` → `0.5u"mm"`), never
  guessed. The complement holds inside: units taken at the door are honored
  by the internals (B1) — dimensionless knobs (`max_growth`, `alpha`,
  ratios) stay dimensionless. (Established pattern: #103 for mesh import,
  `ConstantSpacing{L <: Unitful.Length}`, unitful `suggest_spacing`
  landmarks.)
- **A library is quiet by default.** Progress narration is opt-in.
- **Moves and behavior changes never share a PR.** Reviewers can trust that
  a "split file" diff is `git diff --color-moved` clean.
- **Vocabulary**: "kernel" is reserved for RBF kernels
  (RadialBasisFunctions.jl) — banned from names and prose here.

---

## Tier A — Friction and UX (small pointed edits)

### A1. `TriangleIndex{T}` — build-once triangle vertex cache (the performance item)

Every hot geometric routine — closest-triangle search
(`triangle_octree.jl:441`), octree subdivision (`triangle_octree.jl:402`),
the pseudonormal signed-distance `isinside` query, `repel`'s wall
projection — calls
`_get_triangle_vertices(T, mesh, tri_idx)` (`triangle_octree.jl:125`) fresh
for every triangle it touches. Each call goes through the Meshes.jl object
model: `mesh[tri_idx]` materializes a `Triangle` via topology indirection,
`Meshes.vertices` yields unitful `Point`s, `Meshes.to` + three `ustrip`s
convert each vertex to `SVector{3,T}` — and the result is discarded
immediately. The same triangle is re-converted thousands of times; a
production `repel(cloud, spacing, octree)` run totals ~4.7×10⁹ `Meshes.to`
extractions of identical values.

**Design — indexed, not soup:** at `TriangleOctree` build time, one sweep
over the mesh extracts the **indexed representation**:

- `vertices::Vector{SVector{3,T}}` — unique vertex coordinates, unit-stripped;
- `triangles::Vector{NTuple{3,Int32}}` — per-triangle vertex indices (the
  connectivity — needed to *rebuild* pseudonormals, which require vertex/edge
  sharing; queries themselves never need it);
- derived soup views `v1/v2/v3::Vector{SVector{3,T}}` for the hot query
  paths (no index gather per access), rebuilt from `(vertices, triangles)`
  in one sweep — never serialized;
- `len_unit` — the stripped unit, stored once as a type-parameterized field
  exactly like `_LeafSpacing` already does (`octree.jl:640-655`, the house
  precedent: raw `T` values + one unit entry re-attached at exit). `T` stays
  a plain `T <: Real` type parameter; no per-value unit wrappers anywhere.

Query paths read the soup arrays — contiguous, unit-free, already in `T` —
and never touch the mesh after construction. Memory is trivial (~2.7 MB for
the 25k-triangle bifurcation mesh in Float32). `(vertices, triangles,
len_unit)` is also the serialization format (A3): all isbits vectors,
JLD2-trivial, no Meshes.jl topology objects near the serializer.

**Scope:** mechanical replacement of query-time
`_get_triangle_vertices`/`_get_triangle_normal` call sites with index reads;
build-time extraction becomes the only `Meshes.to` in the triangle path.
Behavior-preserving — same `T`, same values, same results. The archived
`separate_numerics_from_unitful` branch (tip `25b25ba`) carries the
reference implementation.

**Why first:** it is the only item in this plan that makes
`discretize`/`isinside`/`repel` measurably faster, it is independent of every
other PR, and the golden-path demo (A2 onward) should be fast when outside
readers first run it. Side effect: it removes most hot-path point→`SVector`
variants, shrinking B2's scope.

**Verification:** full suite + Float32 pipeline tests; a before/after
benchmark of `isinside` batch queries and one `repel` iteration on
`bifurcation.stl` recorded in the PR description.

**Implementation status (2026-07-13, working tree, uncommitted):**

- DONE: `TriangleIndex{T}` (indexed + soup views + `len_unit`) in
  `triangle_octree.jl`; `TriangleOctree.index` field; all construction and
  query paths on the index; `repel`'s `_project_to_boundary` reads the
  index + cached `pseudonormals.face`; `sample_surface` builds the index
  once for its dart loop; `suggest_spacing` builds it once for bbox+volume.
- VERIFIED: bit-identical vertices/signed-volume/bbox vs mesh path
  (box.stl, all 46,786 triangles); full suite 190,670 pass / 0 fail /
  2 errored / 2 broken — **both errored items proven pre-existing at HEAD**
  (`octree_regression_curvature.jl`: Float64 probe points vs Float32
  octree; reproduced identically on a clean HEAD worktree).
- DONE, NOT YET VERIFIED: the mixed-precision seam fix (B3 policy,
  Davide-directed: "types must be parameters — Float32/Float64 meshes must
  both work"): converting entry points at `_classify_point_octree`,
  `_compute_signed_distance_octree(point, octree)`, and
  `_project_to_boundary` — convert the query to the octree's `T` once at
  entry, strict `T` below. This should turn the 2 pre-existing errors into
  passes.
- REMAINING: (1) rerun the two curvature repros + full `Pkg.test()`;
  (2) Runic on touched files (`triangle_octree.jl`, `repel.jl`,
  `surface_sampling.jl`, `spacing_guidance.jl`); (3) before/after benchmark
  — baseline HEAD worktree + `bench_triangleindex.jl` +
  `repro_curvature.jl`/`repro_sweep.jl` scripts were in the (ephemeral)
  session scratchpad; recreate with `git worktree add <dir> HEAD` +
  copied `Manifest.toml` if gone, and run `git worktree prune` for the
  stale entry; (4) update `mechanics.md` §5b (TriangleIndex + seam) in the
  same PR.

### A2. Mesh-first entry point, Octree as its default

```julia
discretize(mesh::SimpleMesh, spacing; alg = Octree(mesh; spacing), max_points = nothing)
```

Poisson-disk sampled boundary (`PointBoundary(mesh, spacing)`) + bridson
volume fill, one call. This is the quickstart line in README/docs:

```julia
mesh  = import_mesh("part.stl", u"mm")
h     = suggest_spacing(mesh).h_baseline
cloud = discretize(mesh, h)
```

Plus the bare-`Unitful.Length` convenience overload. (No reference
implementation exists on the archived branch — the claim in v1 was wrong; A2
is written from scratch. The pieces, `PointBoundary(mesh, spacing)` and
`Octree(mesh; spacing)`, already exist.)

**The reference pipeline is `examples/octree_boundary_layer.jl`** — the
current state-of-the-art run (bunny, steep `BoundaryLayerSpacing`,
`max_growth = 0.15`, `alpha = 1.0`, bridson placement, auto `max_points`,
quality metrics + visual slice check). A2 is that script's lines 60–73
collapsed into one call: today the user builds the spacing from
`PointBoundary(mesh)` points, samples the boundary, constructs
`Octree(mesh; spacing, …)`, and passes the spacing twice. Acceptance test
for A2: the example rewritten on `discretize(mesh, spacing)` reproduces its
quality numbers (spacing CV, sep/h_min, coordination, worst-neighbour d_NN
ratio). Its `alpha = 1.0` vs the constructor default `2.0` was benchmarked
and resolved: the golden path ships `2.0` (see O4 — quality parity, ~20 %
faster, 6.8× smaller octree); the example's `1.0` is the documented
maximum-fidelity setting.

### A2b. Per-surface spacing via combinators (Option B — bulk is a floor)

The spacing system today has no way to say "0.5 mm near the inlet, 2 mm near
the wall, 5 mm in the bulk" even though `PointBoundary` already carries named
surfaces. `BoundaryLayerSpacing` and `LogLike` both collapse all surfaces into
one aggregate KDTree, throwing away the surface identity that `split_surface!`
creates.

The contract `s(p) -> Unitful.Length` is already the uniform seam — every
consumer (`SpacingCriterion`, `_LeafSpacing`, `sample_surface`, bridson,
`_guard_coarse_spacing`) goes through either `s(p)` or
`_spacing_value(T, s, p)`. So the missing piece is a **composing spacing**,
not a new pipeline stage.

**Design (Option B — bulk is a floor, not a per-surface parameter):**

- A new `SurfaceRefinement(surface_points; at_wall, layer_thickness)` spacing
  that grows monotonically from `at_wall` toward `+∞` with distance from its
  surface. This is `BoundaryLayerSpacing` with `bulk = ∞` — the per-surface
  spacing has *no far-field*; it only tightens near its surface.
- A `MinSpacing` combinator (or `Base.min` overload on `AbstractSpacing`)
  that evaluates all constituents at `p` and returns the finest. ~15 lines.
- The bulk floor is a plain `ConstantSpacing(bulk)` composed into the `min`.
  One place to set bulk, not N.

```julia
bnd = PointBoundary(mesh, h)
split_surface!(bnd, 75°)                     # :inlet, :wall, …
spacing = min(
    SurfaceRefinement(points(bnd[:inlet]); at_wall=0.5mm, layer_thickness=2mm),
    SurfaceRefinement(points(bnd[:wall]);  at_wall=2.0mm, layer_thickness=4mm),
    ConstantSpacing(5mm),                    # ← the one and only bulk floor
)
cloud = discretize(mesh, spacing)
```

**Why `min` is correct:** each per-surface spacing is monotonically
non-decreasing with distance from its surface, so it only *tightens* near its
source and relaxes away. The `min` picks the locally relevant constraint —
the finest applicable spacing wins, which is what physics demands (resolve the
smallest local feature). The `min` of continuous monotone functions is
continuous (with kinks where the active surface changes); no jumps.

**Why Option B over Option A (per-surface `bulk`):** Option A (today's
`BoundaryLayerSpacing` composed with a redundant floor) sets bulk in N+1
places and is error-prone — forget one `bulk=` and the wrong surface silently
sets the far-field. Option B makes bulk a property of the domain (one floor),
not of each surface. The cost is one small new type (`SurfaceRefinement`) or
equivalently a `bulk = nothing` mode on `BoundaryLayerSpacing`.

`BoundaryLayerSpacing` stays as-is for standalone use (one surface, self-
contained far-field). `SurfaceRefinement` is the composing variant.

### A2c. Smooth spacing field by default: enable `max_growth` on the golden path

The per-leaf spacing field is **piecewise constant**: one value stored at the
leaf center, returned for the entire leaf volume. Two adjacent leaves can have
different values, creating a density step at the leaf boundary.

**Mechanism — the gradient limiter (already implemented, off by default):**

`_gradient_limit_field` (`octree.jl:587`) computes a **g-Lipschitz envelope**
of the per-leaf field via min-plus relaxation over a k-NN graph of leaf
centers:

```
h[i] ← min(h[i], min_j h[j] + g·d_ij)
```

swept to a fixpoint. This enforces that spacing can't grow faster than `g`
per unit distance — step discontinuities become controlled ramps. It then
re-subdivides any leaf the limited field now out-resolves (the ramp made it
finer than the leaf can represent), re-limits, iterates to fixpoint (max 12
rounds). The sampler then reads the smoothed field through `_LeafSpacing` (a
`find_leaf` lookup — no KDTree queries during sampling).

The limiter only **reduces** spacing (min-plus propagates fineness outward,
never coarsens). This is correct — you never want spacing coarser than
prescribed — but means a surface with very fine `at_wall` bleeds fineness
outward at rate `g`. The `layer_thickness` parameter bounds the *intended*
refinement zone; `max_growth` bounds the *propagated* smoothing zone. Both
matter and are independent.

**Change:** `max_growth` defaults to `0.0` today (`octree.jl:102`), disabling
the limiter. The golden-path `discretize(mesh, h)` should produce a smooth
field by default — set `max_growth = 0.15` when constructing `Octree` inside
A2's mesh-first default: a Lipschitz cap of 0.15 corresponds to the classic
CFD growth ratio ≈ 1.15 and is the value the reference pipeline
(`examples/octree_boundary_layer.jl`) ships for the steep-boundary-layer
bunny case. Cost: one extra build step (the relaxation sweeps), cheap
relative to the full pipeline. Users who want the raw field pass
`max_growth = 0` explicitly.

**Why the within-leaf constant approximation is acceptable with the limiter
on:** leaf size is proportional to local spacing (`h_box ≤ alpha·h`), so where
spacing changes rapidly, leaves are smaller and steps are spatially smaller.
The worst-case within-leaf field variation is `~alpha·h·|∇h|`, and the
limiter bounds `|∇h| ≤ g`, so the variation is `~alpha·h·g` — small when `h`
is small (the refined zones where smoothness matters). The limiter's
re-subdivision step further tightens leaves where the ramp demands it.

The `min` combinator plugs in unchanged: the composed spacing is evaluated at
leaf centers during subdivision (Phase 1, expensive N-surface KDTree queries),
the limiter smooths the discrete field, and bridson reads the smoothed leaf
values (Phase 2, cheap lookups). No changes to `_LeafSpacing` or the limiter.

### A2d. Multi-mesh domains (tank + impeller + baffles) via normal-aware merge

Realistic geometries are rarely single meshes. The `examples/simpler_geometry`
case — a tank containing an impeller and baffles — is the canonical test: the
fluid domain is *inside the tank but outside the obstacles*. Today there is no
way to express this; `TriangleOctree` and `isinside` operate on one mesh.

**Key insight: normals already carry the inside/outside semantics.**

`TriangleOctree`'s signed distance (`triangle_octree.jl:525`) is:

```
s = dot(point - closest_pt, n)     # n = nearest face pseudonormal
s < 0  →  inside
s > 0  →  outside
```

The sign is determined entirely by normal direction — no Green's function, no
surface integral, no voting. If all mesh normals point **out of the fluid
domain** (out of the tank, into the impeller, into the baffles), a single
merged mesh's signed distance classifies the fluid domain correctly:

| Point location | Nearest face | Normal direction | `dot` | Fluid? |
|---|---|---|---|---|
| In tank, far from obstacles | Tank wall | Out of fluid | `< 0` | inside ✓ |
| In tank, near impeller | Impeller face | Into impeller (out of fluid) | `> 0` | inside ✓ |
| Inside impeller solid | Impeller face | Into impeller (out of fluid) | `< 0` | outside ✓ |
| Outside tank | Tank wall | Out of fluid | `> 0` | outside ✓ |

One merged mesh → one `TriangleOctree` → one `find_leaf` + signed distance.
No CSG composition, no N octrees, no `:inside`/`:outside` tags at query time.

**The only problem: normal orientation across disconnected components.**

`orient_normals!` (MST+DFS, Hoppe 1992) orients normals consistently *within
one connected component*. The tank, impeller, and baffles are three
disconnected components — MST+DFS orients each internally but can't know the
impeller's normals need flipping relative to the tank's.

STL files typically ship with outward-pointing normals per solid. So:
- Tank STL: normals point out of tank solid = out of fluid ✓ (correct as-is)
- Impeller STL: normals point out of impeller solid = **into fluid** ✗ (flip)
- Baffles STL: same as impeller ✗ (flip)

**Design — explicit-role merge with normal flip:**

```julia
domain_mesh = merge_meshes(
    "tank"     => tank,       # :container — normals as-is
    "impeller" => impeller,   # :obstacle  — normals flipped
    "baffles"  => baffles,    # :obstacle  — normals flipped
)
# domain_mesh is a SimpleMesh with all normals pointing out of the fluid

octree = TriangleOctree(domain_mesh; classify_leaves=true)
# isinside(p, octree) now correctly tests the fluid domain — one query
```

The `PointBoundary` and spacing compose as in A2b — each original mesh
contributes a named surface, `SurfaceRefinement` tightens near each, `min`
combines them:

```julia
bnd = PointBoundary(
    [
        :tank     => sample_surface(tank, h),
        :impeller => sample_surface(impeller, h),
        :baffles  => sample_surface(baffles, h),
    ];
    mesh = domain_mesh,        # carried per A3 → Octree built silently
)
spacing = min(
    SurfaceRefinement(points(bnd[:tank]);     at_wall=1mm,  layer_thickness=3mm),
    SurfaceRefinement(points(bnd[:impeller]); at_wall=0.5mm, layer_thickness=1mm),
    SurfaceRefinement(points(bnd[:baffles]);  at_wall=0.5mm, layer_thickness=1mm),
    ConstantSpacing(5mm),
)
cloud = discretize(bnd, spacing)   # named surfaces preserved in the cloud
```

Roles are explicit (`:container` / `:obstacle`), per the "explicit over
implicit" principle. Automatic containment-based inference (obstacle = bbox
inside container) is tempting but fragile for overlapping non-nested parts —
not worth the complexity.

**Scope additions (resolved outstanding issues O1/O2):**

- The `PointBoundary(pairs::Vector{Pair{Symbol, PointSurface}}; mesh =
  nothing)` constructor above **does not exist today** — add it (~5 lines +
  tests). It must promote mismatched machine types across surfaces
  (binary STL → Float32, ASCII → Float64) the way the 3-arg `PointCloud`
  constructor does (`_promote_mactype`, `cloud.jl:38-56`), or throw naming
  the mismatch.
- The multi-mesh path **cannot** use A2's `discretize(mesh, spacing)`
  one-liner: that method internally re-samples the boundary into one
  unnamed surface, discarding the names that the per-surface spacing and
  later boundary-condition assignment need. The multi-mesh golden path is
  the explicit named boundary with the merged mesh attached (as shown),
  then `discretize(bnd, spacing)` — A3 builds the Octree silently from
  `bnd.mesh`.

**Per-component orientation verification (the merge's own guard):**

The existing global signed-volume check (`triangle_octree.jl:352`) cannot
catch a forgotten flip in a merged mesh: correct orientation gives
`V_container − ΣV_obstacles > 0`, but an *unflipped* obstacle makes the total
even more positive — the global witness passes either way. `merge_meshes`
therefore verifies **per input mesh, before merging**: the `:container`'s
signed volume must be positive as given, each `:obstacle`'s must be positive
before its flip (negative after). `_signed_volume` already exists — run it
per component. A wrong role assignment or a pre-flipped input becomes a
constructor `ArgumentError` naming the offending mesh, not a silently
complemented domain (the failure mode of the two cavity corruptions).

The Green's function `isinside` path (`isinside.jl`) stays for backward
compatibility and for cases where you have a `PointCloud`/`PointBoundary` but
no mesh (no `TriangleOctree` can be built). It is O(n) per query and not
usable for large meshes; the golden path uses `TriangleOctree` (O(log n)).

**Normal inspection tools (ship alongside A2d):**

Deciding which meshes need flipping requires *seeing* the normals. Today there
is no tool to visualize mesh normals — `export_vtk` is point-cloud only,
`visualize` is point-only `meshscatter!`. Three inspection tools, all living
in `src/inspection.jl` (new file — see "Tool location" below):

1. **`export_mesh_vtk(filename, mesh; flip=())`** — writes a `.vtp` (polydata)
   with triangle cells and `normals` as cell data. ParaView's Glyph filter
   renders arrows. The `flip` keyword lets you test a re-orientation without
   mutating the mesh: load, see wrong arrows, add to `flip`, re-export,
   confirm. ~30 lines via WriteVTK.jl (already a direct dependency). Handles
   any mesh size — ParaView is comfortable with millions of triangles.

2. **`visualize(mesh; normals=true, normal_step=1, scale=0.0)`** — Makie
   extension method: `mesh!` for the surface + `arrows!` at face centers
   (every `normal_step`-th face, to stay interactive on large meshes). `scale`
   auto-sized to the local spacing when `0`. Lives in the Makie extension
   alongside the existing `visualize(cloud)` / `visualize(boundary)`.

3. **Orientation coloring** — `visualize(mesh; orientation=true)` and an
   `orientation` cell array in `export_mesh_vtk`: each connected component
   colored by the **sign of its component signed volume** (positive =
   normals outward, negative = inward). `_signed_volume` restricted to one
   component is the whole computation — no arrows to read; one glance
   answers "which meshes need flipping". This is the visual twin of the
   per-component merge guard above: same witness, shown instead of thrown.

**Tool location — `src/inspection.jl`:**

The package has a growing set of small, convenient inspection/diagnostic
tools: `suggest_spacing` (geometry probe), `geometry_info` (raw extents),
`metrics`/`spacing_metrics`/`spacing_fidelity_metrics` (quality), and now
`export_mesh_vtk` + `visualize(mesh; normals=…)`. Today these are scattered
across `spacing_guidance.jl`, `io.jl`, `metrics.jl`. They share a philosophy:
small, read-only, "what does my geometry look like?" probes that a user runs
*before* discretizing.

Collect them in one file — `src/inspection.jl` — so a reader finds all
pre-discretization diagnostics in one place. This is a pure move (Tier C
style): `suggest_spacing`, `geometry_info`, `export_mesh_vtk`, and the
`visualize(mesh)` dispatch move there; their exports and call signatures stay
identical. `metrics` and friends stay in `metrics.jl` (they're post-
discretization, different concern). The file opens with a header comment in
the `surface_sampling.jl` / `spacing_guidance.jl` house style.

### A3. Defaults policy: Octree is the silent default everywhere

`Octree` requires a mesh (`Octree(mesh; spacing)` / `Octree(::TriangleOctree)`
are the only constructors), and today the mesh is thrown away right after
`PointBoundary` construction — which is why the bare-boundary methods fell
back to `SlakKosec()`. Fix the root cause, not the symptom:

- **`PointBoundary` keeps a reference to its source mesh.** Today it holds
  only `surfaces` (`boundary.jl:13`) — every constructor drops the mesh on
  the floor. New optional field (`mesh::Union{Nothing, SimpleMesh}`),
  populated by `PointBoundary(filepath, unit)`, `PointBoundary(mesh)`, and
  `PointBoundary(mesh, spacing)`. A plain reference — no copy, no
  computation. Poisson-disk surface sampling needs the *spacing* but not an
  octree (`sample_surface` dart-throws on the triangles with the flat
  `_BridsonGrid`, `surface_sampling.jl:65`), so nothing octree-shaped is
  built at boundary time; hand-built and 2D boundaries carry `nothing`.
  **Serialization (resolves O3):** `save`/JLD2 never stores the
  `SimpleMesh` object (Meshes.jl topology structs are a serialization
  risk); it stores the same indexed data A1 defines — `vertices`,
  `triangles` (vertex-index triples), `len_unit` — all isbits,
  version-stable — and rebuilds the mesh on load. One format shared by
  boundary serialization and `TriangleIndex`. Round-trip test in PR 2.
- **Octree construction stays inside `discretize`, as today.** Each run
  builds two octrees: the spacing-independent `TriangleOctree` (triangle
  lists, pseudonormals, leaf classification) and the spacing-driven node
  octree that consumes it throughout — inherits its bounds, prunes
  subdivision to interior-capable boxes via its leaf classification,
  classifies every node leaf against it (`build_node_octree`,
  `spacing_criterion.jl:87`). Both are per-pipeline working structures; no
  caching layer. (A lazily-memoized `SourceGeometry` cell was considered
  and dropped: its payoff was `repel` reuse and repeat-discretize sweeps,
  and the direct pipeline is designed to never rely on `repel`.)
- **All three `discretize` methods default to Octree.**
  `discretize(mesh, spacing)` (A2) trivially;
  `discretize(bnd::PointBoundary{𝔼{3}}, spacing)` (`discretization.jl:40`)
  and `discretize(cloud::PointCloud, spacing)` (`discretization.jl:73` —
  today *also* silently SlakKosec) build `Octree(bnd.mesh; spacing)` from the
  carried mesh. No user-visible API change; existing scripts silently get
  the better algorithm. Flag in release notes: results change (for the
  better) for anyone who relied on the implicit SlakKosec.
- **Only a truly meshless boundary errors** — an Octree cannot be built
  without geometry, so the no-`alg` call on a `mesh === nothing` boundary
  throws, naming the fix. The message must be spacing-honest:
  `VanDerSandeFornberg` dispatches on `ConstantSpacing` only
  (`vandersande_fornberg.jl:13`), so for a variable spacing there is *no*
  mesh-free algorithm at all:

  > *"This boundary carries no mesh, so the default Octree algorithm cannot
  > be built. Recommended: construct the boundary from a mesh
  > (`PointBoundary(mesh)` / `PointBoundary(file, unit)`). With a
  > `ConstantSpacing` you may instead pass `alg = VanDerSandeFornberg()`;
  > variable spacings require a mesh — no mesh-free algorithm supports
  > them."*

- **SlakKosec is known broken** (wrong output). Keep the code and the
  `alg = SlakKosec(octree)` entry point for whoever repairs it, but: never
  selected implicitly, a `!!! warning` admonition in its docstring, an issue
  tracking the repair, and it is dropped from the error message's suggested
  alternatives above.
- 2D keeps `FornbergFlyer()` as the silent default (it is the only 2D
  algorithm) and the unconditional `@warn` (`discretization.jl:54`) is
  deleted. The worse wart: the 2D method **accepts `alg` and silently
  discards it** (`discretization.jl:57` hardcodes `FornbergFlyer()`).
  Passing a non-`FornbergFlyer` algorithm in 2D becomes an `ArgumentError`,
  not a warn-and-substitute.

### A4. A quiet library: gate the progress narration

`octree.jl`'s `_discretize_volume` and helpers contain ~20 bare
`print`/`println` calls ("Building node octree… done (0.02s)") that a library
user cannot switch off. One pointed edit:

- Add `verbose::Bool = false` to `discretize`/`Octree`'s pipeline, threading
  one flag; wrap the narration in `verbose && …` (or route through
  `@info`/a `_progress(io, verbose, msg)` one-liner).
- `@warn`s stay unconditional — they are diagnostics, not narration.
- `suggest_spacing(; verbose = true)` keeps its default: printing is that
  function's *purpose*. Document the asymmetry.

### A5. Docs restructured around the golden path

- **README**: the quickstart above, verbatim, near the top. One image.
- **guide.md**: pipeline reads import → probe → `discretize(mesh, h)` →
  (optional) `repel` → `export_vtk`. Other algorithms move to a clearly
  labelled "Choosing a different algorithm" subsection instead of being
  interleaved as peers of the default.
- **discretization.md**: Octree section first; the algorithm table gains a
  "How to select" column (`default everywhere` / `alg = VanDerSandeFornberg()`
  / …) **and a "Spacing support" column**: VDF and FF are
  `ConstantSpacing`-only (dispatch-enforced —
  `vandersande_fornberg.jl:13`, `fornberg_flyer.jl:13`); **Octree is the
  only algorithm for variable spacings** (`BoundaryLayerSpacing`,
  `LogLike`, the A2b combinators) — stated as a plain sentence above the
  table, not left implicit in a cell. VDF/FF sections stay, framed as
  alternatives with their genuine niches (VDF: fast *uniform* fill on
  simple shapes; FF: 2D). SlakKosec's section states plainly that it is
  **currently broken and kept for repair** — no niche framing until it
  works again.
- "One mesh, shared everywhere" is stated once, prominently: `import_mesh`
  → the same object feeds `PointBoundary`, `Octree`, `TriangleOctree`,
  `repel`.

### A6. Message and keyword consistency sweep

One pass over user-facing surfaces, no logic changes:

- Error messages follow one pattern: *what failed → why → the line to run*
  (several already do — `"Rebuild with: TriangleOctree(mesh;
  classify_leaves=true)"` is the house style; a few one-word
  `ArgumentError`s are not).
- **Unitless-spacing guard methods**: `discretize(_, spacing::Real)` (and the
  other spacing entry points) today die with a raw `MethodError`. Add
  `::Real` methods that throw an `ArgumentError` naming the fix:
  *"spacing must carry units — write `0.5u"mm"` (or build a spacing type:
  `ConstantSpacing(0.5u"mm")`)"*. Same for `at_wall`/`layer_thickness`
  keywords where a bare number can reach a constructor.
- Keyword vocabulary is uniform across algorithms: `max_points`,
  `verbose`, `factor`/`bridson_factor` (pick one), `min_ratio`/
  `node_min_ratio` documented side by side.
- Export list in `WhatsThePoint.jl` grouped by concern with one-line
  comments (types / pipeline / algorithms / spacings / analysis / io).

---

## Tier B — Unification (mechanical, behavior-preserving except named bugfixes)

### B1. Unit-correctness bugfixes (the only behavior changes in this tier)

Blind `ustrip`s that mis-handle a spacing in a different unit than the
cloud — found during the rebase work, fixes exist on the archived branch:

- `BoundaryLayerSpacing` callable: sigmoid δ must convert to the distance's
  unit.
- `repel`'s CV-monitor spacings and `_deposit_escaped!` threshold:
  `ustrip(len_unit, …)`.
- `spacing_fidelity_metrics`: strip in the cloud's length unit.
- **Octree exits hardcode meters**: `_discretize_volume` builds its returned
  volume as `Point(pt...)` from raw magnitudes (`octree.jl:886,951`), and
  `Cartesian` labels bare numbers `m` unconditionally
  (`addunit.(coords, m)`, CoordRefSystems `cartesian.jl:36`). Any mesh
  imported in a non-meter unit therefore produces meter-labeled
  wrong-magnitude volume points and a CRS-mismatch throw at `PointCloud`
  assembly. Unexposed today only because every test and example uses
  meters. Fix: re-attach the recorded `len_unit` (A1) at the exit —
  `Point((pt .* len_unit)...)`.

Ship as one PR titled "fix: unit-aware stripping", each fix with a
mixed-unit regression test (`spacing = ConstantSpacing(5u"mm")` on a cloud
in `m`, plus a full Octree `discretize` on a `u"mm"`-imported mesh).

### B2. One conversion idiom (the minimal seam — not the full boundary)

The codebase has ~48 `ustrip` calls across 12 files and five variants of the
same point→`SVector` conversion (`_raw_point`, `_extract_vertex`, three
inline forms). Minimal unification, no architecture:

- One small file (`src/conversions.jl`, ~40 lines): `length_unit(x)`,
  `raw_point(T, p, len_unit)`, `raw_scalar(T, x, len_unit)` — unit-aware
  (`ustrip(len_unit, …)`), machine-type-explicit.
- Mechanical call-site replacement. After A1, `_extract_vertex` runs only at
  `TriangleIndex` build time, so the hot-path caveat is gone — the point is
  one *implementation*, not one *name*.
- This is deliberately *not* the `numerical_spacing`/closure program from
  the deferred appendix; per-call stripping survives for now. It just makes
  every strip correct and greppable.

### B3. Write the conventions down (CLAUDE.md, ~15 lines)

- **Seam policy**: typed spatial structures (`SpatialOctree{_,T}`,
  `_BridsonGrid{T}`) convert foreign-precision queries at their public entry
  points; internal calls must already be in `T`. (The converting
  `isinside(::SVector{3,<:Real}, octree)` methods on the archived branch are
  the pattern.)
- **Literal discipline** in `T`-generic code: `T(…)`/`oftype`/`typemax(T)`,
  never bare `0.5`/`Inf`.
- **Quiet-by-default** and the error-message pattern from A6.

### B4. Test hygiene

- Promote `rebuild(T, mesh)` from `test/octree.jl` into `testsetup.jl`;
  precision-sensitive tests state their `T` instead of inheriting Float32
  from the STL format (half of one debugging day traced to this).
- Add the mixed-unit regression tests from B1.

### B5. Example and doc-script refresh

`docs/generate_images.jl` and `examples/` run against the current API
(unit-required imports from #103); regenerate the algorithm-comparison image
with Octree included. Examples are the first thing an outside reader runs —
they must work verbatim.

---

## Tier C — File readability (pure moves, one PR per file)

Current worst offenders: `octree.jl` (952), `spatial_octree.jl` (799),
`repel.jl` (626), `triangle_octree.jl` (613). Splitting is mechanical if the
rule is absolute: **the diff contains zero logic changes** (`git diff
--color-moved` shows only moves; Runic clean).

Priority order by reader value:

1. `discretization/algorithms/octree.jl` → `octree.jl` (struct +
   constructors + `_discretize_volume`, ~300), `bridson.jl` (grid +
   sampler), `spacing_field.jl` (gradient limiter + `_LeafSpacing`),
   `leaf_sampling.jl` (LeafGroup machinery).
2. `repel.jl` → `repel.jl` (public methods + docstrings, ~250),
   `repel_relax.jl` (`_relax!` + per-iteration helpers),
   `repel_walls.jl` (octree constraint/projection/deposition/cull).
3. `spatial_octree.jl` → structure/traversal/balance/classification —
   only if appetite remains; it is internal and less read by outsiders.

Each split file opens with a 3–6 line header comment saying what lives here
and what deliberately doesn't (the existing headers in `surface_sampling.jl`
and `spacing_guidance.jl` are the house style — keep that voice).

---

## Deferred (post-JCon appendix) — the architectural program

Unchanged in substance from v1, parked: the single strip-once boundary
(`numerical_spacing(s, len_unit, ::Type{T})` returning `T`-valued closures,
killing per-call `_spacing_value`) and the `Numerics` submodule with
compiler-enforced purity. (The `TriangleIndex{T}` triangle cache was part of
this program in v2; v3 promotes it to A1.) The archived
`separate_numerics_from_unitful` branch (tip `25b25ba`, pushed) is the
reference implementation and the worked example of every precision-seam
problem the boundary must handle; do not delete it until that program lands
or is abandoned deliberately.

Also parked: porting the constructive side (`sample_surface`, face-center
`PointBoundary(mesh)`, `_signed_volume`, `merge_meshes`,
`visualize(mesh)`) from `SimpleMesh` onto A1's indexed
`(vertices, triangles)` data — they only iterate triangles. Once done, the
`SimpleMesh` becomes an import-time-only object (no longer kept in memory
or in any struct) and `TriangleOctree` can drop its `{M, C}` type
parameters.

Nothing in Tiers A–C blocks or is thrown away by this program: A2/A3 are its
UX PRs, A1 and B1 its performance and bugfix PRs, B2 a strict subset of its
boundary, C's file splits are the moves it needs anyway.

## Suggested PR sequence

| # | Content | Tier | Risk |
|---|---------|------|------|
| 1 | `TriangleIndex{T}` build-once triangle cache + benchmark | A1 | none (behavior-preserving; perf win) |
| 2 | Mesh-first `discretize` + mesh ref on `PointBoundary` + Octree-everywhere defaults + 2D alg fix | A2, A3 | behavior change: implicit-SlakKosec calls now get Octree (flagged) |
| 2b | `max_growth` default on golden path (smooth field by default) | A2c | none (behavior improves; `max_growth=0` opts out) |
| 2c | `SurfaceRefinement` + `MinSpacing` combinator | A2b | none (additive; new types) |
| 2d | `merge_meshes` + per-component orientation guard | A2d | none (additive; new function) |
| 2e | Normal inspection tools (`export_mesh_vtk`, `visualize(mesh)`) + `src/inspection.jl` consolidation | A2d | none (additive + pure moves) |
| 3 | `verbose` gating of octree narration | A4 | none |
| 4 | Unit-aware stripping bugfixes + regression tests | B1, B4 | behavior fix |
| 5 | Conversion idiom unification | B2 | none (mechanical) |
| 6 | CLAUDE.md conventions + message/keyword sweep + export grouping | A6, B3 | none |
| 7 | Docs/README/examples golden path | A5, B5 | none |
| 8+ | File splits, one per PR | C | none (pure moves) |

PR 1 is independent of everything below it — it touches only the triangle
query paths and can land (or be reviewed) in parallel with PR 2. PR 2b–2e can
proceed independently of PR 2 once the `discretize(mesh, spacing)` signature
lands — they are additive types, a default-value change, and new inspection
tools, not breaking. Sequence them after PR 2 so the golden path they enhance
already exists. PR 2d (merge) should land before 2e (inspection tools) so the
normal-visualization tools can be tested against the merged multi-mesh case.

## Verification (every PR)

- Full suite: `julia --project -e 'using Pkg; Pkg.test()'` — including the
  Float32 pipeline tests (the precision contract) at every step.
- Runic on touched files (CI enforces Runic 1.x).
- PR 1: before/after benchmark (`isinside` batch + one `repel` iteration on
  `bifurcation.stl`) in the PR description.
- PRs 2–3: docstrings/docs updated in the same PR (user-facing surface).
- PR 5: `grep -rn "ustrip" src/` hits only `conversions.jl` and the
  legitimately-unitful API/display code (`io.jl`, `metrics.jl` display,
  `spacing_guidance.jl` printing).
- Tier C PRs: `git diff --color-moved=dimmed-zebra` reviewed as moves-only;
  public API surface (exports) unchanged.

## Outstanding issues

### O1. `Vector{Pair{Symbol, PointSurface}}` constructor — **RESOLVED**, folded into A2d

The constructor didn't exist; A2d's scope now includes it explicitly
(`PointBoundary(pairs; mesh = nothing)`, with mactype promotion across
surfaces per the 3-arg `PointCloud` precedent).

### O2. Multi-mesh loses named surfaces through A2's one-liner — **RESOLVED**, folded into A2d

The A2d example is corrected: multi-mesh goes through the explicit named
boundary with the merged mesh attached (O1's `mesh` keyword), then
`discretize(bnd, spacing)` — A3's silent Octree default supplies the
algorithm; no new `discretize` overload needed.

### O3. JLD2 serialization of `SimpleMesh` — **RESOLVED**, folded into A1 + A3

Never serialize the `SimpleMesh`. A1's `TriangleIndex` is designed as the
indexed representation — `vertices::Vector{SVector{3,T}}`,
`triangles::Vector{NTuple{3,Int32}}` (vertex indices; connectivity, needed
to rebuild pseudonormals), `len_unit` — all isbits, version-stable. A3
serializes exactly that and rebuilds the mesh on load. Side discovery while
resolving this: the Octree exits hardcode meters
(`Point(pt...)` at `octree.jl:886,951`) — added to B1's bug list. Fully
mesh-free memory (porting `sample_surface` etc. off `SimpleMesh`) is parked
in the deferred appendix.

### O4. `alpha` default — **RESOLVED: keep `2.0`** (benchmark 2026-07-12)

Reference pipeline (bunny, 69,664 triangles, 146,657 shared boundary points,
`max_growth = 0.15`, seeded), `alpha = 1.0` vs `2.0`:

| | `1.0` | `2.0` |
|---|---|---|
| node octree leaves | 2,515,318 | 369,349 |
| total wall time | 75.5 s | 60.6 s |
| volume points | 1,101,730 | 1,104,278 |
| spacing CV | 0.0496 | 0.0532 |
| sep/h_min | 0.751 | 0.751 |
| coordination | 15.1 | 15.2 |
| worst d_NN ratio | 1.539 | 1.596 |

Quality parity within noise; `2.0` is ~20 % faster end-to-end with a 6.8×
smaller node octree (`1.0` spends 2.3 leaves per generated point). The
gradient limiter's re-subdivision refines the coarse scaffold exactly where
fineness matters, which is why quality barely moves. **Decision:** the
constructor default `alpha = 2.0` is the golden-path default. The reference
example keeps `alpha = 1.0`, documented as the explicit maximum-fidelity
setting; A5's docs present `alpha` as *the* fidelity-vs-cost knob. Caveat
recorded: one geometry, one spacing config — a steeper wall layer or
thin-feature part could widen the quality gap.
