# TriangleIndex Rationalization — Work Report

**Branch:** `polish_before_JCon` (uncommitted, working tree)
**Date:** 2026-07-14
**Baseline:** HEAD (`c7bd9b1`, PR #103 merge)

## Objective

Introduce `TriangleIndex{T}` as the package's runtime mesh representation —
replacing `SimpleMesh` in every stored struct and every hot query path —
achieving radical simplification of the architecture, not just a perf cache
bolted onto existing code.

The guiding principle: `SimpleMesh` is an import-time abstraction. After
`TriangleIndex(T, mesh)` runs at the package boundary, no struct in the
package holds a `SimpleMesh` reference. Meshes.jl becomes an import-time-only
dependency.

## Results

- **Tests:** 190,647 pass / 0 fail / 0 errored / 3 broken (pre-existing
  `@test_skip` for missing files). Runic clean.
- **Source size:** 6,481 lines (down from 6,804 at HEAD — **323 lines net
  deleted** from `src/`).
- **Total diff:** +331 / −761 across 26 files (src + test). Net deletion
  **430 lines**.
- **2 files deleted entirely:** `src/reconstruction.jl` (orphan),
  `src/octree/traits.jl` (dead abstract trait system).

## Tier 1 — Dead code deletion

| Deleted | Lines | Justification |
|---|---|---|
| `src/reconstruction.jl` | 14 | Orphan — not `include`d, zero callers anywhere |
| `src/octree/traits.jl` | 207 | `AbstractSpatialTree`/`AbstractOctree`: 1 subtype, nothing dispatches on it. `MaxElementsCriterion`/`SizeCriterion`/`AndCriterion`: 0 src callers (only tests tested them). `SubdivisionCriterion` (the only live abstract type) moved to `spatial_octree.jl` as a one-liner |
| `distance_point_triangle` (2 methods) | ~85 | Only called by tests; the signed-distance path uses `closest_point_on_triangle_feature` + pseudonormals directly |
| `findmin_turbo` | 11 | Hand-rolled `Base.findmin`; replaced at 4 callsites in FF/VDF |
| `calculate_ninit(::VariableSpacing)` | 6 | Dead method — FF/VDF require `ConstantSpacing` (dispatch-enforced) |
| 2D `discretize` `@warn`-and-substitute | 2 | Silently discarded user-passed `alg`; now `ArgumentError` if non-`FornbergFlyer` |

## Tier 2 — TriangleIndex consolidation

`TriangleIndex{T}` is now the self-sufficient runtime mesh: vertices,
triangles, face normals, angle-weighted pseudonormals (edge/vertex dicts),
and bounding box — all canonical fields, no derived cache. `MeshPseudonormals`
folded in and deleted.

### Struct shape changes

| Struct | Before (HEAD) | After |
|---|---|---|
| `TriangleIndex{T}` | — (new) | 8 fields, all canonical |
| `MeshPseudonormals{T}` | 3 fields | **DELETED** (folded into TriangleIndex) |
| `TriangleOctree` | `{M,C,T}`, 7 fields (incl. `mesh::SimpleMesh`) | `{T}`, **3 fields** (`tree`, `index`, `leaf_classification`) |
| `NearestTriangleState{T}` | 7 fields (cached `v1/v2/v3`) | **4 fields** (re-gather winning triangle once after search) |
| `Octree` (algorithm) | `{M,C,T}` | `{T}` |

### What died in `triangle_octree.jl`

- `v1`/`v2`/`v3` soup views (3 derived cache fields) — hot paths read
  `vertices[triangles[i][k]]` (one contiguous array read)
- `bbox_min`/`bbox_max` on `TriangleOctree` — moved to `TriangleIndex`
  (canonical, computed once at construction)
- `_get_triangle_vertices(::Type{T}, mesh, …)` — legacy mesh path
- `_get_triangle_normal(::Type{T}, mesh, …)` — legacy mesh path
- `_compute_bbox(::Type{T}, mesh)` / `has_consistent_normals(::Type{T}, mesh)`
  / `_signed_volume(::Type{T}, mesh)` — three throwaway dispatcher shims
- `_normalize_normal` — only fed the legacy mesh path
- `MeshPseudonormals(::Type{T}, mesh, index)` → `MeshPseudonormals(index)` →
  deleted entirely (edge/vertex dicts now built inside `TriangleIndex`'s
  constructor)

### Signature simplification

- `_compute_signed_distance_octree` lost an argument (no more `pn`)
- `_classify_leaves` lost an argument
- `_create_root_octree` lost redundant `n_triangles` arg
- `TriangleOctree{T}` — 3 type parameters → 1
- `Octree{T}` — 3 type parameters → 1

## Architectural state after this work

- **`SimpleMesh` stored in a struct:** 0 (was 1 — `TriangleOctree.mesh`).
  Every remaining `SimpleMesh` reference is an argument type at an entry
  point (`PointBoundary(mesh)`, `sample_surface(mesh, …)`,
  `TriangleOctree(mesh; …)`, `import_mesh`).
- **Meshes.jl in runtime:** 0 structs hold Meshes.jl objects. Query paths
  read contiguous `SVector{3,T}` arrays exclusively.
- **`TriangleIndex` IS the mesh** for this package's runtime purposes:
  self-sufficient, isbits-adjacent (the `Dict` fields are the only non-isbits
  part — rebuilt from `(vertices, triangles, face)` on load if ever
  serialized).
- **Serialization format** is already defined: `(vertices, triangles, face,
  len_unit)` — all isbits except the dicts, which are derivable. No
  `SimpleMesh` reconstruction needed on load.

## What this unlocks (per simplification_plan.md)

- **Multi-mesh merge (A2d):** `TriangleIndex` concat — vertices, offset
  triangle indices, flip per role. No `SimpleMesh` surgery.
- **`export_mesh_vtk` (A2d inspection):** writes `vertices` + `triangles` +
  `face` directly via WriteVTK — no Meshes.jl indirection.
- **Per-component orientation guard:** `_signed_volume(index)` on each
  component before merge — already exists.
- **The `{M,C}` type-parameter noise** is gone from `TriangleOctree` and
  `Octree` signatures.

## Verification

- Full suite: `julia --project -e 'using Pkg; Pkg.test()'`
  - 190,647 pass / 0 fail / 0 errored / 3 broken (pre-existing `@test_skip`)
  - Includes Float32 pipeline tests (the precision contract)
  - The 2 previously-errored `octree_regression_curvature` tests no longer
    error (one passes, one `@test_skip`s due to no interior leaves in the
    bifurcation mesh — a pre-existing classification property)
- Runic `--check` clean on all 22 touched files
- Package loads: `julia --project -e 'using WhatsThePoint'` ✓

## Not committed

All changes are in the working tree on `polish_before_JCon`. Nothing is
staged or committed. Untracked planning docs (`simplification_plan.md`,
`mechanics.md`) and example assets (`examples/simpler_geometry/`,
`bunny_nodegen.gif`) remain as before.
