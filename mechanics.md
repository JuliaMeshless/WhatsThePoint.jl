# WhatsThePoint.jl — Package Mechanics

*Written 2026-07-12 from a source read-through, as a shared reference for
Davide and Claude. Every claim carries a `file:line` anchor to the code it
came from. When the mechanics change, change this document in the same PR.
Companion to `simplification_plan.md` (what we intend to change); this file
records what IS.*

---

## 1. The one-paragraph picture

WhatsThePoint turns a surface mesh into a point cloud suitable for meshless
PDE solvers. The direct pipeline — the golden path — is: **import** a mesh
with an explicit unit → **probe** it (`suggest_spacing`) → **sample the
boundary** with graded Poisson-disk (`PointBoundary(mesh, spacing)`) →
**fill the volume** with the `Octree` algorithm (spacing-driven node octree +
global Bridson front seeded from the boundary) → **check quality**
(`metrics`, `spacing_fidelity_metrics`) → **export**. The pipeline is
designed to produce good clouds *on its own*; `repel` is optional polish,
never a required stage.

## 2. Data model

Composition, bottom-up (include order in `WhatsThePoint.jl:34-101` is the
dependency order):

| Type | Storage | File |
|---|---|---|
| `SurfaceElement{M,C,N,A}` | one `(point, normal, area)` triple | `surface.jl:13` |
| `PointSurface{M,C,T,G}` | `StructVector` of SurfaceElements + topology | `surface.jl:33` |
| `PointBoundary{M,C}` | `LittleDict{Symbol, PointSurface}` — named surfaces, **nothing else** | `boundary.jl:13` |
| `PointVolume{M,C,T,V}` | plain vector of `Point`s + topology (`V` generic — GPU-ready) | `volume.jl:12` |
| `PointCloud{M,C,T}` | `boundary + volume + topology` | `cloud.jl:11` |

Facts that are easy to get wrong:

- **`PointBoundary` holds no mesh.** Every constructor extracts
  points/normals/areas and discards the source (`boundary.jl:48-55`).
  (The plan's A3 adds an optional mesh reference.)
- **All four types are immutable structs** with a functional API; `!`
  functions (`split_surface!`, `orient_normals!`) mutate *contents* of the
  surfaces dict / StructArrays, not the structs.
- `PointCloud(boundary)` **deepcopies** the boundary (`cloud.jl:24`).
- The 3-arg `PointCloud` constructor **promotes mismatched machine types**
  (Float32 boundary + Float64 volume → Float64 cloud) and throws on any
  other CRS mismatch (`cloud.jl:38-56`).
- Topology (`KNNTopology`/`RadiusTopology`) is optional, built eagerly by
  `set_topology`, local indices on surface/volume, global on cloud.

## 3. Units and machine type — the two currencies

Two independent axes, both threaded through everything:

- **Units (Unitful):** the user-facing currency. Every API length is a
  `Unitful.Length`. Mesh files carry no unit metadata, so `import_mesh`
  **reinterprets** raw numbers in the user's unit — a stored `46` becomes
  `46 mm`, no conversion (`io.jl:15-31`). `geometry_info` probes raw extents
  first.
- **Machine type `T = CoordRefSystems.mactype(C)`:** the internal currency.
  Numerical kernels work on raw `SVector{3,T}`; **binary STL loads as
  Float32**, so `T` is often Float32 unless the caller rebuilds the mesh
  (the `rebuild(T, mesh)` helper in `test/octree.jl:392`).

The crossing point between currencies is
`_spacing_value(T, spacing, p) = T(ustrip(spacing(Point(p...))))`
(`spacing_criterion.jl:38`) — **note the bare `ustrip`: it assumes the
spacing's unit matches the coordinates' unit.** This assumption is violated
in several places (mixed-unit bugs; `simplification_plan.md` B1). ~48
`ustrip` calls exist across 12 files; five variants of the point→`SVector`
conversion.

## 4. The spacing system

Contract: **a spacing is a callable `s(p::Point) -> Unitful.Length`**. Every
consumer goes through `s(p)` or `_spacing_value(T, s, p)`.

| Type | Formula | Notes |
|---|---|---|
| `ConstantSpacing{L}` | `Δx` | unit enforced by type (`spacings.jl:35`) |
| `LogLike` | `base·x/(a+x)`, `a = base·(2−g)` | distance to nearest boundary point via KDTree (`spacings.jl:49-70`) |
| `BoundaryLayerSpacing` | `at_wall + (bulk−at_wall)·σ(d)`, sigmoid centered `δ/2`, width `δ/6` | KDTree; **blind `ustrip` on `d` and `δ`** — unit bug when spacing unit ≠ coordinate unit (`spacings.jl:121-135`) |

Both variable spacings collapse **all** boundary points into one aggregate
KDTree — surface identity from `split_surface!` is not usable for
per-surface refinement today (plan A2b adds combinators).

`suggest_spacing` (`spacing_guidance.jl`) is the step-0 probe: reports
`h_ceiling` (coarsest spacing with any interior, `L_min/(2·bridson_factor)`),
`h_baseline` (≈10 points across the thinnest extent, clamped to
`h_ceiling/2`), `h_fine` (`h_baseline/2`) — all returned unitful
(`spacing_guidance.jl:114-130`).

## 5. The octree layer — three things called "octree"

This distinction is the most important one in the package. Confusing them
produces wrong architecture conclusions.

### 5a. `SpatialOctree{E,T}` — the generic container (`octree/spatial_octree.jl:57`)

Flat-array octree on an integer coordinate system `(i,j,k,N)` (level-`N`
integer coords; box center recoverable in O(1)). Fields: `parent`,
`children` (8 per box), `coords`, `element_lists`, `num_boxes::Ref`.
Provides `subdivide!` (line 285), `find_leaf` (328, O(log n)),
6-directional neighbor finding, and **2:1 balance enforcement**
(`balance_octree!`, line 617). It is a mutable working structure.

### 5b. `TriangleOctree{M,C,T}` — the geometry authority (`octree/triangle_octree.jl:31`)

A `SpatialOctree{Int}` whose element lists are triangle indices, plus the
mesh itself, the exact mesh bbox, **angle-weighted pseudonormals**
(Bærentzen & Aanæs 2005 — `sign(dot(p−cp, n_feature))` is provably correct
for watertight consistently-wound meshes; this *replaced* an older
distance-weighted sign vote, `triangle_octree.jl:4-17`), and optional
per-leaf classification (`LEAF_INTERIOR/EXTERIOR/BOUNDARY`).

- Subdivision criterion: triangle/vertex density. **The TriangleOctree's
  subdivision is spacing-independent** (spacing drives the separate node octree,
  §5c).
- Construction guard: the **global signed volume** must be positive
  (inside-out mesh → `ArgumentError`, `triangle_octree.jl:349-362`) — an
  O(n) exact witness for closed meshes, gated on `classify_leaves`.
- Signed distance query: nearest-triangle search down the tree
  (`_update_closest_triangle!`, line 436) then feature pseudonormal for the
  sign (line 509-530). **There is no ray casting anywhere in the package.**
- Every triangle touched by any query is re-extracted from the Meshes.jl
  object model per touch (`_get_triangle_vertices`, line 125:
  `mesh[i]` → `vertices` → `Meshes.to` + `ustrip`×3). This is the
  `TriangleIndex{T}` target (plan A1). Pseudonormals, by contrast, are
  already precomputed once at construction (line 374).

### 5c. The node octree — per-discretization, spacing-driven (`octree/spacing_criterion.jl`)

A fresh `SpatialOctree{Int}` built by **every** `discretize` run via
`build_node_octree(triangle_octree, spacing, alpha, node_min_ratio)`
(line 87). Subdivides while `h_box > alpha·h(center)` (`SpacingCriterion`,
line 44), down to `absolute_min`. It **consumes the TriangleOctree
throughout**:

- inherits its root bounds (line 90-91),
- prunes subdivision to boxes that may contain interior, by querying the
  triangle octree's leaf classification (`_box_may_contain_interior`,
  line 106),
- after building + 2:1 balancing, classifies **every node leaf** against the
  TriangleOctree (`classify_node_octree`, line 189).

So: TriangleOctree = geometry, reusable per mesh, spacing-independent.
Node octree = resolution scaffold for one `(spacing, alpha)`, rebuilt per
run **by design** — that rebuild is essential work, not waste.

## 6. The `Octree` discretization pipeline, end to end

`Octree` the *algorithm struct* (`discretization/algorithms/octree.jl:84-92`)
wraps a TriangleOctree plus knobs: `boundary_oversampling`, `placement`
(default `:bridson`), `alpha` (leaf size ≤ `alpha·h`; constructor default
2.0, **reference example uses 1.0**), `node_min_ratio` (auto from
`h_min/alpha/extent` when spacing is Constant/BoundaryLayer —
`_extract_min_spacing`, line 218-227 — else falls back to the geometric
ratio), `bridson_factor` (0.75), `max_growth` (default 0 = limiter off;
**reference example uses 0.15 ≈ CFD growth ratio 1.15**).

`_discretize_volume` (line 816) runs, in order:

1. **Coarse-spacing guard** (`:bridson` only): clamps a spacing too coarse
   to host any interior, loudly, *before* the node octree since its
   resolution is spacing-driven (line 830-835; logic in
   `spacing_guidance.jl`).
2. **Build node octree** (`build_node_octree`, §5c) and **classify** its
   leaves against the TriangleOctree.
3. **Gradient limiter** (only when `max_growth > 0`;
   `_apply_gradient_limit`, line 667): computes a g-Lipschitz envelope of
   the per-leaf spacing field by min-plus relaxation over a k-NN graph of
   leaf centers (`h[i] ← min(h[i], h[j] + g·d_ij)` to fixpoint,
   `_gradient_limit_field`, line 587). The limiter only *reduces* spacing.
   Any leaf the limited field now out-resolves (`box_size > alpha·h`) is
   re-subdivided, re-balanced, re-classified, re-limited — fixpoint
   iteration, max 12 rounds, warn on non-convergence (line 680-707).
   Downstream reads the limited field through `_LeafSpacing`: a
   `find_leaf` lookup with fallback to the raw spacing (line 630-655) — no
   KDTree queries during sampling.
4. **Auto `max_points`** when unset: the spacing integral
   `Σ box_volume/h³` over non-exterior leaves (line 861-864).
5. **Volume fill.** Default `:bridson` (line 868-887): one **global**
   graded Poisson-disk front — boundary points are inserted as seeds that
   occupy the background grid (so volume keeps wall distance) but are not
   returned; darts from a random active point at radius `r·(1+rand)`;
   acceptance = inside test (leaf classification, exact signed-distance
   only in `LEAF_BOUNDARY` leaves — `_bridson_inside`, line 467) +
   separation `‖xᵢ−xⱼ‖ ≥ min(rᵢ,rⱼ)` with `r = factor·h(x)` via the flat
   uniform `_BridsonGrid` hashed at `r_min`. Runs to front saturation or
   `max_points`; on cap, a 200-dart probe distinguishes benign saturation
   from real truncation before warning (line 786-807). Zero interior
   points after guarding → error, never a silent empty cloud (line 879).
   Legacy placements (`:random/:jittered/:lattice`) instead allocate
   per-leaf point budgets by weighted volume, oversample near-surface
   leaves, filter by `isinside`, and fill deficits from interior leaves —
   interior leaves are trusted without per-point inside checks because
   classification demotes any leaf overlapping a boundary leaf
   (line 889-948).
6. Returns a `PointVolume`; `discretize` wraps it with the (deep-copied)
   boundary into a new `PointCloud` with `NoTopology`.

## 7. Boundary sampling (`surface_sampling.jl`)

`sample_surface(mesh, spacing)` = the surface counterpart of the volume
Bridson pass, **sharing `_BridsonGrid` but not any octree**: area-weighted
random triangle pick, uniform point in the triangle, accepted under the same
`min(rᵢ,rⱼ)` criterion with `r = factor·h(x)`. Runs to saturation
(`stall_limit` consecutive rejections). Samples carry the parent triangle's
normal; areas preserve total mesh area, distributed ∝ `r²`. Separation is
3D Euclidean — opposite faces of a sub-`h` thin wall block each other
(header, lines 1-14). `PointBoundary(mesh, spacing)` wraps it (line 110).
The alternative `PointBoundary(mesh)` takes **face centers** — density
follows the tessellation, not a prescribed spacing.

## 7b. The `_BridsonGrid` — shared Poisson-disk accelerator (`octree.jl:388`)

Both the volume Bridson pass (§6) and the surface sampler (§7) need
Poisson-disk separation checks: `‖xᵢ−xⱼ‖ ≥ min(rᵢ, rⱼ)` for every candidate
against every accepted point. A KDTree would be O(log n) per query but
rebuilds are expensive in a growing front; instead both paths use a flat
uniform background grid — `_BridsonGrid{T}` — that gives O(1) average-case
separation checks with no tree at all.

Structure (`octree.jl:388-394`): a 3D array of cells, each cell size
`h_min/√3` (so at most one point fits per cell diagonal for the finest disk).
Cells store a singly-linked list via `head` (per-cell, most recent point
index) + `nxt` (per-point, link to previous point in same cell) — `Int32`
throughout, so the grid is ~8 bytes per point plus the bucket array.

Construction (`octree.jl:396-409`): `h_min` is the finest disk radius in the
domain (`f · min spacing over non-exterior leaves` for volume,
`f · min spacing over all triangle centers` for surface). A memory guard
coarsens the cell size if the domain is huge relative to `h_min` (cap:
2²⁷ cells) — correctness is unaffected (buckets hold multiple points), only
the neighbor scan widens slightly.

Insertion (`_grid_insert!`, line 425): O(1) — compute cell index, push-link.
Separation check (`_bridson_separated`, line 439): scan the
`(2m+1)³`-cell neighborhood of the candidate's cell, where
`m = ceil(r_c / cell)` — only cells within `r_c` can hold a rejecting point.
Each cell's linked list is walked; `‖c − q‖² < min(r_c, r_q)²` rejects.
Accepted candidates are inserted immediately, so the grid reflects the live
front.

This is why neither the volume nor the surface sampler needs a KDTree during
point generation: the grid handles all separation checks, and the octree (or
triangle pick) handles only the inside/domain test.

## 8. Inside/outside — three mechanisms, no rays

| Dimension / structure | Mechanism | File |
|---|---|---|
| 2D | winding number (points must form an ordered loop) | `isinside.jl:7-66` |
| 3D, no octree | Green's-function surface sum (O(n) per query; needs normals+areas) | `isinside.jl:90-104` |
| 3D, `TriangleOctree` | cached leaf classification; exact pseudonormal signed distance only in boundary leaves | `triangle_octree.jl:567-596` |

## 9. The other algorithms

- **`VanDerSandeFornberg`** (3D): advancing-front sphere packing over a 2D
  shadow-plane grid, layer by layer in z. **Dispatch requires
  `ConstantSpacing`** (`vandersande_fornberg.jl:13`).
- **`FornbergFlyer`** (2D): same idea one dimension down (height field over
  x). Also **`ConstantSpacing` only** (`fornberg_flyer.jl:13`). The 2D
  `discretize` method warns unconditionally and silently ignores a
  user-passed `alg` (`discretization.jl:54-57`).
- **`SlakKosec`** (3D): candidate points on spheres around accepted points,
  accepted if inside + separated; optional TriangleOctree acceleration.
  **Currently broken (wrong output — Davide, 2026-07-12).** Still the
  silent default of `discretize(bnd, spacing)` and
  `discretize(cloud, spacing)` (`discretization.jl:40,73`) — the plan's A3
  changes that.

## 10. `repel` — optional polish, by design

Position: the direct pipeline must produce good clouds alone; `repel` exists
for post-hoc improvement and is **never relied upon**.

- `repel(cloud, spacing)` (`repel.jl:56`): boundary fixed, volume points
  relax under a force model (`ClippedSpacingForce(β)` default — repulsion
  only, compact support, preserves already-good Poisson-disk clouds);
  escaped points are dropped by an `isinside` filter; stopping by force
  residual `tol`, spacing-CV target, or stall backstop.
- `repel(cloud, spacing, octree)` (`repel.jl:122`): all points move,
  boundary re-projected onto the mesh every iteration, escapees bounce or —
  with `deposit_ratio > 0` — convert into boundary points (emergent surface
  sampling, one-way, self-limiting).
- `cull_ratio` is a near-duplicate safety net that must never fire in
  healthy generation (warns when it does — cull = upstream defect signal).
- Returns a new cloud with `NoTopology`.

### 10a. The force model (`repel_forces.jl`)

The contract: `compute_force(model, u::Real) -> Real` where `u = r/s`
(distance normalized by local spacing). Positive = repulsive (push apart),
negative = attractive (pull together). The scalar multiplies the unit
direction `(xᵢ − xⱼ)/r` in the relaxation step.

Four concrete laws:

| Model | Formula | Root | Behavior |
|---|---|---|---|
| `InverseDistanceForce(β)` | `1/(u²+β)²` | none | pure repulsion, monotonically decreasing; equilibrium only via damping |
| `SpacingEquilibriumForce(β)` | `(1−u²)/(u²+β)²` | `u=1` | repulsive below spacing, attractive above |
| `ClippedSpacingForce(β,u0)` | `max((u0²−u²)/(u²+β)², 0)` | `u=u0` | repulsion-only with compact support; **default** |
| `StrongSpacingForce(β,γ)` | `(1−u²)/(u²+β)^γ` | `u=1` | stronger repulsive core (`u^(-2γ)`); breaks standoffs |

**Why `ClippedSpacingForce` is the default** (`repel_forces.jl:62-100`):
`SpacingEquilibriumForce`'s attractive branch slowly *condenses* already-good
clouds — the preferred bond length `s` is unreachable at the prescribed
density `1/s³`, so the cloud drifts into locally denser clusters plus voids.
This is a measured instability (spacing CV and coordination rise, separation
falls), not noise. `ClippedSpacingForce` removes the attractive branch: any
configuration whose pairwise distances all exceed `u0·s` is an exact
equilibrium — the Poisson-disk property — so an already-blue-noise cloud is
preserved or improved, never re-packed.

**The residual plateau**: a saturated repulsion-only packing's force residual
doesn't vanish (a frustrated glass — the last few points can't settle
simultaneously). It plateaus at a small nonzero value, so `tol` alone never
fires. This is why `stall_after` (no CV improvement for N iterations) is the
practical stop, and why `cv_target ≈ 0.07` is the recommended primary
criterion for the direct pipeline.

### 10b. The relaxation loop (`_relax!`, `repel.jl:202-339`)

The core iteration behind both `repel` methods. Each iteration:

1. **Snapshot refresh** (every `rebuild_every` iterations, `repel.jl:245-253`):
   copy movable points into the `snap` vector (fixed boundary head + movable
   volume tail), rebuild the KDTree over `snap`. On stale trees (rebuild_every
   > 1), a point's own entry may not be its first neighbor — handled by
   **index-skip** (`self = id + n_fixed`), not position-skip, to avoid
   dropping a genuine nearest neighbor and repelling off a self-ghost.

2. **Jacobi-style sweep** (`repel.jl:256-292`): every force is evaluated
   against the same frozen snapshot (`p_old`), not the live positions. Points
   don't see each other's within-iteration moves — a correctness choice
   (Gauss-Seidel would bias toward earlier-indexed points). Parallelized via
   `tmap!` with `TaskLocalValue` buffers for `knn!` (in-place into task-local
   `(ids, dists)` tuples — the per-query allocations of the generic
   `searchdists` path were 66% of the loop's allocations).

3. **Per-point adaptive step** (`repel.jl:284-290`):
   `α_i = clamp(1/|F|, α_lo, α_max)` — inverse of the force magnitude, clamped
   to a safe range. Displacement is `s · α_i · F̂` (spacing-scaled), capped at
   one local spacing (`d_norm > s → scale down`). The step is **per-point**,
   not a global learning rate: strong forces take small steps, weak forces
   take large steps, but nobody jumps more than one spacing.

4. **Constraint** (`constrain` callback): in the volume-only method, the
   constraint is identity (no boundary interaction). In the octree method,
   `_constrain_octree` (`repel.jl:448`) re-projects boundary points onto the
   mesh and reverts escaped volume points (flagging them for deposition).

5. **Stopping** (`repel.jl:293-337`): three independent criteria, checked in
   order:
   - **Force residual `tol`**: `max_i(|F_i|·s_i) < tol` — the classic
     equilibrium test. Rarely fires for `ClippedSpacingForce` (residual
     plateau).
   - **`cv_target`**: coefficient of variation of `d_NN/s` over movable
     points (`_dnn_cv`, line 374) drops below target. On hit, reverts to
     `p_old` — the sweep mutated positions, but a cloud at target should
     return untouched (the CV was measured on the pre-sweep snapshot).
   - **`stall_after`**: CV hasn't improved by >0.1% for N iterations. The
     backstop that terminates default runs.
   - If none fire, `max_iters` is reached (warns).

6. **Closest-pair tracking** (`repel.jl:294-303`): during the sweep, each
   point's nearest neighbor is recorded (`nn_dist`, `nn_id`). This gives the
   global closest pair for free (no extra search), used by `trace` (record
   per-iteration) and `kick_after` (break frozen standoffs with a small random
   nudge to the closer member of the stalled pair).

## 11. Normals

`compute_normals`: PCA on k-NN neighborhoods. `orient_normals!`: global
consistency via minimum spanning tree + DFS (Hoppe 1992). Orientation is
consistent **within a connected component**; disconnected solids (multi-mesh
domains) can end up mutually inconsistent — the motivation for the plan's
per-component signed-volume checks (A2d).

## 11b. Surface splitting (`surface_operations.jl`)

`split_surface!(cloud, angle; k=10)` partitions a single named surface into
sub-surfaces at normal discontinuities. The algorithm
(`surface_operations.jl:58-94`):

1. Build a k-NN graph on the surface points (`k=10` default).
2. For each edge `(i, j)`, keep it only if the angle between the normals at
   `i` and `j` is below `angle` — edges across sharp features are removed.
3. Label connected components of the resulting graph; each becomes a new
   named surface (`:surface1`, `:surface2`, …, auto-named to avoid clashes).
4. **Points are permuted in place** (`many_permute!`) to group each
   component's points contiguously in the underlying arrays — so surface
   ranges are contiguous, not scattered indices.

`combine_surfaces!` is the inverse: concatenates the geometries/normals of
named surfaces into one, keeping the first name, deleting the rest
(`surface_operations.jl:7-27`).

The named surfaces produced by `split_surface!` are the identity the spacing
combinators (plan A2b) and multi-mesh boundary (plan A2d) key on:
`points(bnd[:inlet])`, `SurfaceRefinement(points(bnd[:wall]); …)`, etc.

## 12. Quality instrumentation

- `metrics(cloud; k)`: separation (min pair distance), fill distance, mesh
  ratio.
- `spacing_fidelity_metrics(cloud, spacing; k)`: d_NN/h statistics — CV
  (< 0.15 good; direct-pipeline raw quality ≈ 0.07), coordination number
  (12–14 ideal). Blind-`ustrip` unit bug when spacing unit ≠ cloud unit
  (`metrics.jl:61-62,102`).
- The **executable spec** is `examples/octree_boundary_layer.jl`: bunny,
  steep `BoundaryLayerSpacing` (5× wall:bulk), `max_growth = 0.15`,
  `alpha = 1.0`, auto `max_points`; checks spacing CV, sep/h_min,
  coordination, worst-neighbour d_NN ratio, plus the visual slab-slice
  colored by h(x). New pipeline work must not regress it.

## 13. Known sharp edges (as of 2026-07-12)

1. **Mixed-unit `ustrip` bugs** in `BoundaryLayerSpacing`, `repel`'s CV
   monitor / deposit threshold, `spacing_fidelity_metrics` (plan B1).
2. **SlakKosec broken** yet still the implicit 3D default (plan A3).
3. **2D `discretize` discards a user-passed `alg`** silently and nags on
   every correct call (plan A3).
4. **~19 bare `print`s** narrate the Octree pipeline with no off switch
   (plan A4).
5. **Binary STL → Float32** propagates into every downstream `T`; tests
   that assume Float64 inherit Float32 silently unless they rebuild.
6. **Per-query triangle extraction** dominates hot geometric paths
   (plan A1, `TriangleIndex{T}`).
7. ~~`Octree` constructor default `alpha = 2.0` vs reference example
   `alpha = 1.0` — unresolved.~~ **Resolved 2026-07-12** (benchmark in
   `simplification_plan.md` O4): `2.0` stays the default — quality parity,
   ~20 % faster, 6.8× smaller node octree; `1.0` is the documented
   maximum-fidelity setting.
8. **Octree exits hardcode meters.** `_discretize_volume` returns
   `Point(pt...)` from raw magnitudes (`octree.jl:886,951`); `Cartesian`
   attaches `m` to bare numbers unconditionally (CoordRefSystems
   `cartesian.jl:36`, `addunit.(coords, m)`). A mesh imported in any
   non-meter unit yields meter-labeled wrong-magnitude volume points and a
   CRS-mismatch throw at `PointCloud` assembly. Unexposed: every test and
   example uses meters (plan B1).
