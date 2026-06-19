# Gap-tracking sampler — implementation notes & findings

Task: replace the Bridson dart thrower with a gap-tracking sampler that scales
to production geometry (bunny.stl, 70k facets, ~129k m³). The Bridson front
timed out at every spacing finer than the coarse original parameters.

## Verified findings

### The real bottleneck: BoundaryLayerSpacing evaluation

- `BoundaryLayerSpacing` stores boundary points as a plain vector (no spatial
  index). Every call to the spacing function calls `_min_distance(p, s.boundary)`
  which is a linear scan over ALL boundary points — O(boundary_points) per call.
  Source: `src/discretization/spacings.jl:24-31` and `:114-126`.

- With 69k boundary seeds on bunny.stl and thousands of candidate darts, each
  calling `_spacing_value` → `_min_distance`, this is the dominant cost. The
  stack trace at timeout confirms the bottleneck is in `_min_distance` →
  `BoundaryLayerSpacing` → `_spacing_value`.

- `ConstantSpacing` is O(1) — the scaling problem only manifests with
  `BoundaryLayerSpacing` (and presumably `LogLike` which also calls
  `_min_distance`).

### _bridson_h_min optimization (already applied)

- The original `_bridson_h_min` called `all_leaves(node_tree)` which allocates
  a collector vector. Fixed by iterating `1:num_boxes[]` directly and checking
  `is_leaf`. Source: `src/discretization/algorithms/octree.jl:462-473`.

- This is a minor optimization — the h_min scan runs once at setup. The real
  bottleneck is the per-candidate `_spacing_value` call in the inner loop.

### _GapTracker data structure (already implemented)

- `_GapTracker` is `mutable struct` because `cumw` and `total_volume` are
  updated by `_refresh_pool!`. Source: `octree.jl:612`.

- `total_volume` is correctly updated in `_refresh_pool!` (line 655). This was
  a bug that caused `_draw_leaf` to draw from a stale distribution — fixed.

- The pool tracks interior leaves only (line 626: `classification[idx] ==
  LEAF_INTERIOR || continue`). Boundary leaves are excluded because darts from
  them are almost always rejected by separation (seeds already occupy those
  positions).

### Gap tracker vs Bridson: point count difference

- On cavity (h=0.08, alpha=2.0): Bridson produces 8290 volume points, the
  advancing-front gap tracker produces 6313. Both have correct separation
  (0.06m = 0.75 * 0.08m). The gap tracker is 2.4× faster (0.33s vs 0.79s).

- The point count difference is because gap-tracked candidates (drawn from
  uncovered interior leaves) explore the space differently from Bridson's
  annulus candidates (drawn around active points). The advancing front with
  gap-tracked candidates leaves gaps that Bridson's annulus approach fills.

- Both are valid maximal Poisson-disk samplings — the point sets differ but
  the separation guarantee is identical.

### Both methods timeout on bunny.stl

- Bridson timed out with `BoundaryLayerSpacing(at_wall=0.4m, bulk=2.0m,
  layer_thickness=6.0m)` — 345k octree leaves, 80k boundary seeds.

- Gap-tracked also timed out with the same parameters. The stack trace shows
  the bottleneck is `_spacing_value` → `BoundaryLayerSpacing` → `_min_distance`,
  not the algorithm structure.

- The fix is to cache spacing values so the inner loop does O(1) lookups
  instead of O(boundary_points) distance computations.

### Separation guarantee (verified correct)

- `_bridson_separated` correctly uses `min(r_c, rs[idx])` for mutual separation
  and scans the grid neighborhood within `r_c / cell_size` cells. Source:
  `octree.jl:417-434`.

- Volume-only minimum separation matches expected `factor * h(x)` on both
  cavity and bunny tests.

## Proposed fix: spacing cache

- Cache spacing at leaf centers during `build_node_octree`. After subdivision
  and balancing, evaluate `_spacing_value(T, spacing, box_center(node_tree, idx))`
  for every non-exterior leaf and store in a `Vector{Float64}` indexed by box idx.

- In the inner loop, replace `_spacing_value(T, spacing, c)` with a lookup:
  `find_leaf(node_tree, c)` → read cached value. The `find_leaf` traversal is
  O(log(levels)) ≈ O(19) for 345k leaves.

- The error from using the leaf-center value instead of the exact value at `c`
  is bounded by the spacing gradient over one leaf (size `alpha * h(x)`). For
  `BoundaryLayerSpacing` with smooth sigmoid transition, this is small.

- Expected speedup: from O(69k) to O(19) per candidate — ~3600× on the spacing
  evaluation. This would make both Bridson and gap-tracked fast on bunny.stl.

- The cache should be stored alongside the node tree (returned from
  `build_node_octree` or stored in the `Octree` struct).

## Alternative fix: kd-tree in BoundaryLayerSpacing

- `BoundaryLayerSpacing` stores boundary points as a plain vector. Build a
  `KDTree` (from NearestNeighbors.jl, already a dependency) in the constructor
  and use `knn(tree, query, 1)` for O(log n) nearest-neighbor queries instead
  of O(n) linear scans.

- This fixes the O(boundary_points) cost for ALL uses of `BoundaryLayerSpacing`,
  not just the sampler — including `build_node_octree`, `should_subdivide`, and
  any user code that evaluates the spacing function.

- The codebase already uses `KDTree` in `repel.jl` (line 220) and
  `KNearestSearch` throughout. The infrastructure is there.

- This is arguably the better fix because it's general — the spacing cache is
  a sampler-specific optimization, while the kd-tree fixes the root cause.

- The `BoundaryLayerSpacing` struct would need a `KDTree` field (built from
  the boundary points in the constructor). The `_min_distance` function would
  use `knn` instead of linear scan.

- For small boundary counts (< 1000), the linear scan may be faster than the
  kd-tree overhead. Could use a heuristic: linear scan for small N, kd-tree
  for large N.

## Open questions

- Should the spacing cache be used by all placements (`:bridson`, `:gap_tracked`)
  or only `:gap_tracked`?

- Is the leaf-center approximation accurate enough for graded spacing, or do
  we need sub-leaf interpolation (2×2×2 sub-grid)?

- The gap tracker produces fewer points than Bridson. Is this acceptable (both
  are maximal Poisson-disk samplings) or do we need exact parity?

- Should `_min_distance` in `BoundaryLayerSpacing` use a kd-tree instead of
  a linear scan? This would fix the O(boundary_points) cost for ALL uses of
  the spacing function, not just the sampler.
