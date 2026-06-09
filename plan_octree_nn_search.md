# Plan: Octree-based nearest-neighbor search for repel

> **Status (2026-06-07): this is now a primary workstream — "Lever 2" in
> `NODEGEN_FINDINGS.md`.** The motivation has firmed up: repel needs to run *inside*
> the shape-optimization loop, where the boundary-point count and connectivity change
> on every design step, so the cloud must be re-relaxed cheaply and repeatedly. Once
> Lever 1 (adaptive force) cuts the iteration count, the per-iteration kd-tree rebuild
> becomes the dominant cost — which is exactly what this plan removes. The
> `separation=0` bug referenced below is already fixed independently (`_safe_direction`
> in `src/repel.jl`); any octree-side deduplication here is now defense-in-depth, not
> the fix. Quality (separation/mesh_ratio) must stay identical to the kd-tree path —
> this is a pure performance change.

## Problem statement

The `repel` function in WhatsThePoint.jl uses `KNearestSearch` from Meshes.jl
(a kd-tree) for nearest-neighbor queries. The kd-tree is rebuilt from scratch
every `rebuild_every` iterations (default 1 = every iteration). This is the
dominant cost in the repel loop.

WhatsThePoint.jl already has a `SpatialOctree` data structure — the best tree
data structure for this task — but it is **not used for NN search**. The octree
is only used for:
- Geometry queries (`TriangleOctree`)
- Discretization (`SpacingCriterion`, `build_node_octree`)
- Boundary projection (`_nearest_triangle_octree!`)

**We already have the best tree and we're not using it for the hot path.**

## Current NN search flow (repel)

```
each iteration:
    1. rebuild kd-tree from current positions       ← O(N log N) build
    2. for each point (parallel tmap!):
        a. search kd-tree for k nearest neighbors   ← O(k log N) per query
        b. compute repel force from neighbors
        c. move point
```

Total per iteration: O(N log N) build + O(N k log N) queries = O(N k log N).

## Proposed octree NN search flow

```
each iteration:
    1. rebuild point octree from current positions   ← O(N) build (hash-based)
    2. for each point (parallel tmap!):
        a. find leaf containing point                ← O(log L) traversal
        b. collect points from leaf + 26 neighbors   ← O(k_local) amortized
        c. if |candidates| < k, expand to face/edge/corner neighbors
        d. sort by distance, take k nearest           ← O(k log k)
        e. compute repel force from neighbors
        f. move point
```

Total per iteration: O(N) build + O(N · (log L + k_local + k log k)) queries.

### Key advantage: O(N) build vs O(N log N)

The kd-tree rebuild is O(N log N). The octree "rebuild" is just re-inserting
each point into the appropriate leaf — O(N) with a hash-based leaf lookup.
The integer coordinate system `(i,j,k,N)` gives O(1) neighbor calculation.

### Key advantage: spatial locality

The kd-tree does a global binary search for each query. The octree gives
**direct spatial access** — the leaf containing the query point is found in
O(log L) steps (L = tree depth ≈ 10-15), and the k-nearest neighbors are
almost always within the leaf and its 26 adjacent leaves (3×3×3 neighborhood).

## Architecture analysis

### Option A: Replace kd-tree in repel (minimal change)

Add an `OctreeNN` wrapper around `SpatialOctree` that implements the same
`KNearestSearch`-like interface (`searchdists`), and pass it to repel instead
of the kd-tree.

**Pros:**
- Minimal change to repel code (just swap the search method)
- The octree can be rebuilt in-place (no allocation) by clearing element lists
  and re-inserting points

**Cons:**
- Need to implement the NN query on the octree
- Need to handle the case where the 3×3×3 neighborhood has fewer than k points
  (expand search radius)

### Option B: Build a point octree during repel (integrated)

Instead of using an external search method, integrate the octree directly into
the repel loop. Build the octree once, update it in-place each iteration.

**Pros:**
- No external search method abstraction
- Can amortize the tree construction (only re-insert moved points)
- Can use the octree for both NN search AND boundary projection

**Cons:**
- More invasive change to repel
- Need to handle point deletion/insertion efficiently

### Recommendation: Option A

Option A is the cleanest approach. It keeps the repel code generic (any search
method works) and allows benchmarking the octree vs kd-tree without changing
the repel logic.

## Implementation plan

### Step 1: Define OctreeNN type

```julia
struct OctreeNN{E, T}
    tree::SpatialOctree{E, T}
    k::Int
    points::Vector{SVector{3, T}}  # reference to current positions
end
```

### Step 2: Implement `searchdists(octree_nn, query_point)`

Algorithm:
1. Find leaf containing query point: `find_leaf(tree, query_point)`
2. Collect all points from the leaf and its 26 neighbors (3×3×3 neighborhood)
3. If |candidates| < k, expand to 5×5×5 neighborhood (face-adjacent at distance 2)
4. Continue expanding until |candidates| ≥ k or all leaves are exhausted
5. Sort candidates by distance, return k nearest

The 26-neighbor collection uses `find_neighbor` in each of the 6 directions,
plus the leaf itself. For a balanced octree, each leaf has 6 face neighbors,
12 edge neighbors, and 8 corner neighbors — but the current `find_neighbor`
only handles face neighbors (6 directions). We need to extend it.

### Step 3: Extend find_neighbor for edge and corner neighbors

The current `find_neighbor` handles 6 face directions. For the 3×3×3
neighborhood, we need 26 directions: all `(di, dj, dk) ∈ {-1, 0, 1}³ \ (0,0,0)`.

The integer coordinate system makes this trivial: the neighbor at offset
`(di, dj, dk)` has coordinates `(i+di, j+dj, k+dk)` at the same refinement
level. If it doesn't exist (boundary), try coarser/finer levels.

### Step 4: Implement in-place rebuild

```julia
function rebuild!(nn::OctreeNN, points)
    # Clear all element lists
    for i in 1:nn.tree.num_boxes[]
        empty!(nn.tree.element_lists[i])
    end
    # Re-insert each point into its leaf
    for (idx, p) in enumerate(points)
        leaf = find_leaf(nn.tree, p)
        push!(nn.tree.element_lists[leaf], idx)
    end
end
```

This is O(N) — just a tree traversal per point. No allocations.

### Step 5: Benchmark vs kd-tree

Compare:
- Build time: octree O(N) vs kd-tree O(N log N)
- Query time: octree O(log L + k_local) vs kd-tree O(k log N)
- Total repel time: wall-clock for 100 iterations on the cavity geometry
- Quality: separation, mesh_ratio (should be identical since same force model)

Expected: the octree should be significantly faster for large N (>10k points)
because:
1. Build is O(N) vs O(N log N)
2. Queries are O(log L) vs O(log N) where L ≈ 12-15 and log N ≈ 14-17
3. The octree gives spatial locality (neighbors are nearby in memory)

### Step 6: Integrate with repel

Add a `search_method` kwarg to `repel`:

```julia
function repel(cloud, spacing, octree::TriangleOctree;
               search_method = :octree,  # or :kdtree
               k = 21, ...)
```

Default to `:octree` when a `TriangleOctree` is provided (the octree is
already available). Fall back to `:kdtree` for the volume-only method.

## Open questions

### Q1: What is the typical k_local (points per leaf)?

For a uniform cloud with spacing Δ and leaf size h, each leaf contains
approximately (h/Δ)³ points. With the current octree parameters:
- h ≈ Δ/alpha = Δ (alpha=1.0)
- k_local ≈ 1 point per leaf (too few!)

This means the 3×3×3 neighborhood (27 leaves) would give ~27 points, which
is enough for k=21. But for non-uniform clouds or coarser octrees, we may
need to expand further.

**Mitigation**: use a coarser octree for NN search than for discretization.
The NN octree should have leaf size ≈ 2-3Δ so each leaf has 8-27 points.

### Q2: How does the octree handle the boundary projection?

The boundary-projected repel method projects points to the nearest mesh
triangle using `_nearest_triangle_octree!`. This already uses the
`TriangleOctree`. The point octree for NN search is separate — it only
stores point indices, not triangles.

The two octrees serve different purposes:
- `TriangleOctree`: mesh geometry (triangles, classification)
- Point octree: spatial index for point cloud (NN search)

### Q3: Can we deduplicate in the octree?

The `separation=0` bug was traced to a `0/0` NaN singularity in the force
calculation at `r=0` — now fixed with `_safe_direction`. The octree naturally
groups coincident points in the same leaf, so it could serve as an efficient
deduplication detector during the rebuild step (defense-in-depth):

```julia
# During rebuild: detect and merge coincident points
for (idx, p) in enumerate(points)
    leaf = find_leaf(nn.tree, p)
    existing = nn.tree.element_lists[leaf]
    for j in existing
        if norm(points[j] - p) < eps_tol
            # Duplicate detected — skip or merge
            continue
        end
    end
    push!(nn.tree.element_lists[leaf], idx)
end
```

This is a natural advantage of the octree for the repel use case.

## Complexity comparison

| Operation | kd-tree | Octree |
|-----------|---------|--------|
| Build | O(N log N) | O(N) |
| k-NN query | O(k log N) | O(log L + k_local + k log k) |
| Space | O(N) | O(N + n_leaves) |
| Rebuild | allocate + build | clear + re-insert |
| Duplicate detection | none | O(1) per leaf |
| Neighbor finding | implicit (tree traversal) | explicit (26-direction lookup) |

For N=50k, k=21, L=15:
- kd-tree build: ~50k × 16 = 800k comparisons
- octree build: ~50k leaf lookups
- kd-tree query: ~21 × 16 = 336 comparisons per point
- octree query: ~15 (traversal) + ~30 (candidates) + ~21 × 4.4 (sort) ≈ 140 per point

## Files to create/modify

1. `src/octree/octree_nn.jl` — new file: `OctreeNN` type, `searchdists`,
   `rebuild!`
2. `src/octree/spatial_octree.jl` — extend `find_neighbor` for 26 directions
3. `src/repel.jl` — add `search_method` kwarg, use `OctreeNN` when available
4. `test/octree_nn.jl` — new file: unit tests for octree NN search
5. `validate_octree_nn.jl` — benchmark: octree NN vs kd-tree on cavity geometry

## Risk assessment

- **Low risk**: the octree NN search is a pure performance optimization.
  The repel algorithm is unchanged; only the search method differs.
- **Medium risk**: the 26-direction neighbor finding needs careful testing
  at octree boundaries and across refinement levels.
- **High risk**: if k_local is too small (leaf size ≈ Δ), the search may
  need many expansion steps, negating the O(N) build advantage. Mitigation:
  use a coarser NN octree (leaf size ≈ 2-3Δ).

## Recommendation

Implement Option A (OctreeNN wrapper) with the following priority:
1. First: extend `find_neighbor` to 26 directions (this is the foundation)
2. Second: implement `OctreeNN` with `searchdists` and `rebuild!`
3. Third: benchmark on cavity geometry
4. Fourth: integrate with repel as `search_method` kwarg
5. Fifth: add deduplication in the rebuild step (defense-in-depth; the
   `separation=0` bug is already fixed by `_safe_direction`)

Steps 1-3 are pure additions (no changes to existing code). Step 4 is a
minor refactor of repel. Step 5 is now defense-in-depth (the `separation=0` bug is
already fixed by `_safe_direction`); the octree just makes coincidence detection O(1).
