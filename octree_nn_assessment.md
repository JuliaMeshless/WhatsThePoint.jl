# Octree-based NN search — assessment & deferral

## What

Replace the per-iteration `KNearestSearch` kd-tree rebuild in `repel` with an
`OctreeNN` wrapper around the existing `SpatialOctree`. The integer coordinate
system already gives O(1) neighbor lookup; extending `find_neighbor` from 6 face
directions to 26 (all `(di,dj,dk) ∈ {-1,0,1}³ \ (0,0,0)`) is the only
structural prerequisite. In-place rebuild = clear `element_lists` + re-insert
each point into its leaf = O(N) vs O(N log N).

## Why defer

Per `repel_convergence_ideas.md` §0, the staged diagnostic (cavity Δ=0.08,
1500 iters) showed:

- Bulk equilibrium is already excellent (`fill/Δ ≈ 0.88–0.96`).
- `separation` is frozen at exactly `0.0550·Δ` for 1400 iterations — a handful
  of stuck volume-point pairs, not iteration starvation.
- `mesh_ratio ≈ 16` is dominated by those outliers, not by covering voids.

The blocker is **node quality** (force/step dynamics), not **per-iteration cost**
(search speed). The octree NN search is a pure performance change — it produces
identical node positions — so it belongs in the wall-clock optimization phase,
after the quality levers (adaptive step, force law) have been landed.

## When to revisit

Once Tier A work closes the `mesh_ratio` gap (target <3) and repel converges
in <50 iterations, the kd-tree rebuild becomes the dominant wall-clock cost.
That is the right time to implement this. The plan in `plan_octree_nn_search.md`
is ready; no design work remains.

## Summary

| Aspect | kd-tree (current) | Octree NN (planned) |
|--------|-------------------|---------------------|
| Build | O(N log N) | O(N) |
| Query | O(k log N) | O(log L + k_local + k log k) |
| Quality | identical | identical |
| Status | working | deferred — quality first |
