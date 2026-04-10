# Node Repel Refinement - Refined Implementation Plan

## Executive Summary

This plan addresses three critical issues in the node repel algorithm:

1. **Correctness bug (Phase 1):** Points can escape the domain boundary, causing volume depletion
2. **Spacing preservation (Phase 1):** Repel must maintain the spacing function from discretization - this is paramount for meshless methods
3. **Performance bottleneck (Phase 3):** Static neighbor search becomes a critical limitation for large clouds (N > 10,000)

**Your assessment is correct:** Octree-based neighbor search is **essential** for production use with large point clouds. Without it, the algorithm will not scale beyond research-sized problems. However, we prioritize fixing the correctness bug first, then implement performant neighbor queries.

**Implementation priority:**
1. **Phase 1 (MVP):** Fix boundary projection bug → Get algorithm working correctly
2. **Phase 2 (Enhancement):** Add custom force models → Enable physics flexibility
3. **Phase 3 (Production):** Implement octree neighbor search → Enable large-scale applications

## Problem Statement

The current `repel()` implementation ([src/repel.jl](src/repel.jl)) has two critical flaws:

1. **Volume points can escape:** Interior volume points can be pushed outside the domain boundary during force application, then discarded by `isinside()` filter, causing volume depletion
2. **Boundary points are frozen:** Only volume points move (line 22: `p = copy(volume(cloud).points)`), but frozen boundary points can collide with moving volume points, creating poor distributions

**Current problematic code:**
```julia
return xi + Vec(s * α * repel_force)  # Can push points outside - no boundary constraints
```

### Required Behavior: All Points Must Move

**Critical requirement:** ALL points (both volume AND boundary) must participate in repel forces to avoid point collisions and maintain good spacing.

**Point movement and projection rules:**

1. **Volume points (interior):**
   - Apply repel forces normally
   - **If moved position is still inside:** Accept new position, remain volume point
   - **If moved position is outside:** Project to boundary surface, **reclassify as boundary point**

2. **Boundary points (on surface):**
   - Apply repel forces normally (they interact with all neighbors)
   - **Always project back to boundary surface** after force application
   - **Remain boundary points** (cannot become volume points)

3. **Projection strategy:**
   - Find nearest triangle on boundary mesh
   - Project point onto triangle (closest point on triangle)
   - Apply small inward offset for numerical stability

**Result:** Boundary mesh adapts during optimization, volume points that reach surface become part of boundary.

### Surface Structure During Repel

**Design decision:** Treat all boundary points as a unified "boundary" surface during repel optimization:

**Why this simplification?**
- Boundary points need to interact with ALL neighbors (regardless of which surface they belong to)
- Surface distinctions are geometric metadata, not physical constraints during force calculations
- Reassigning surfaces after each iteration is complex and error-prone
- Users can re-establish surface splits after repel using `split_surface!(boundary, angle)`

**Phase 1 MVP approach:**
1. Flatten all named surfaces into unified boundary during repel
2. Return cloud with single "boundary" surface
3. Document that users should re-split if needed: `split_surface!(cloud.boundary, 75°)`

**Phase 2 enhancement (optional):**
- Implement surface reassignment after repel to preserve original surface structure
- Map boundary points back to surfaces based on nearest mesh triangles
- More complex but maintains workflow compatibility

## Critical Performance Assessment: Neighbor Search Strategy

**Your concern is valid:** For large-scale repel refinement (N > 10,000 points), the neighbor search strategy is a critical bottleneck that will determine whether the algorithm is practical.

### Current Implementation Analysis

The existing `repel()` builds a `KNearestSearch` tree **once** at the start (line 26):
```julia
method = KNearestSearch(all_p, k)  # Built once, never updated
```

**Problem:** As points move through iterations, neighbors become stale:
- Iteration 1: accurate neighbors
- Iteration 100: neighbors may be from opposite side of domain!
- Result: Incorrect forces, poor convergence, potential instability

**Cost analysis for large clouds:**

| Approach | Build Cost | Query Cost/Iter | Total Cost (100 iters) | Neighbor Accuracy |
|----------|-----------|----------------|----------------------|------------------|
| Static KNN (current) | O(N log N) | O(Nk) | O(N log N + 100Nk) | Degrades over time |
| Rebuild KNN each iter | O(N log N) | O(Nk) | O(100N log N) | Perfect, but **very slow** |
| Octree-based queries | O(N) | O(N log N) | O(100N log N) | Perfect, **much faster** |

For N=100,000, k=21, rebuilding KNN every iteration is **prohibitively expensive** (~10-20 minutes per iteration).

### Recommended Strategy: Phased Approach

**Phase 1 (MVP):** Accept stale neighbors as a limitation
- Document in docstring: "Best for clouds with N < 10,000 points"
- Focus on fixing the boundary projection bug (the real critical issue)
- Get the algorithm working correctly first

**Phase 3 (Production-ready):** Implement octree-based neighbor search
- **This is ESSENTIAL for large-scale applications** - you're absolutely right
- Build a lightweight point octree for O(log N) neighbor queries
- Update incrementally as points move
- Expected speedup: **10-50x** for large clouds

**Decision:** We'll include octree neighbor search in the implementation plan, but as a **Phase 3 high-priority item** rather than Phase 1. This allows us to:
1. Fix the correctness bug immediately (boundary projection)
2. Test the algorithm on small/medium clouds
3. Implement performant neighbor search before scaling up

## Architecture Overview

The refined implementation should integrate with existing infrastructure:

1. **TriangleOctree** ([src/octree/triangle_octree.jl](src/octree/triangle_octree.jl)) - Already provides:
   - Fast signed distance queries via `_compute_signed_distance_octree()`
   - Nearest triangle search via `_nearest_triangle_octree!()`
   - Point projection via `closest_point_on_triangle()` in [src/octree/geometric_utils.jl](src/octree/geometric_utils.jl)

2. **Neighbor search performance bottleneck** - Currently uses `KNearestSearch` (Meshes.jl)
   - **Critical Issue:** For N points with k neighbors, naive rebuilding every iteration is O(N²k) per iteration
   - **Current implementation rebuilds KNN tree once** at start (line 26 in repel.jl) - this is actually fine!
   - The tree is built on `all_p = points(cloud)` which includes both boundary and volume points
   - Points move each iteration, but the tree is **static** - uses stale neighbor information
   - **Performance assessment:**
     - For small clouds (N < 1000): Current approach is acceptable
     - For large clouds (N > 10,000): Stale neighbors may slow convergence or cause instability
   - **Phase 1 decision:** Keep current `KNearestSearch` for simplicity - it works
   - **Phase 3 option:** Octree-based dynamic neighbor updates if profiling shows neighbor staleness is an issue

3. **Immutable design pattern** - All WhatsThePoint types are immutable (AD compatibility)
   - Return new `PointCloud` with updated positions
   - No in-place octree updates needed

## Detailed Implementation Plan

### Phase 1: Core Boundary-Aware Repel

**Goal:** Fix the point-ejection bug with proper boundary projection.

**Changes to `repel()` function:**

1. **Accept optional `TriangleOctree`** parameter:
   ```julia
   function repel(
       cloud::PointCloud{𝔼{N}, C},
       spacing;
       β = 0.2,
       α = minimum(spacing.(to(cloud))) * 0.05,
       k = 21,
       max_iters = 1000,
       tol = 1.0e-6,
       convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
       octree::Union{Nothing, TriangleOctree} = nothing,  # NEW
   ) where {N, C <: CRS}
   ```

2. **Build octree if not provided** (for backward compatibility):
   ```julia
   if isnothing(octree)
       octree = TriangleOctree(boundary(cloud).mesh; classify_leaves=true)
   end
   ```

3. **Move ALL points (volume + boundary), not just volume**:
   ```julia
   # OLD: p = copy(volume(cloud).points)  # Only volume points
   # NEW: Move all points, track which are boundary
   all_p = points(cloud)  # Boundary + volume combined
   npoints = length(all_p)
   p = copy(all_p)
   p_old = deepcopy(p)

   # Track boundary point indices for projection
   n_boundary = length(boundary(cloud))
   boundary_indices = 1:n_boundary
   volume_indices = (n_boundary+1):npoints
   ```

4. **Replace point update logic** - apply forces to ALL points:
   ```julia
   tmap!(p, 1:npoints) do id
       xi = p_old[id]
       ids, dists = searchdists(xi, method)
       ids = @view ids[2:end]
       neighborhood = @view all_p[ids]
       rij = norm.(@view dists[2:end])
       s = spacing(xi)

       repel_force = sum(zip(neighborhood, rij)) do z
           xj, r = z
           @inbounds F(r / s) * (xi - xj) / r
       end

       # Compute proposed new position (same for volume and boundary)
       x_new = xi + Vec(s * α * repel_force)

       # Different projection rules for volume vs boundary points
       is_boundary = id <= n_boundary
       return _constrain_to_domain(x_new, xi, octree, is_boundary)
   end
   ```

5. **Implement `_constrain_to_domain()` with boundary/volume distinction**:
   ```julia
   function _constrain_to_domain(
       x_proposed::Point{𝔼{3}},
       x_original::Point{𝔼{3}},
       octree::TriangleOctree,
       is_boundary_point::Bool
   )
       # Boundary points: ALWAYS project to surface
       if is_boundary_point
           return _project_to_boundary(x_proposed, x_original, octree)
       end

       # Volume points: only project if they escaped
       if isinside(x_proposed, octree)
           return x_proposed  # Still inside, accept new position
       else
           # Escaped → project to boundary (will be reclassified later)
           return _project_to_boundary(x_proposed, x_original, octree)
       end
   end

   function _project_to_boundary(
       x_proposed::Point{𝔼{3}},
       x_fallback::Point{𝔼{3}},
       octree::TriangleOctree
   )
       sv_proposed = _extract_vertex(Float64, x_proposed)
       state = NearestTriangleState{Float64}(sv_proposed)
       _nearest_triangle_octree!(sv_proposed, octree.tree, octree.mesh, 1, state)

       if state.closest_idx == 0
           # Projection failed - use fallback position
           return x_fallback
       end

       # Project to closest point on nearest triangle
       v1, v2, v3 = _get_triangle_vertices(Float64, octree.mesh, state.closest_idx)
       projected_sv = closest_point_on_triangle(sv_proposed, v1, v2, v3)

       # Offset slightly inward to ensure numerical stability
       normal = _get_triangle_normal(Float64, octree.mesh, state.closest_idx)
       offset = -1e-8 * normal  # Small inward nudge

       return Point{𝔼{3}}(projected_sv + offset)
   end
   ```

6. **Reclassify points and re-establish surface distinctions**:
   ```julia
   # After all points moved, separate into volume and boundary
   # Strategy: Treat all boundary as unified during repel,
   # re-establish surface distinctions at the end

   new_boundary_points = Point{𝔼{3}}[]
   new_volume_points = Point{𝔼{3}}[]

   for (i, point) in enumerate(p)
       was_boundary = i <= n_boundary
       is_on_surface = !isinside(point, octree)  # On or very near surface

       if was_boundary || is_on_surface
           push!(new_boundary_points, point)
       else
           push!(new_volume_points, point)
       end
   end

   # Reconstruct cloud with updated volume
   new_volume = PointVolume(new_volume_points)

   # Re-establish boundary surface distinctions
   new_boundary = _reassign_boundary_surfaces(
       new_boundary_points,
       boundary(cloud),
       octree
   )

   return PointCloud(new_boundary, new_volume, NoTopology())
   ```

7. **Implement `_reassign_boundary_surfaces()` to restore surface structure**:
   ```julia
   function _reassign_boundary_surfaces(
       boundary_points::Vector{Point{𝔼{3}}},
       original_boundary::PointBoundary,
       octree::TriangleOctree
   )
       # Strategy: Assign each boundary point to the surface whose
       # original points are closest (or whose mesh triangles are nearest)

       # Get original surface structure
       surface_names = keys(original_boundary.surfaces)
       n_surfaces = length(surface_names)

       # For each boundary point, find which original surface it's closest to
       surface_assignments = Dict{Symbol, Vector{Point{𝔼{3}}}}()
       for name in surface_names
           surface_assignments[name] = Point{𝔼{3}}[]
       end

       for pt in boundary_points
           # Find nearest triangle and determine which surface it belongs to
           sv = _extract_vertex(Float64, pt)
           state = NearestTriangleState{Float64}(sv)
           _nearest_triangle_octree!(sv, octree.tree, octree.mesh, 1, state)

           if state.closest_idx > 0
               # Determine which original surface this triangle belongs to
               # This requires mapping mesh triangle indices to surface names
               surface_name = _get_surface_for_triangle(
                   state.closest_idx,
                   original_boundary,
                   octree.mesh
               )
               push!(surface_assignments[surface_name], pt)
           end
       end

       # Reconstruct PointBoundary with updated points but original structure
       # Need to preserve normals/areas - recompute or interpolate from nearest
       # (Implementation details depend on PointBoundary structure)
       return _rebuild_boundary_from_assignments(surface_assignments, original_boundary)
   end
   ```

**Implementation note:** The surface reassignment requires:
- **Mapping triangles → surfaces**: Need to track which mesh triangles belong to which named surfaces
- **Normal/area recomputation**: Boundary points need normals and areas recomputed after movement
- **Fallback handling**: Points that can't be assigned (projection failed, etc.) go to a default surface

**Simplification option for Phase 1 MVP:**
- Merge all surfaces into single "boundary" surface during repel
- Accept that surface distinctions are lost
- Document that users should re-split surfaces after repel if needed (using `split_surface!`)
- Defer proper surface preservation to Phase 2 or future work

**Recommended approach:**
```julia
# Simplified Phase 1: Unified boundary
new_boundary = PointBoundary("unified", new_boundary_points, mesh)
return PointCloud(new_boundary, new_volume, NoTopology())

# User can re-establish surface splits after repel:
# split_surface!(new_cloud.boundary, 75°)
```

### Phase 1.5: Spacing Enforcement and Validation

**Critical requirement:** The repel algorithm must **maintain the spacing function** that was computed during discretization. The octree discretization algorithm carefully places points according to a spacing function - repel must preserve this, not destroy it.

**Current spacing integration (already present):**
```julia
s = spacing(xi)                           # Get local target spacing
repel_force = sum(...) do
    F(r / s) * (xi - xj) / r              # Force normalized by spacing
end
return xi + Vec(s * α * repel_force)      # Movement scaled by spacing
```

**Problem:** While forces are spacing-aware, there's no **validation** that final spacing matches target.

**Required additions:**

1. **Spacing validation metrics**:
   ```julia
   function _compute_spacing_errors(
       cloud::PointCloud,
       spacing::AbstractSpacing,
       k::Int = 21
   )
       all_p = points(cloud)
       method = KNearestSearch(all_p, k)

       max_error = 0.0
       mean_error = 0.0

       for (i, xi) in enumerate(all_p)
           s_target = spacing(xi)
           ids, dists = searchdists(xi, method)

           # Check nearest neighbor distance vs target
           r_actual = minimum(@view dists[2:end])  # Skip self
           error = abs(r_actual - s_target) / s_target

           max_error = max(max_error, error)
           mean_error += error
       end

       mean_error /= length(all_p)
       return (max_error = max_error, mean_error = mean_error)
   end
   ```

2. **Post-repel spacing validation**:
   ```julia
   # After repel completes, validate spacing
   spacing_errors = _compute_spacing_errors(new_cloud, spacing, k)

   if spacing_errors.max_error > 0.5  # 50% error threshold
       @warn "Repel did not converge to target spacing" spacing_errors
   end

   # Optionally return metrics
   return (cloud = new_cloud, convergence = conv, spacing_errors = spacing_errors)
   ```

3. **Adaptive α parameter based on spacing**:
   ```julia
   # Current: α is global constant
   # Better: α adapts based on how well spacing is being maintained

   function _adaptive_step_size(spacing_error, α_base, iteration)
       # If spacing error is high, reduce step size
       # If spacing error is low and convergence slow, increase step size
       if spacing_error > 0.2
           return α_base * 0.5  # Reduce step when spacing is violated
       elseif spacing_error < 0.05 && iteration > 50
           return α_base * 1.2  # Increase step when spacing is well-maintained
       else
           return α_base
       end
   end
   ```

4. **Force model must respect spacing equilibrium**:
   ```julia
   # The force model should have equilibrium at r ≈ s
   # Current: F(r/s) = 1 / ((r/s)^2 + β)^2
   # At r = s: F(1) = 1 / (1 + β)^2  (not zero!)

   # Improved force with proper equilibrium:
   struct SpacingAwareForce{T<:Real} <: RepelForceModel
       β::T
       equilibrium_ratio::T  # Target r/s ratio (default: 1.0)
   end

   function compute_force(model::SpacingAwareForce, xi, xj, r, s)
       ratio = r / s
       target = model.equilibrium_ratio

       # Repulsive when r < target*s, attractive when r > target*s
       # Zero force at r = target*s
       return (target - ratio) / ((ratio - target)^2 + model.β)^2 * (xi - xj) / r
   end
   ```

**Implementation priority:**
- ✅ Current code already uses spacing in force calculation (good!)
- ⚠️ Add spacing validation metrics (Phase 1 - essential for testing)
- 🔧 Add adaptive α based on spacing error (Phase 2 - nice to have)
- 🔬 Research better force models with proper equilibrium (Phase 2/3)

**Testing requirement:**
```julia
@testitem "repel maintains target spacing" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0mm)

    # Create cloud with octree discretization (perfect spacing)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves=true)
    cloud = discretize(boundary, spacing; alg=Octree(octree), max_points=1000)

    # Measure initial spacing quality
    errors_before = _compute_spacing_errors(cloud, spacing)

    # Apply repel
    new_cloud = repel(cloud, spacing; max_iters=100)

    # Measure final spacing quality
    errors_after = _compute_spacing_errors(new_cloud, spacing)

    # Spacing should not degrade significantly
    @test errors_after.mean_error < errors_before.mean_error * 1.5  # Allow 50% degradation
    @test errors_after.max_error < 0.5  # Max 50% error anywhere

    # Ideally, spacing should improve or stay same
    @test errors_after.mean_error < 0.2  # Mean error < 20%
end
```

### Phase 2: Customizable Repel Force Functions

**Goal:** Allow users to define custom force functions for different physics.

**Implementation:**

1. **Define force function type with spacing awareness**:
   ```julia
   # Abstract type for force models
   abstract type RepelForceModel end

   # Default inverse-distance force (current implementation - SPACING UNAWARE!)
   struct InverseDistanceForce{T<:Real} <: RepelForceModel
       β::T  # Regularization parameter
   end
   InverseDistanceForce(; β=0.2) = InverseDistanceForce(β)

   # ⚠️ WARNING: This force does NOT have equilibrium at r=s!
   # At r=s: F(1) = β/(1+β)^2 ≠ 0
   # This means forces never balance, only damping (via α) controls movement
   function compute_force(
       model::InverseDistanceForce,
       xi::Point,      # Current point
       xj::Point,      # Neighbor point
       r::Real,        # Distance
       s::Real         # Local spacing
   )
       return model.β / ((r/s)^2 + model.β)^2 * (xi - xj) / r
   end

   # IMPROVED: Spacing-aware force with proper equilibrium
   struct SpacingEquilibriumForce{T<:Real} <: RepelForceModel
       β::T           # Smoothing parameter
       r_eq::T        # Equilibrium ratio r/s (typically 1.0)
   end
   SpacingEquilibriumForce(; β=0.2, r_eq=1.0) = SpacingEquilibriumForce(β, r_eq)

   # Force is zero when r/s = r_eq, repulsive when closer, attractive when farther
   function compute_force(
       model::SpacingEquilibriumForce,
       xi::Point,
       xj::Point,
       r::Real,
       s::Real
   )
       ratio = r / s
       deviation = model.r_eq - ratio
       # Force magnitude decreases with distance from equilibrium
       magnitude = deviation / ((deviation)^2 + model.β)^2
       return magnitude * (xi - xj) / r
   end
   ```

2. **Add examples of other force models** (for user guidance):
   ```julia
   # Lennard-Jones potential
   struct LennardJonesForce{T<:Real} <: RepelForceModel
       σ::T  # Zero-force distance
       ε::T  # Potential well depth
   end

   function compute_force(model::LennardJonesForce, xi, xj, r, s)
       σ_r = model.σ / r
       return 24 * model.ε * σ_r^7 * (2*σ_r^6 - 1) * (xi - xj) / r^2
   end

   # Simple spring force
   struct LinearSpringForce{T<:Real} <: RepelForceModel
       k::T  # Spring constant
   end

   function compute_force(model::LinearSpringForce, xi, xj, r, s)
       return model.k * (s - r) * (xi - xj) / r
   end
   ```

3. **Update `repel()` signature**:
   ```julia
   function repel(
       cloud::PointCloud{𝔼{N}, C},
       spacing;
       force_model::RepelForceModel = InverseDistanceForce(),  # NEW
       β = 0.2,  # Deprecated, kept for backward compatibility
       α = minimum(spacing.(to(cloud))) * 0.05,
       k = 21,
       max_iters = 1000,
       tol = 1.0e-6,
       convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
       octree::Union{Nothing, TriangleOctree} = nothing,
   ) where {N, C <: CRS}
   ```

4. **Replace force computation** (line 48-51):
   ```julia
   # OLD:
   # F = let β = β
   #     r -> 1 / (r^2 + β)^2
   # end
   # repel_force = sum(zip(neighborhood, rij)) do z
   #     xj, r = z
   #     @inbounds F(r / s) * (xi - xj) / r
   # end

   # NEW:
   repel_force = sum(zip(neighborhood, rij)) do z
       xj, r = z
       @inbounds compute_force(force_model, xi, xj, r, s)
   end
   ```

### Phase 3: Optimization and Parallelization

**Important:** This phase should be **DEFERRED** until Phases 1-2 are complete, tested, and merged. Premature optimization risks code complexity without proven benefit.

**Potential optimizations** (for future consideration):

#### 3.1 Octree-Based Neighbor Search (High Priority)

**Problem:** Current implementation builds KNN tree once at start, then uses stale neighbors throughout iterations.

**Why this matters for large clouds:**
- For N=100,000 points with typical repel iterations (~100-1000), neighbor relationships change significantly
- Stale neighbors → incorrect force calculations → slow/unstable convergence
- Rebuilding KNearestSearch every iteration is O(N log N) build + O(Nk log N) queries - expensive!

**Octree-based solution:**

1. **Build point octree for neighbor queries:**
   ```julia
   # In repel() function, replace KNearestSearch with octree-based queries
   # Build a spatial octree for volume points
   node_octree = _build_node_octree(p, spacing)  # One-time build
   ```

2. **Implement efficient radius/KNN queries:**
   ```julia
   function _octree_knn_query(
       point::Point,
       k::Int,
       octree::SpatialOctree,
       all_points::Vector{Point}
   )
       # Traverse octree to find k nearest neighbors
       # Much faster than rebuilding full KNN tree
       # Can be updated incrementally as points move
       # Returns (neighbor_indices, distances)
   end
   ```

3. **Incremental octree updates:**
   ```julia
   # After each iteration, update octree for moved points
   # Only rebalance if points moved significantly
   if max_movement > threshold
       _update_octree_positions!(node_octree, p)
   end
   ```

**Expected speedup:** 5-10x for large clouds (N > 50,000) with many iterations

**Complexity:**
- Current: O(N log N) build once, O(k) queries per point per iteration (stale neighbors)
- Octree: O(N log N) build, O(log N + k) queries per point, O(N) incremental updates

#### 3.2 Spatial Coloring for Safe Parallelism

**Goal:** Process multiple points simultaneously without race conditions.

1. **Partition points by octree leaves:**
   - Assign each point to its containing octree leaf
   - Points in non-adjacent leaves can be updated in parallel

2. **Graph coloring algorithm:**
   ```julia
   # Color octree leaves such that no adjacent leaves share a color
   # Process all leaves of color 1, then color 2, etc.
   # Typically needs 8 colors for 3D octree
   leaf_colors = _color_octree_leaves(node_octree)
   ```

3. **Parallel iteration:**
   ```julia
   for color in 1:max_colors
       colored_points = filter_by_color(p, leaf_colors, color)
       tmap!(colored_points) do xi  # Parallel over same-color points
           # Compute repel forces safely (no adjacent points moving)
       end
   end
   ```

**Expected speedup:** Near-linear with thread count for large clouds

#### 3.3 Incremental Neighbor Updates

**Strategy:** Only rebuild neighbor structure when points move significantly.

```julia
# Track cumulative movement since last rebuild
cumulative_movement = zeros(npoints)

while i < max_iters
    # ... compute forces and move points ...

    cumulative_movement .+= norm.(p .- p_old)

    if maximum(cumulative_movement) > rebuild_threshold * spacing
        # Rebuild neighbor structure (octree or KNN)
        update_neighbor_structure!(method, p)
        cumulative_movement .= 0
    end
end
```

**Trade-off:** Accuracy vs speed - larger thresholds = fewer rebuilds but staler neighbors

#### 3.4 Adaptive Force Scaling

**Goal:** Prevent oscillations and speed up convergence in later iterations.

```julia
# Exponential decay of step size
α_iter = α * exp(-decay_rate * i / max_iters)

# Or adaptive based on convergence
if convergence_improving
    α_iter *= 1.1  # Increase step
else
    α_iter *= 0.9  # Decrease step
end
```

**Recommendation:** Start simple. Measure performance. Profile to identify actual bottlenecks. Implement optimizations in priority order:
1. **Octree neighbor search** (if N > 10,000 and convergence is slow)
2. **Spatial coloring** (if wall-clock time is limiting factor)
3. **Incremental updates** (if neighbor rebuild is profiler hotspot)
4. **Adaptive scaling** (if convergence behavior is unstable)

## Testing Strategy

### Unit Tests

1. **Boundary projection test** - Add to [test/repel.jl](test/repel.jl):
   ```julia
   @testitem "repel projects points to boundary" setup = [TestData, CommonImports] begin
       boundary = PointBoundary(TestData.BOX_PATH)
       octree = TriangleOctree(TestData.BOX_PATH; classify_leaves=true)
       spacing = _relative_spacing(boundary)

       # Start with points very close to boundary
       cloud = discretize(boundary, spacing; alg=SlakKosec(octree), max_points=50)

       # Apply strong repulsion to likely eject some points
       new_cloud = repel(cloud, spacing; β=0.1, α=0.5, max_iters=20, octree=octree)

       # Verify no points were lost (projection kept them in)
       @test length(volume(new_cloud)) == length(volume(cloud))

       # All points must be inside
       for p in volume(new_cloud).points
           @test isinside(p, octree)
       end
   end
   ```

2. **Custom force model test**:
   ```julia
   @testitem "repel accepts custom force models" setup = [TestData, CommonImports] begin
       # Test that different force models work
       boundary = PointBoundary(TestData.BOX_PATH)
       spacing = _relative_spacing(boundary)
       cloud = discretize(boundary, spacing; max_points=30)

       # Test default force
       cloud1 = repel(cloud, spacing; max_iters=5)
       @test length(volume(cloud1)) > 0

       # Test custom force model
       lj_force = LennardJonesForce(σ=1.0, ε=1.0)
       cloud2 = repel(cloud, spacing; force_model=lj_force, max_iters=5)
       @test length(volume(cloud2)) > 0
   end
   ```

3. **Fix flaky tests** - Re-enable skipped tests:
   ```julia
   # In test/repel.jl:16 and :89
   # Change @test_skip to @test after boundary projection is implemented
   @test length(volume(new_cloud)) > 0  # Should never fail now
   ```

### Integration Tests

1. **Large-scale refinement** - Test with bifurcation.stl (24,780 points)
2. **Boundary layer preservation** - Verify points near boundaries stay properly distributed
3. **Convergence analysis** - Plot convergence rates for different force models

## Implementation Checklist

- [ ] Phase 1: Core boundary projection with point reclassification
  - [ ] Add `octree` parameter to `repel()`
  - [ ] **Change from moving only volume to moving ALL points** (volume + boundary)
  - [ ] Track boundary vs volume point indices during movement
  - [ ] Implement `_constrain_to_domain()` with boundary/volume distinction
    - [ ] Volume points: project only if escaped, reclassify if escaped
    - [ ] Boundary points: always project to surface, stay boundary
  - [ ] Implement `_project_to_boundary()` helper for surface projection
  - [ ] Implement post-movement reclassification logic
    - [ ] Identify which volume points escaped (became boundary)
    - [ ] **Phase 1 MVP:** Merge all boundary into unified "boundary" surface (simplest)
    - [ ] **Phase 2 (optional):** Implement `_reassign_boundary_surfaces()` to preserve surface distinctions
      - [ ] Map mesh triangles to original surface names
      - [ ] Assign each boundary point to nearest surface
      - [ ] Recompute normals and areas for boundary points
    - [ ] Reconstruct PointCloud with updated volume/boundary separation
  - [ ] Remove the problematic `isinside` filter
  - [ ] Add 2D fallback (error or simplified projection)
  - [ ] **CRITICAL:** Implement spacing validation and enforcement
    - [ ] Add `_compute_spacing_errors()` to measure spacing quality
    - [ ] Validate spacing after repel completes
    - [ ] Add test to verify spacing is maintained (not degraded)
    - [ ] Document spacing behavior in docstring
- [ ] Phase 2: Custom force models
  - [ ] Define `RepelForceModel` abstract type
  - [ ] Implement `InverseDistanceForce`, `LennardJonesForce`, `LinearSpringForce`
  - [ ] Add `force_model` parameter to `repel()`
  - [ ] Update force computation loop
  - [ ] Deprecate standalone `β` parameter (keep for backward compat)
- [ ] Testing
  - [ ] Add boundary projection test
  - [ ] Add custom force model test
  - [ ] Re-enable skipped tests (`:16`, `:89`)
  - [ ] Run full test suite
- [ ] Documentation
  - [ ] Update `repel()` docstring with new parameters
  - [ ] Add examples for custom force models
  - [ ] Document projection behavior in user guide
  - [ ] Update [TEST_STATUS.md](TEST_STATUS.md)
- [ ] Phase 3: Performance optimizations (**REQUIRED for N > 10,000**)
  - [ ] **HIGH PRIORITY:** Implement octree-based neighbor search
    - [ ] Build spatial octree for volume points
    - [ ] Implement `_octree_knn_query()` for O(log N) neighbor queries
    - [ ] Add incremental octree updates between iterations
    - [ ] Benchmark against static KNN (expect 10-50x speedup for large N)
  - [ ] MEDIUM PRIORITY: Spatial coloring for parallelism
    - [ ] Implement octree leaf coloring algorithm
    - [ ] Parallel repel force computation per color
    - [ ] Benchmark multi-threaded performance
  - [ ] LOW PRIORITY: Adaptive optimizations
    - [ ] Incremental neighbor updates with movement threshold
    - [ ] Adaptive force scaling based on convergence
  - [ ] Performance validation
    - [ ] Profile with N=1,000, 10,000, 100,000 point clouds
    - [ ] Measure convergence rate vs neighbor staleness
    - [ ] Document performance characteristics in user guide

## API Compatibility

**Backward compatible:** All existing code continues to work without changes.
- Default `octree=nothing` triggers auto-construction
- Default `force_model=InverseDistanceForce()` matches current behavior
- Existing `β` parameter still works (passed to default force model)

## References

- **Miotti (2023)** - Original node repulsion algorithm
- **Current implementation:** [src/repel.jl](src/repel.jl)
- **Octree utilities:** [src/octree/triangle_octree.jl](src/octree/triangle_octree.jl)
- **Geometric utilities:** [src/octree/geometric_utils.jl](src/octree/geometric_utils.jl)
- **Test failures:** [TEST_STATUS.md](TEST_STATUS.md)
- **Related branch:** `repel_integration` (mentioned in TEST_STATUS.md)
