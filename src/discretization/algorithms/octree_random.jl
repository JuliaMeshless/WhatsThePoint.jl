"""
    OctreeRandom <: AbstractNodeGenerationAlgorithm

Octree-guided random point generation for volume discretization.

This algorithm uses a pre-built TriangleOctree with leaf classification to efficiently
generate random points inside the mesh. It's much faster than rejection sampling because:
- Only samples in interior/boundary regions (skips exterior entirely)
- Interior boxes need no filtering (100% acceptance rate)
- Only boundary boxes require isinside() checks

# Fields
- `octree::TriangleOctree` - Pre-built octree with leaf classification
- `boundary_oversampling::Float64` - Oversampling factor for boundary boxes (default: 2.0)

# Constructors
```julia
OctreeRandom(octree::TriangleOctree)                    # Default oversampling = 2.0
OctreeRandom(octree::TriangleOctree, oversampling)      # Custom oversampling factor
```

# Performance
- **Interior leaves**: O(1) per point (no filtering needed)
- **Boundary leaves**: O(k) per point where k ≈ 10-50 triangles
- **Much faster than bounding box rejection**: Typical acceptance 5-20% vs 90%+ here

# Usage Examples

## Basic Usage
```julia
using WhatsThePoint

# Build octree from STL file
octree = TriangleOctree("bunny.stl"; h_min=0.01, classify_leaves=true)

# Create discretization algorithm
alg = OctreeRandom(octree)

# Discretize (spacing parameter is ignored for this algorithm)
boundary = PointBoundary("bunny.stl")
cloud = discretize(boundary, 1.0u"m"; alg=alg, max_points=10_000)
println("Generated ", length(volume(cloud)), " interior points")
```

## With Custom Oversampling
```julia
# Higher oversampling (2.5×) generates more boundary candidates
# Better for thin features but slower
alg = OctreeRandom(octree, 2.5)
cloud = discretize(boundary, 1.0u"m"; alg=alg, max_points=10_000)
```

# Notes
- Requires octree built with `classify_leaves=true`
- The `spacing` parameter is **not used** (random uniform distribution)
- For spacing-based discretization, use SlakKosec or VanDerSandeFornberg
- Actual point count may be slightly less than `max_points` due to boundary filtering
- Oversampling 1.5-3.0 is recommended; higher values waste computation

# Algorithm Details

1. **Identify regions**: Separate octree leaves into interior (2) and boundary (1)
2. **Allocate points**: Distribute target count proportionally to leaf volumes
3. **Generate interior points**: Random sampling in interior boxes (no filtering)
4. **Generate boundary points**: Oversample and filter with isinside()
5. **Return**: Combined set of validated interior points

# When to Use

✅ **Good for**:
- Quick initial discretization
- Uniform random distributions
- Maximum point count targets
- Testing and prototyping

❌ **Not ideal for**:
- Spacing-controlled discretization (use SlakKosec instead)
- Adaptive refinement
- Smooth point distributions
"""
struct OctreeRandom{M <: Manifold, C <: CRS, T <: Real} <: AbstractNodeGenerationAlgorithm
    octree::TriangleOctree{M, C, T}
    boundary_oversampling::Float64
end

OctreeRandom(octree::TriangleOctree{M, C, T}) where {M, C, T} =
    OctreeRandom{M, C, T}(octree, 2.0)
OctreeRandom(octree::TriangleOctree{M, C, T}, oversampling::Real) where {M, C, T} =
    OctreeRandom{M, C, T}(octree, Float64(oversampling))

"""
    _allocate_counts_by_volume(volumes, total_count; ensure_one=false)

Allocate integer counts across leaves proportionally to `volumes` using
largest-remainder rounding, preserving exact total count.

If `ensure_one=true` and `total_count >= length(volumes)`, each leaf gets at
least one allocation before distributing the remainder by volume.
"""
function _allocate_counts_by_volume(
        volumes::Vector{T},
        total_count::Int;
        ensure_one::Bool = false,
    ) where {T <: Real}
    n = length(volumes)
    n == 0 && return Int[]
    total_count <= 0 && return zeros(Int, n)

    total_volume = sum(volumes; init = zero(T))
    weights = if total_volume > zero(T)
        volumes ./ total_volume
    else
        fill(inv(T(n)), n)
    end

    counts = zeros(Int, n)
    remaining = total_count

    if ensure_one && total_count >= n
        counts .= 1
        remaining -= n
    end

    if remaining <= 0
        return counts
    end

    expected = remaining .* weights
    base = floor.(Int, expected)
    counts .+= base

    leftover = remaining - sum(base)
    if leftover > 0
        frac = expected .- base
        idxs = sortperm(frac; rev = true)
        for i in 1:leftover
            counts[idxs[i]] += 1
        end
    end

    return counts
end

function _discretize_volume(
        cloud::PointCloud{𝔼{3}, C},
        spacing::AbstractSpacing,  # Not used - random distribution
        alg::OctreeRandom;
        max_points = 1_000,
    ) where {C}
    # Note: cloud and spacing parameters are required by the interface but not used
    # This algorithm generates random points directly from octree classification

    isnothing(alg.octree.leaf_classification) &&
        error("TriangleOctree must be built with classify_leaves=true")

    T = Float64

    # Collect interior and boundary leaves with their volumes
    interior_leaves = Int[]
    boundary_leaves = Int[]
    interior_volumes = T[]
    boundary_volumes = T[]

    for leaf_idx in all_leaves(alg.octree.tree)
        classification = alg.octree.leaf_classification[leaf_idx]

        if classification == LEAF_INTERIOR
            push!(interior_leaves, leaf_idx)
            box_sz = box_size(alg.octree.tree, leaf_idx)
            push!(interior_volumes, box_sz^3)
        elseif classification == LEAF_BOUNDARY
            push!(boundary_leaves, leaf_idx)
            box_sz = box_size(alg.octree.tree, leaf_idx)
            push!(boundary_volumes, box_sz^3)
        end
        # Skip exterior leaves (classification == 0)
    end

    # Calculate total volume
    total_interior_volume = sum(interior_volumes; init = zero(T))
    total_boundary_volume = sum(boundary_volumes; init = zero(T))
    total_volume = total_interior_volume + total_boundary_volume

    if total_volume ≈ 0
        @warn "No interior or boundary volume found in octree"
        return PointVolume(Point{𝔼{3}, C}[])
    end

    # Allocate points proportionally to volume
    n_interior = round(Int, max_points * (total_interior_volume / total_volume))
    n_boundary = max_points - n_interior

    # Account for boundary filtering - generate more to compensate
    n_boundary_samples = round(Int, n_boundary * alg.boundary_oversampling)

    interior_counts = _allocate_counts_by_volume(
        interior_volumes,
        n_interior;
        ensure_one = true,
    )
    boundary_sample_counts = _allocate_counts_by_volume(boundary_volumes, n_boundary_samples)

    # Pre-allocate result array
    raw_points = SVector{3, T}[]
    sizehint!(raw_points, max_points)

    # Generate points in interior leaves.
    # We still validate each point with signed distance to prevent
    # occasional false positives from box-level classification.
    if !isempty(interior_leaves)
        for (leaf_idx, n_leaf_points) in zip(interior_leaves, interior_counts)
            n_leaf_points == 0 && continue

            bbox_min, bbox_max = box_bounds(alg.octree.tree, leaf_idx)

            for _ in 1:n_leaf_points
                # Random point in box
                x = bbox_min[1] + rand(T) * (bbox_max[1] - bbox_min[1])
                y = bbox_min[2] + rand(T) * (bbox_max[2] - bbox_min[2])
                z = bbox_min[3] + rand(T) * (bbox_max[3] - bbox_min[3])
                pt = SVector{3, T}(x, y, z)

                if _compute_signed_distance_octree(pt, alg.octree.mesh, alg.octree.tree) < 0
                    push!(raw_points, pt)
                end
            end
        end
    end

    # Generate points in boundary leaves (with filtering)
    if !isempty(boundary_leaves)
        boundary_candidates = SVector{3, T}[]
        sizehint!(boundary_candidates, n_boundary_samples)

        for (leaf_idx, n_leaf_samples) in zip(boundary_leaves, boundary_sample_counts)
            n_leaf_samples == 0 && continue

            bbox_min, bbox_max = box_bounds(alg.octree.tree, leaf_idx)

            for _ in 1:n_leaf_samples
                # Random point in box
                x = bbox_min[1] + rand(T) * (bbox_max[1] - bbox_min[1])
                y = bbox_min[2] + rand(T) * (bbox_max[2] - bbox_min[2])
                z = bbox_min[3] + rand(T) * (bbox_max[3] - bbox_min[3])

                push!(boundary_candidates, SVector{3, T}(x, y, z))
            end
        end

        # Filter boundary candidates using octree-accelerated isinside
        for pt in boundary_candidates
            if isinside(pt, alg.octree)
                push!(raw_points, pt)
                # Stop when we have enough points
                if length(raw_points) >= max_points
                    break
                end
            end
        end
    end

    # Top up to requested count using interior leaves if boundary filtering underfills.
    # This keeps output close to max_points and reduces empty interior leaves.
    if length(raw_points) < max_points && !isempty(interior_leaves)
        missing = max_points - length(raw_points)
        cumw = cumsum(interior_volumes)
        totalw = cumw[end]

        for _ in 1:missing
            r = rand(T) * totalw
            idx = searchsortedfirst(cumw, r)
            idx = clamp(idx, 1, length(interior_leaves))
            leaf_idx = interior_leaves[idx]

            bbox_min, bbox_max = box_bounds(alg.octree.tree, leaf_idx)
            x = bbox_min[1] + rand(T) * (bbox_max[1] - bbox_min[1])
            y = bbox_min[2] + rand(T) * (bbox_max[2] - bbox_min[2])
            z = bbox_min[3] + rand(T) * (bbox_max[3] - bbox_min[3])
            pt = SVector{3, T}(x, y, z)
            _compute_signed_distance_octree(pt, alg.octree.mesh, alg.octree.tree) < 0 &&
                push!(raw_points, pt)
        end
    end

    # Convert to Point objects with proper manifold/CRS
    result_points = Point{𝔼{3}, C}[]
    sizehint!(result_points, length(raw_points))

    for pt in raw_points
        # Create point with same CRS as boundary
        push!(result_points, Point(pt[1], pt[2], pt[3]))
    end

    return PointVolume(result_points)
end
