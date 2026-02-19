"""
    OctreeRandom <: AbstractNodeGenerationAlgorithm

Octree-guided random point generation for volume discretization.

This algorithm uses a `TriangleOctree` with leaf classification to efficiently
generate random points inside the mesh. It's much faster than rejection sampling because:
- Only samples in interior/boundary regions (skips exterior entirely)
- Interior boxes need no filtering (100% acceptance rate)
- Only boundary boxes require isinside() checks

# Fields
- `octree::TriangleOctree` - Octree with leaf classification
- `boundary_oversampling::Float64` - Oversampling factor for boundary boxes (default: 2.0)
- `verify_interior::Bool` - Per-point signed distance verification for interior leaves (default: false)

# Constructors
```julia
# From mesh (recommended) — builds octree automatically with classified leaves
OctreeRandom(mesh::SimpleMesh)                          # Auto h_min
OctreeRandom(mesh; h_min=0.01, boundary_oversampling=2.0)  # Custom h_min

# From file path — loads mesh then builds octree
OctreeRandom("bunny.stl")
OctreeRandom("bunny.stl"; h_min=0.01)

# From pre-built octree (advanced)
OctreeRandom(octree::TriangleOctree)
OctreeRandom(octree, oversampling; verify_interior=false)
```

# Performance
- **Interior leaves**: O(1) per point (no filtering needed)
- **Boundary leaves**: O(k) per point where k ≈ 10-50 triangles
- **Much faster than bounding box rejection**: Typical acceptance 5-20% vs 90%+ here

# Usage Examples

## Recommended Usage
```julia
using WhatsThePoint

mesh = GeoIO.load("bunny.stl").geometry
boundary = PointBoundary(mesh)
cloud = discretize(boundary, OctreeRandom(mesh); max_points=100_000)
```

## With Spacing (backward compatible)
```julia
octree = TriangleOctree(mesh; h_min=0.01, classify_leaves=true)
cloud = discretize(boundary, 1.0u"m"; alg=OctreeRandom(octree), max_points=10_000)
```

## With Custom Oversampling
```julia
alg = OctreeRandom(mesh; boundary_oversampling=2.5)
cloud = discretize(boundary, alg; max_points=10_000)
```

# Notes
- The mesh convenience constructor always sets `classify_leaves=true`
- When `h_min` is omitted, it is computed as `bbox_diagonal / (2 * cbrt(n_triangles))`
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

**Good for**:
- Quick initial discretization
- Uniform random distributions
- Maximum point count targets
- Testing and prototyping

**Not ideal for**:
- Spacing-controlled discretization (use SlakKosec instead)
- Adaptive refinement
- Smooth point distributions
"""
struct OctreeRandom{M <: Manifold, C <: CRS, T <: Real} <: AbstractNodeGenerationAlgorithm
    octree::TriangleOctree{M, C, T}
    boundary_oversampling::Float64
    verify_interior::Bool
end

@inline function _rand_point_in_box(bbox_min::SVector{3, T}, bbox_max::SVector{3, T}) where {T}
    return bbox_min + rand(SVector{3, T}) .* (bbox_max - bbox_min)
end

OctreeRandom(octree::TriangleOctree{M, C, T}) where {M, C, T} =
    OctreeRandom{M, C, T}(octree, 2.0, false)
function OctreeRandom(
        octree::TriangleOctree{M, C, T},
        oversampling::Real;
        verify_interior::Bool = false,
    ) where {M, C, T}
    oversampling > 0 || throw(ArgumentError("boundary_oversampling must be positive, got $oversampling"))
    return OctreeRandom{M, C, T}(octree, Float64(oversampling), verify_interior)
end

"""
    _auto_h_min(::Type{T}, mesh::SimpleMesh) where {T}

Compute a default `h_min` for octree construction based on mesh geometry.

Heuristic: `bbox_diagonal / (2 * cbrt(n_triangles))`. Scales leaf size to the
mesh's characteristic triangle spacing so the octree resolves surface detail
without excessive subdivision.
"""
function _auto_h_min(::Type{T}, mesh::SimpleMesh) where {T}
    bbox_min, bbox_max = _compute_bbox(T, mesh)
    diagonal = norm(bbox_max - bbox_min)
    n = Meshes.nelements(mesh)
    return diagonal / (2 * cbrt(T(n)))
end

"""
    OctreeRandom(mesh::SimpleMesh; h_min=nothing, max_triangles_per_box=50,
                 boundary_oversampling=2.0, verify_interior=false, verify_orientation=true)

Convenience constructor that builds a classified `TriangleOctree` internally.

When `h_min` is not provided, an automatic value is computed from the mesh's
bounding box diagonal and triangle count.

# Example
```julia
mesh = GeoIO.load("bunny.stl").geometry
alg = OctreeRandom(mesh)
cloud = discretize(boundary, alg; max_points=100_000)
```
"""
function OctreeRandom(
        mesh::SimpleMesh{M, C};
        h_min = nothing,
        max_triangles_per_box::Int = 50,
        boundary_oversampling::Real = 2.0,
        verify_interior::Bool = false,
        verify_orientation::Bool = true,
    ) where {M <: Manifold, C <: CRS}
    T = Float64
    h_min_val = isnothing(h_min) ? _auto_h_min(T, mesh) : T(ustrip(h_min))
    octree = TriangleOctree(mesh;
        h_min = h_min_val,
        max_triangles_per_box,
        classify_leaves = true,
        verify_orientation,
    )
    return OctreeRandom(octree, boundary_oversampling; verify_interior)
end

"""
    OctreeRandom(filepath::String; kwargs...)

Load a mesh from `filepath` and build an `OctreeRandom` algorithm.
Accepts the same keyword arguments as `OctreeRandom(mesh; ...)`.

# Example
```julia
alg = OctreeRandom("bunny.stl")
cloud = discretize(boundary, alg; max_points=100_000)
```
"""
function OctreeRandom(filepath::String; kwargs...)
    geo = GeoIO.load(filepath)
    return OctreeRandom(geo.geometry; kwargs...)
end

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

"""
    _collect_classified_leaves(octree::TriangleOctree)

Collect interior and boundary leaves with their volumes from a classified octree.

Returns `(interior_leaves, interior_volumes, boundary_leaves, boundary_volumes)`.
"""
function _collect_classified_leaves(octree::TriangleOctree)
    T = Float64
    interior_leaves = Int[]
    boundary_leaves = Int[]
    interior_volumes = T[]
    boundary_volumes = T[]

    for leaf_idx in all_leaves(octree.tree)
        classification = octree.leaf_classification[leaf_idx]

        if classification == LEAF_INTERIOR
            push!(interior_leaves, leaf_idx)
            push!(interior_volumes, box_size(octree.tree, leaf_idx)^3)
        elseif classification == LEAF_BOUNDARY
            push!(boundary_leaves, leaf_idx)
            push!(boundary_volumes, box_size(octree.tree, leaf_idx)^3)
        end
    end

    return interior_leaves, interior_volumes, boundary_leaves, boundary_volumes
end

# Delegate spacing-based call to spacing-free version (spacing is unused by OctreeRandom)
function _discretize_volume(
        cloud::PointCloud{𝔼{3}, C},
        ::AbstractSpacing,
        alg::OctreeRandom;
        max_points = 1_000,
    ) where {C}
    return _discretize_volume(cloud, alg; max_points)
end

function _discretize_volume(
        cloud::PointCloud{𝔼{3}, C},
        alg::OctreeRandom;
        max_points = 1_000,
    ) where {C}
    isnothing(alg.octree.leaf_classification) &&
        error("TriangleOctree must be built with classify_leaves=true")

    T = Float64

    interior_leaves, interior_volumes, boundary_leaves, boundary_volumes =
        _collect_classified_leaves(alg.octree)

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
    # Conservative classification already ensures correctness; per-point signed
    # distance check is optional and expensive.
    if !isempty(interior_leaves)
        for (leaf_idx, n_leaf_points) in zip(interior_leaves, interior_counts)
            n_leaf_points == 0 && continue

            bbox_min, bbox_max = box_bounds(alg.octree.tree, leaf_idx)

            for _ in 1:n_leaf_points
                pt = _rand_point_in_box(bbox_min, bbox_max)

                if alg.verify_interior
                    if _compute_signed_distance_octree(pt, alg.octree.mesh, alg.octree.tree) < 0
                        push!(raw_points, pt)
                    end
                else
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
                push!(boundary_candidates, _rand_point_in_box(bbox_min, bbox_max))
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
            pt = _rand_point_in_box(bbox_min, bbox_max)

            if alg.verify_interior
                _compute_signed_distance_octree(pt, alg.octree.mesh, alg.octree.tree) < 0 &&
                    push!(raw_points, pt)
            else
                push!(raw_points, pt)
            end
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
