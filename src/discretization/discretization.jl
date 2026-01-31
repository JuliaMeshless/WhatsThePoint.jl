abstract type AbstractNodeGenerationAlgorithm end

include("algorithms/fornberg_flyer.jl")
include("algorithms/vandersande_fornberg.jl")
include("algorithms/slak_kosec.jl")

"""
    discretize(bnd::PointBoundary, spacing; alg=auto, max_points=10_000_000, use_octree=false, octree_h_min=nothing)

Generate volume points for the given boundary and return a new PointCloud.

`spacing` can be either an `AbstractSpacing` object or a bare `Unitful.Length` value
(which will be wrapped in `ConstantSpacing`).

# Keyword Arguments
- `alg`: Discretization algorithm (default: `SlakKosec()` for 3D)
- `max_points`: Maximum number of volume points to generate
- `use_octree`: If `true`, builds octree from boundary's source mesh for fast isinside queries
- `octree_h_min`: Minimum octree cell size (required if `use_octree=true` and not using ConstantSpacing)
  For `ConstantSpacing`, defaults to `0.1 * Δx` if not provided

# Example
```julia
boundary = PointBoundary("model.stl")
cloud = discretize(boundary, 3.0m; use_octree=true, max_points=100_000)
```
"""
function discretize(
        bnd::PointBoundary{𝔼{3}},
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = SlakKosec(),
        max_points = 10_000_000,
        use_octree::Bool = false,
        octree_h_min = nothing,
    )
    actual_alg = _maybe_add_octree(bnd, spacing, alg, use_octree, octree_h_min)
    cloud = PointCloud(bnd)
    new_volume = _discretize_volume(cloud, spacing, actual_alg; max_points = max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

function discretize(
        bnd::PointBoundary{𝔼{2}},
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = FornbergFlyer(),
        max_points = 10_000_000,
    )
    @warn "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it."
    cloud = PointCloud(bnd)
    new_volume =
        _discretize_volume(cloud, spacing, FornbergFlyer(); max_points = max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

"""
    discretize(cloud::PointCloud, spacing; alg=auto, max_points=10_000_000, use_octree=false, octree_h_min=nothing)

Generate volume points for an existing cloud and return a new PointCloud with the volume populated.
"""
function discretize(
        cloud::PointCloud,
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = SlakKosec(),
        max_points = 10_000_000,
        use_octree::Bool = false,
        octree_h_min = nothing,
    )
    actual_alg = _maybe_add_octree(boundary(cloud), spacing, alg, use_octree, octree_h_min)
    new_volume = _discretize_volume(cloud, spacing, actual_alg; max_points = max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

function calculate_ninit(cloud::PointCloud{𝔼{3}}, s::VariableSpacing)
    min_s = s(first(points(cloud)))
    bbox = boundingbox(cloud)
    extent = bbox.max - bbox.min
    return (ceil(Int, extent[1] * 10 / min_s), ceil(Int, extent[2] * 10 / min_s))
end

function calculate_ninit(cloud::PointCloud{𝔼{3}}, s::ConstantSpacing)
    bbox = boundingbox(cloud)
    extent = bbox.max - bbox.min
    return (ceil(Int, extent[1] * 10 / s.Δx), ceil(Int, extent[2] * 10 / s.Δx))
end

function calculate_ninit(cloud::PointCloud{𝔼{2}}, s::ConstantSpacing)
    bbox = boundingbox(cloud)
    extent = bbox.max - bbox.min
    return ceil(Int, extent[1] * 10 / s.Δx)
end

# Convenience overloads: accept bare Unitful.Length and wrap in ConstantSpacing
function discretize(
        bnd::PointBoundary,
        spacing::Unitful.Length;
        kwargs...,
    )
    return discretize(bnd, ConstantSpacing(spacing); kwargs...)
end

function discretize(
        cloud::PointCloud,
        spacing::Unitful.Length;
        kwargs...,
    )
    return discretize(cloud, ConstantSpacing(spacing); kwargs...)
end

# Internal helper: builds octree for SlakKosec if use_octree=true
function _maybe_add_octree(
        bnd::PointBoundary,
        spacing::AbstractSpacing,
        alg::SlakKosec,
        use_octree::Bool,
        octree_h_min,
    )
    !use_octree && return alg
    !isnothing(alg.octree) && return alg  # Already has octree

    if !has_source_mesh(bnd)
        throw(ArgumentError(
            "use_octree=true requires boundary to have source mesh. " *
            "Create boundary from file (PointBoundary(\"file.stl\")) or build octree manually."
        ))
    end

    h_min = _compute_h_min(spacing, octree_h_min)
    octree = TriangleOctree(bnd; h_min = h_min, classify_leaves = true)
    return SlakKosec(alg.n, octree)
end

# Fallback for other algorithms that don't support octree
function _maybe_add_octree(
        ::PointBoundary,
        ::AbstractSpacing,
        alg::AbstractNodeGenerationAlgorithm,
        use_octree::Bool,
        ::Any,
    )
    use_octree && @warn "use_octree is only supported with SlakKosec algorithm, ignoring"
    return alg
end

# Compute h_min from spacing or explicit value
_compute_h_min(::AbstractSpacing, h_min) = h_min
function _compute_h_min(spacing::ConstantSpacing, h_min)
    isnothing(h_min) || return h_min
    return 0.1 * ustrip(spacing.Δx)
end
