abstract type AbstractNodeGenerationAlgorithm end

# Discretization algorithms
include("algorithms/fornberg_flyer.jl")
include("algorithms/vandersande_fornberg.jl")
include("algorithms/slak_kosec.jl")
include("algorithms/octree.jl")
include("spacing_guidance.jl")

"""
    discretize(bnd::PointBoundary, spacing; alg=auto, max_points=nothing)

Generate volume points for the given boundary and return a new PointCloud.

`spacing` can be either an `AbstractSpacing` object or a bare `Unitful.Length` value
(which will be wrapped in `ConstantSpacing`).

# Keyword Arguments
- `alg`: Discretization algorithm (default: `SlakKosec()` for 3D)
- `max_points`: Maximum number of volume points to generate. For the `Octree`
  algorithm, defaults to an automatic estimate from the spacing integral
  (`∫ 1/h(x)³ dx`) when `nothing`; other algorithms default to 10_000_000.

# Example
```julia
mesh = GeoIO.load("model.stl").geometry
boundary = PointBoundary(mesh)
octree = TriangleOctree(mesh; min_ratio=1e-6)
cloud = discretize(boundary, 3.0m; alg=SlakKosec(octree), max_points=100_000)
```

!!! note
    WhatsThePoint's `discretize` generates volume fill points from a boundary.
    This differs from Meshes.jl's `discretize` which converts continuous geometry
    into a mesh. No dispatch collision exists — argument types are distinct.
"""
function discretize(
        bnd::PointBoundary{𝔼{3}},
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = SlakKosec(),
        max_points::Union{Int, Nothing} = nothing,
    )
    # The Octree algorithm emits Float64 volume points; a Float32 boundary
    # (binary STL precision) would make the cloud assembly below fail.
    alg isa Octree && (bnd = _ensure_float64_boundary(bnd))
    cloud = PointCloud(bnd)
    new_volume = _discretize_volume(cloud, spacing, alg; max_points = max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

function discretize(
        bnd::PointBoundary{𝔼{2}},
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = FornbergFlyer(),
        max_points::Union{Int, Nothing} = nothing,
    )
    @warn "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it."
    cloud = PointCloud(bnd)
    new_volume =
        _discretize_volume(cloud, spacing, FornbergFlyer(); max_points = max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

"""
    discretize(cloud::PointCloud, spacing; alg=auto, max_points=nothing)

Generate volume points for an existing cloud and return a new PointCloud with the volume populated.

For the `Octree` algorithm, `max_points` defaults to an automatic estimate from
the spacing integral (`∫ 1/h(x)³ dx`) when `nothing`. Other algorithms default
to 10_000_000.
"""
function discretize(
        cloud::PointCloud,
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = SlakKosec(),
        max_points::Union{Int, Nothing} = nothing,
    )
    new_volume = _discretize_volume(cloud, spacing, alg; max_points = max_points)
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
