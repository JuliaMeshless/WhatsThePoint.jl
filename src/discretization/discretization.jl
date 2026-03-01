abstract type AbstractNodeGenerationAlgorithm end

include("algorithms/fornberg_flyer.jl")
include("algorithms/vandersande_fornberg.jl")
include("algorithms/slak_kosec.jl")
include("algorithms/octree_random.jl")

"""
    discretize(bnd::PointBoundary, spacing; alg=auto, max_points=10_000_000)

Generate volume points for the given boundary and return a new PointCloud.

`spacing` can be either an `AbstractSpacing` object or a bare `Unitful.Length` value
(which will be wrapped in `ConstantSpacing`).

# Keyword Arguments
- `alg`: Discretization algorithm (default: `SlakKosec()` for 3D)
- `max_points`: Maximum number of volume points to generate

# Example
```julia
mesh = GeoIO.load("model.stl").geometry
boundary = PointBoundary(mesh)
octree = TriangleOctree(mesh; min_ratio=1e-6)
cloud = discretize(boundary, 3.0m; alg=SlakKosec(octree), max_points=100_000)
```
"""
function discretize(
    bnd::PointBoundary{𝔼{3}},
    spacing::AbstractSpacing;
    alg::AbstractNodeGenerationAlgorithm=SlakKosec(),
    max_points=10_000_000,
)
    cloud = PointCloud(bnd)
    new_volume = _discretize_volume(cloud, spacing, alg; max_points=max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

function discretize(
    bnd::PointBoundary{𝔼{2}},
    spacing::AbstractSpacing;
    alg::AbstractNodeGenerationAlgorithm=FornbergFlyer(),
    max_points=10_000_000,
)
    @warn "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it."
    cloud = PointCloud(bnd)
    new_volume =
        _discretize_volume(cloud, spacing, FornbergFlyer(); max_points=max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

"""
    discretize(cloud::PointCloud, spacing; alg=auto, max_points=10_000_000)

Generate volume points for an existing cloud and return a new PointCloud with the volume populated.
"""
function discretize(
    cloud::PointCloud,
    spacing::AbstractSpacing;
    alg::AbstractNodeGenerationAlgorithm=SlakKosec(),
    max_points=10_000_000,
)
    new_volume = _discretize_volume(cloud, spacing, alg; max_points=max_points)
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

"""
    discretize(bnd::PointBoundary{𝔼{3}}, alg::OctreeRandom; max_points=10_000_000)

Generate volume points using `OctreeRandom` without requiring a spacing parameter.

OctreeRandom generates uniformly random points and does not use spacing, so this
overload removes the need for a dummy spacing value.

# Example
```julia
mesh = GeoIO.load("bunny.stl").geometry
boundary = PointBoundary(mesh)
cloud = discretize(boundary, OctreeRandom(mesh); max_points=100_000)
```
"""
function discretize(
    bnd::PointBoundary{𝔼{3}},
    alg::OctreeRandom;
    max_points=10_000_000,
)
    cloud = PointCloud(bnd)
    new_volume = _discretize_volume(cloud, alg; max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
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
