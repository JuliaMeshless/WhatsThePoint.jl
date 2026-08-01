"""
    AbstractNodeGenerationAlgorithm

Abstract supertype for volume discretization algorithms
(`SlakKosec`, `VanDerSandeFornberg`, `FornbergFlyer`, `Octree`).
"""
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
mesh = import_mesh("model.stl", u"m")
boundary = PointBoundary(mesh)
cloud = discretize(boundary, 3.0m; alg=Octree(mesh))
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
    alg isa FornbergFlyer ||
        throw(ArgumentError("FornbergFlyer is the only 2D discretization algorithm; pass `alg=FornbergFlyer()` (the default) or omit `alg`."))
    cloud = PointCloud(bnd)
    new_volume =
        _discretize_volume(cloud, spacing, alg; max_points = max_points)
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

"""
    refine(cloud::PointCloud, spacing; alg::Octree, max_points=nothing) -> PointVolume

Continue the volume fill of `cloud` at a finer `spacing`, returning **only the
points that were added**. The result nests: `cloud` and the increment are
disjoint and their union is blue-noise at the new spacing, because the existing
boundary *and* volume points seed the advancing front at the new radius.

Only `Octree` supports this — it is the algorithm whose Bridson pass is seeded.

Pair with [`refine`](@ref) on a `PointBoundary` for the surface. Together they
build an oversampled least-squares system whose unknowns are `cloud` and whose
extra equations sit at the increment: every unknown keeps an equation centred on
itself, so the assembled matrix is the square system with rows appended and
`σ_min` can only improve.
"""
function refine(
        cloud::PointCloud{𝔼{3}},
        spacing::AbstractSpacing;
        alg::Octree,
        max_points::Union{Int, Nothing} = nothing,
    )
    return _discretize_volume(
        cloud, spacing, alg; max_points = max_points, seed_volume = true
    )
end

function refine(cloud::PointCloud, spacing::Unitful.Length; kwargs...)
    return refine(cloud, ConstantSpacing(spacing); kwargs...)
end
