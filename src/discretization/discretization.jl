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
- `alg`: Discretization algorithm (default: `SlakKosec()` in 3D; in 2D, an
  `Octree` built from the boundary loops — pass `alg = FornbergFlyer()` for the
  older height-field fill, which requires `ConstantSpacing`)
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
        alg::Union{Nothing, AbstractNodeGenerationAlgorithm} = nothing,
        max_points::Union{Int, Nothing} = nothing,
    )
    return discretize(PointCloud(bnd), spacing; alg, max_points)
end

"""
    _resolve_2d_alg(bnd, spacing, alg) -> AbstractNodeGenerationAlgorithm

Pick (or validate) the algorithm for a 2D boundary. Single place for the
decision, so the `PointBoundary` and `PointCloud` entry points cannot drift, and
so every rejected combination throws an instructive `ArgumentError` rather than
deferring to a raw `MethodError` inside `_discretize_volume` dispatch.
"""
_resolve_2d_alg(bnd, spacing, ::Nothing) = Octree(bnd; spacing)

_resolve_2d_alg(bnd, spacing, alg::Octree{𝔼{2}}) = alg

function _resolve_2d_alg(bnd, spacing, alg::Octree)
    throw(
        ArgumentError(
            "this `Octree` indexes 3D geometry ($(nameof(typeof(alg.geometry)))) and " *
                "cannot fill a 2D boundary — build the 2D one with `Octree(bnd; spacing)`, " *
                "which indexes the boundary loops with a `SegmentQuadtree`."
        )
    )
end

function _resolve_2d_alg(bnd, spacing, alg::FornbergFlyer)
    spacing isa ConstantSpacing || throw(
        ArgumentError(
            "FornbergFlyer only supports ConstantSpacing; got $(nameof(typeof(spacing))). " *
                "For a graded spacing in 2D use the default `Octree` (drop the `alg` " *
                "keyword, or pass `alg = Octree(bnd; spacing)`)."
        )
    )
    return alg
end

function _resolve_2d_alg(bnd, spacing, alg::AbstractNodeGenerationAlgorithm)
    throw(
        ArgumentError(
            "2D discretization supports Octree (the default) and FornbergFlyer " *
                "(ConstantSpacing only); got $(nameof(typeof(alg)))."
        )
    )
end

"""
    discretize(cloud::PointCloud, spacing; alg=auto, max_points=nothing)

Generate volume points for an existing cloud and return a new PointCloud with the volume populated.

For the `Octree` algorithm, `max_points` defaults to an automatic estimate from
the spacing integral (`∫ 1/h(x)³ dx`) when `nothing`. Other algorithms default
to 10_000_000.
"""
function discretize(
        cloud::PointCloud{𝔼{3}},
        spacing::AbstractSpacing;
        alg::AbstractNodeGenerationAlgorithm = SlakKosec(),
        max_points::Union{Int, Nothing} = nothing,
    )
    new_volume = _discretize_volume(cloud, spacing, alg; max_points = max_points)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

function discretize(
        cloud::PointCloud{𝔼{2}},
        spacing::AbstractSpacing;
        alg::Union{Nothing, AbstractNodeGenerationAlgorithm} = nothing,
        max_points::Union{Int, Nothing} = nothing,
    )
    resolved = _resolve_2d_alg(boundary(cloud), spacing, alg)
    new_volume = _discretize_volume(cloud, spacing, resolved; max_points = max_points)
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
