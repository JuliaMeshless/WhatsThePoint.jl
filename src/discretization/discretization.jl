"""
    AbstractNodeGenerationAlgorithm

Abstract supertype for volume discretization algorithms
(`SlakKosec`, `VanDerSandeFornberg`, `FornbergFlyer`, `Orthtree`).
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
  `Orthtree` built from the boundary loops — pass `alg = FornbergFlyer()` for the
  older height-field fill, which requires `ConstantSpacing`)
- `max_points`: Maximum number of volume points to generate. For the `Orthtree`
  algorithm, defaults to an automatic estimate from the spacing integral
  (`∫ 1/h(x)ᴺ dx`, `N` the spatial dimension) when `nothing`; other algorithms
  default to 10_000_000.

# Example
```julia
mesh = import_mesh("model.stl", u"m")
boundary = PointBoundary(mesh)
cloud = discretize(boundary, 3.0m; alg=Orthtree(mesh))
```

!!! note
    WhatsThePoint's `discretize` generates volume fill points from a boundary.
    This differs from Meshes.jl's `discretize` which converts continuous geometry
    into a mesh. No dispatch collision exists — argument types are distinct.
"""
function discretize(
        bnd::PointBoundary,
        spacing::AbstractSpacing;
        alg::Union{Nothing, AbstractNodeGenerationAlgorithm} = nothing,
        max_points::Union{Int, Nothing} = nothing,
    )
    return discretize(PointCloud(bnd), spacing; alg, max_points)
end

"""
    _resolve_alg(bnd, spacing, alg) -> AbstractNodeGenerationAlgorithm

Pick (or validate) the algorithm for a boundary, dispatching on its manifold.
Single place for the decision, so the `PointBoundary` and `PointCloud` entry
points cannot drift, and so every rejected combination throws an instructive
`ArgumentError` rather than deferring to a raw `MethodError` inside
`_discretize_volume` dispatch.
"""
_resolve_alg(bnd::PointBoundary{𝔼{2}}, spacing, ::Nothing) = Orthtree(bnd; spacing)

_resolve_alg(::PointBoundary{𝔼{3}}, spacing, ::Nothing) = SlakKosec()

_resolve_alg(::PointBoundary{𝔼{2}}, spacing, alg::Orthtree{𝔼{2}}) = alg

_resolve_alg(::PointBoundary{𝔼{3}}, spacing, alg::Orthtree{𝔼{3}}) = alg

function _resolve_alg(::PointBoundary{𝔼{2}}, spacing, alg::Orthtree)
    throw(
        ArgumentError(
            "this `Orthtree` indexes 3D geometry ($(nameof(typeof(alg.geometry)))) and " *
                "cannot fill a 2D boundary — build the 2D one with `Orthtree(bnd; spacing)`, " *
                "which indexes the boundary loops with a `SegmentQuadtree`."
        )
    )
end

function _resolve_alg(::PointBoundary{𝔼{3}}, spacing, alg::Orthtree)
    throw(
        ArgumentError(
            "this `Orthtree` indexes 2D geometry ($(nameof(typeof(alg.geometry)))) and " *
                "cannot fill a 3D boundary — build the 3D one with `Orthtree(mesh; spacing)`, " *
                "which indexes the surface mesh with a `TriangleOctree`."
        )
    )
end

function _resolve_alg(::PointBoundary{𝔼{2}}, spacing, alg::FornbergFlyer)
    spacing isa ConstantSpacing || throw(
        ArgumentError(
            "FornbergFlyer only supports ConstantSpacing; got $(nameof(typeof(spacing))). " *
                "For a graded spacing in 2D use the default `Orthtree` (drop the `alg` " *
                "keyword, or pass `alg = Orthtree(bnd; spacing)`)."
        )
    )
    return alg
end

_resolve_alg(::PointBoundary{𝔼{3}}, spacing, alg::SlakKosec) = alg

function _resolve_alg(::PointBoundary{𝔼{3}}, spacing, alg::VanDerSandeFornberg)
    spacing isa ConstantSpacing || throw(
        ArgumentError(
            "VanDerSandeFornberg only supports ConstantSpacing; got " *
                "$(nameof(typeof(spacing))). For a graded spacing in 3D use `SlakKosec` " *
                "(the default) or `Orthtree(mesh; spacing)`."
        )
    )
    return alg
end

function _resolve_alg(::PointBoundary{𝔼{2}}, spacing, alg::AbstractNodeGenerationAlgorithm)
    throw(
        ArgumentError(
            "2D discretization supports Orthtree (the default) and FornbergFlyer " *
                "(ConstantSpacing only); got $(nameof(typeof(alg)))."
        )
    )
end

function _resolve_alg(::PointBoundary{𝔼{3}}, spacing, alg::AbstractNodeGenerationAlgorithm)
    throw(
        ArgumentError(
            "3D discretization supports SlakKosec (the default), VanDerSandeFornberg " *
                "(ConstantSpacing only), and Orthtree; got $(nameof(typeof(alg)))."
        )
    )
end

function _resolve_alg(bnd::PointBoundary, spacing, alg)
    throw(
        ArgumentError(
            "discretize supports 2D and 3D Euclidean boundaries; got $(typeof(bnd))."
        )
    )
end

"""
    discretize(cloud::PointCloud, spacing; alg=auto, max_points=nothing)

Generate volume points for an existing cloud and return a new PointCloud with the volume populated.

For the `Orthtree` algorithm, `max_points` defaults to an automatic estimate from
the spacing integral (`∫ 1/h(x)ᴺ dx`, `N` the spatial dimension) when `nothing`. Other algorithms default
to 10_000_000.
"""
function discretize(
        cloud::PointCloud,
        spacing::AbstractSpacing;
        alg::Union{Nothing, AbstractNodeGenerationAlgorithm} = nothing,
        max_points::Union{Int, Nothing} = nothing,
    )
    resolved = _resolve_alg(boundary(cloud), spacing, alg)
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
