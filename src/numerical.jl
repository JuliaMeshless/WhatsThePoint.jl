# ======================================================================
# The unitful ↔ numerical conversion boundary
# ======================================================================
#
# Algorithms in this package work on plain `SVector{N, Float64}` coordinates;
# the public API works on unitful `Point`s and `AbstractSpacing`s. This file is
# the single sanctioned crossing point between the two worlds — the only place
# where geometric quantities are stripped of (or re-attached to) their units.
#
# Every strip is *unit-aware*: values are converted to an explicit `len_unit`
# before stripping (`ustrip(len_unit, x)`), so mixed units (a cloud in `m` with
# a spacing in `mm`) convert correctly instead of silently corrupting the
# numbers. `len_unit` is derived once per entry point via [`length_unit`](@ref)
# and threaded through.
#
# Reconstruction (`from_numerical`) returns Cartesian points — the package
# operates on Euclidean manifolds and the numerical core has no notion of the
# source CRS.

"""
    length_unit(::Type{C}) where {C <: CRS} -> Unitful.Units
    length_unit(p::Point) -> Unitful.Units

Length unit of a coordinate reference system (or of a point's coordinates) —
the unit in which coordinates cross the numerical boundary.
"""
length_unit(::Type{C}) where {C <: CRS} = Unitful.unit(Meshes.lentype(C))
length_unit(p::Point) = Unitful.unit(Meshes.to(p)[1])

"""
    to_numerical(p, len_unit) -> SVector{N, Float64}
    to_numerical(x::Unitful.Length, len_unit) -> Float64
    to_numerical(x::Real, len_unit) -> Float64

Plain `Float64` coordinates of a point (dimension-generic) or value of a scalar
length, converted to `len_unit` before stripping. A plain `Real` scalar is
assumed to already be in `len_unit` and passes through. Float32 CRS
coordinates are promoted so search trees have a uniform eltype.
"""
@inline to_numerical(p::Point, len_unit) = Float64.(ustrip.(len_unit, Meshes.to(p)))
@inline to_numerical(v::Vec, len_unit) = Float64.(ustrip.(len_unit, v))
@inline to_numerical(x::Unitful.Length, len_unit) = Float64(ustrip(len_unit, x))
@inline to_numerical(x::Real, len_unit) = Float64(x)

"""
    from_numerical(x::StaticVector, len_unit) -> Point
    from_numerical(xs::AbstractVector{<:StaticVector}, len_unit) -> Vector{Point}

Reattach units to numerical coordinates, producing Cartesian point(s) in
`len_unit`.
"""
@inline from_numerical(x::StaticVector, len_unit) = Point((x .* len_unit)...)
from_numerical(xs::AbstractVector{<:StaticVector}, len_unit) =
    [from_numerical(x, len_unit) for x in xs]

"""
    numerical_spacing(spacing, len_unit) -> h

Wrap a spacing (any callable `Point -> Unitful.Length`, e.g. an
[`AbstractSpacing`](@ref)) as a numerical spacing function
`h(::SVector{3, Float64}) -> Float64` that reads coordinates and returns
values in `len_unit`. This is how spacings cross into algorithm code — wrap
once at the entry point, never per evaluation site.
"""
function numerical_spacing(spacing, len_unit)
    return c -> Float64(ustrip(len_unit, spacing(Point((c .* len_unit)...))))
end
