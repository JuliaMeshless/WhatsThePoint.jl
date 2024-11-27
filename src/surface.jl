"""
    abstract type AbstractSurface{M<:Manifold,C<:CRS} end

A surface of a [PointSurface](@ref).
"""
abstract type AbstractSurface{M<:Manifold,C<:CRS} <: Domain{M,C} end

"""
    struct SurfaceElement{M,C,N,A}

Representation of a point on a `<:PointSurface`.
"""
struct SurfaceElement{M,C,N,A} <: Geometry{M,C}
    point::Point{M,C}
    normal::N
    area::A
end

"""
    struct PointSurface{M,C} <: AbstractSurface{M,C}

This is a typical representation of a surface via points.
"""
struct PointSurface{M<:Manifold,C<:CRS,S} <: AbstractSurface{M,C}
    geoms::StructVector{SurfaceElement}
    shadow::S
    function PointSurface(geoms::StructVector{SurfaceElement}, shadow::S) where {S}
        p = first(geoms.point)
        M = manifold(p)
        C = crs(p)
        return new{M,C,S}(geoms, shadow)
    end
end

function PointSurface(
    points::AbstractVector{Point{M,C}}, normals::N, areas::A; shadow::S=nothing
) where {M<:Manifold,C<:CRS,N,A,S}
    @assert length(points) == length(normals) == length(areas) "All inputs must be same length. Got $(length(points)), $(length(normals)), $(length(areas))."
    geoms = StructArray{SurfaceElement}((points, normals, areas))
    return PointSurface(geoms, shadow)
end

function PointSurface(points::Domain, normals, areas; shadow=nothing)
    p = _get_underlying_vector(points)
    return PointSurface(p, normals, areas; shadow=shadow)
end
_get_underlying_vector(points::PointSet) = parent(points)
_get_underlying_vector(points::SubDomain) = points.domain[points.inds]

function PointSurface(points::AbstractVector{P}, normals) where {P}
    T = CoordRefSystems.mactype(crs(P))
    # TODO estimate areas
    areas = zeros(T, length(normals))
    return PointSurface(points, normals, areas)
end

function PointSurface(points::AbstractVector{P}; k::Int=5) where {P}
    normals = compute_normals(points; k=k)
    T = CoordRefSystems.mactype(crs(P))
    # TODO estimate areas
    areas = zeros(T, length(normals))
    surf = PointSurface(points, normals, areas)
    orient_normals!(surf; k=k)
    return surf
end

function PointSurface(filepath::String)
    points, normals, areas, _ = import_surface(filepath)
    return PointSurface(points, normals, areas)
end

function (s::PointSurface)(shadow::ShadowPoints)
    return PointSurface(point(s), normal(s), area(s); shadow=shadow)
end

Base.parent(surf::PointSurface) = surf.geoms
Base.firstindex(::PointSurface) = 1
Base.lastindex(surf::PointSurface) = length(surf)
Base.getindex(surf::PointSurface, index::Int) = getindex(parent(surf), index)
Base.iterate(surf::PointSurface, state=1) = iterate(parent(surf), state)

Meshes.pointify(surf::PointSurface) = point(surf)
Meshes.elements(surf::PointSurface) = (elem for elem in parent(surf))
Meshes.nelements(surf::PointSurface) = length(parent(surf))
Meshes.centroid(surf::PointSurface) = centroid(PointSet(point(surf)))
Meshes.boundingbox(surf::PointSurface) = boundingbox(point(surf))

ChunkSplitters.is_chunkable(::PointSurface) = true

Base.view(surf::PointSurface, range::UnitRange) = view(parent(surf), range)
Base.view(surf::PointSurface, range::StepRange) = view(parent(surf), range)

to(surf::PointSurface) = to.(parent(surf).point)
point(surf::PointSurface) = parent(surf).point
normal(surf::PointSurface) = parent(surf).normal
area(surf::PointSurface) = parent(surf).area

function generate_shadows(surf::PointSurface, shadow::ShadowPoints)
    return generate_shadows(to(surf), normal(surf), shadow)
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", surf::PointSurface{M,C}) where {M,C}
    println(io, "PointSurface{$M,$C}")
    println(io, "├─Number of points: $(length(surf))")

    a = area(surf)
    if isnothing(a)
        println(io, "├─Area: not defined")
    else
        println(io, "├─Area: $(sum(a))")
    end

    s = surf.shadow
    _shadow_order(::ShadowPoints{O}) where {O} = O
    shadow_str = "└─Shadow:"
    if isnothing(s)
        println(io, "$shadow_str none")
    else
        println(io, "$shadow_str $(_shadow_order(s))")
    end
    return nothing
end

Base.show(io::IO, ::PointSurface) = println(io, "PointSurface")
