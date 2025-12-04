"""
    struct PointBoundary{Dim,T,P}

A boundary of points.
"""
struct PointBoundary{M<:Manifold,C<:CRS} <: Domain{M,C}
    surfaces::LittleDict{Symbol,AbstractSurface{M,C}}
    function PointBoundary(
        surfaces::LittleDict{Symbol,AbstractSurface{M,C}}
    ) where {M<:Manifold,C<:CRS}
        return new{M,C}(surfaces)
    end
end

function PointBoundary(points, normals, areas)
    surf = PointSurface(points, normals, areas)
    M = manifold(surf)
    C = crs(surf)
    surfaces = LittleDict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    return PointBoundary(surfaces)
end

function PointBoundary(points)
    normals = compute_normals(points)
    areas = zeros(length(points)) * Unitful.m^2
    return PointBoundary(points, normals, areas)
end

function PointBoundary(filepath::String)
    println("Importing surface from $filepath")
    points, normals, areas, _ = import_surface(filepath)
    surf = PointSurface(points, normals, areas)
    M = manifold(surf)
    C = crs(surf)
    surfaces = LittleDict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    return PointBoundary(surfaces)
end

to(boundary::PointBoundary) = to.(Meshes.pointify(boundary))
centroid(boundary::PointBoundary) = centroid(PointSet(Meshes.pointify(boundary)))
boundingbox(boundary::PointBoundary) = boundingbox(Meshes.pointify(boundary))

boundary(boundary::PointBoundary) = boundary
namedsurfaces(boundary::PointBoundary) = boundary.surfaces
surfaces(boundary::PointBoundary) = values(boundary.surfaces)
normal(boundary::PointBoundary) = mapreduce(normal, vcat, surfaces(boundary))
area(boundary::PointBoundary) = mapreduce(area, vcat, surfaces(boundary))

hassurface(boundary::PointBoundary, name) = haskey(namedsurfaces(boundary), name)

Meshes.pointify(boundary::PointBoundary) = mapreduce(pointify, vcat, surfaces(boundary))
Meshes.nelements(boundary::PointBoundary) = length(boundary)

Base.length(boundary::PointBoundary) = sum(length, surfaces(boundary))
Base.names(boundary::PointBoundary) = collect(keys(namedsurfaces(boundary)))
Base.size(boundary::PointBoundary) = (length(boundary),)
Base.getindex(boundary::PointBoundary, name::Symbol) = namedsurfaces(boundary)[name]
function Base.getindex(boundary::PointBoundary, index::Int)
    if index > length(boundary)
        throw(
            BoundsError(
                "attempt to access PointBoundary at index [$index], but there are only $(length(boundary)) points.",
            ),
        )
    end
    offset = 0
    for surf in surfaces(boundary)
        index <= (length(surf) + offset) && return surf[index - offset]
        offset += length(surf)
    end
end
function Base.setindex!(boundary::PointBoundary, surf::PointSurface, name::Symbol)
    hassurface(boundary, name) && throw(ArgumentError("surface name already exists."))
    namedsurfaces(boundary)[name] = surf
    return nothing
end

function Base.iterate(boundary::PointBoundary, state=1)
    return state > length(boundary) ? nothing : (boundary[state], state + 1)
end

Base.delete!(boundary::PointBoundary, name::Symbol) = delete!(namedsurfaces(boundary), name)

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", boundary::PointBoundary{Dim,T}) where {Dim,T}
    println(io, "PointBoundary{$Dim, $T}")
    println(io, "├─$(length(boundary)) points")
    if !isnothing(namedsurfaces(boundary))
        println(io, "└─Surfaces")
        N = length(namedsurfaces(boundary))
        for (i, name) in enumerate(names(boundary))
            i < N ? println(io, "  ├─$(name)") : println(io, "  └─$(name)")
        end
    end
end

Base.show(io::IO, ::PointBoundary) = println(io, "PointBoundary")
