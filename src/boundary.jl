"""
    struct PointBoundary{Dim,T,P}

A boundary of points.
"""
struct PointBoundary{M<:Manifold,C<:CRS} <: Domain{M,C}
    points::Domain{M,C}
    surfaces::Dict{Symbol,AbstractSurface{M,C}}
    function PointBoundary(
        points::Domain{M,C}, surfaces::Dict{Symbol,AbstractSurface{M,C}}
    ) where {M<:Manifold,C<:CRS}
        return new{M,C}(points, surfaces)
    end
end

function PointBoundary(points, normals, areas)
    surf = PointSurface(points, normals, areas)
    M = manifold(surf)
    C = crs(surf)
    surfaces = Dict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    return PointBoundary(PointSet(points), surfaces)
end

function PointBoundary(points)
    normals = compute_normals(points)
    areas = zeros(length(points)) * Unitful.m^2
    return PointBoundary(points, normals, areas)
end

function PointBoundary(filepath::String)
    points, normals, areas, _ = import_surface(filepath)
    surf = PointSurface(points, normals, areas)
    pointset = PointSet(points)
    M = manifold(pointset)
    C = crs(pointset)
    surfaces = Dict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    return PointBoundary(pointset, surfaces)
end

to(boundary::PointBoundary) = to.(boundary.points)
centroid(boundary::PointBoundary) = centroid(PointSet(boundary.points))
boundingbox(boundary::PointBoundary) = boundingbox(boundary.points)

boundary(boundary::PointBoundary) = boundary
surfaces(boundary::PointBoundary) = values(boundary.surfaces)
normals(boundary::PointBoundary) = mapreduce(normals, vcat, surfaces(boundary))
areas(boundary::PointBoundary) = mapreduce(areas, vcat, surfaces(boundary))

hassurface(boundary::PointBoundary, name) = haskey(boundary.surfaces, name)

Meshes.pointify(boundary::PointBoundary) = Meshes.pointify(boundary.points)
Meshes.nelements(boundary::PointBoundary) = Meshes.nelements(boundary.points)

Base.names(boundary::PointBoundary) = keys(boundary.surfaces)
Base.size(boundary::PointBoundary) = (length(boundary),)
Base.getindex(boundary::PointBoundary, name::Symbol) = boundary.surfaces[name]
Base.getindex(boundary::PointBoundary, index::Int) = boundary.points[index]
function Base.setindex!(boundary::PointBoundary, surf::PointSurface, name::Symbol)
    hassurface(boundary, name) && throw(ArgumentError("surface name already exists."))
    boundary.surfaces[name] = surf
    return nothing
end

Base.view(boundary::PointBoundary, range::UnitRange) = view(boundary.points, range)
Base.view(boundary::PointBoundary, range::StepRange) = view(boundary.points, range)

function Base.iterate(boundary::PointBoundary, state=1)
    return state > length(boundary) ? nothing : (boundary[state], state + 1)
end

Base.delete!(boundary::PointBoundary, name::Symbol) = delete!(boundary.surfaces, name)

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", boundary::PointBoundary{Dim,T}) where {Dim,T}
    println(io, "PointBoundary{$Dim, $T}")
    println(io, "├─$(length(boundary.points)) points")
    if !isnothing(surfaces(boundary))
        println(io, "└─Surfaces")
        N = length(surfaces(boundary))
        for (i, name) in enumerate(names(boundary))
            i < N ? println(io, "  ├─$(name)") : println(io, "  └─$(name)")
        end
    end
end

Base.show(io::IO, ::PointBoundary) = println(io, "PointBoundary")
