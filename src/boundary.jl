"""
    struct PointBoundary{M,C} <: Domain{M,C}

A boundary of points.

# Fields
- `surfaces`: Named surfaces forming the boundary

# Type Parameters
- `M <: Manifold`: The manifold type
- `C <: CRS`: The coordinate reference system
"""
struct PointBoundary{M<:Manifold,C<:CRS} <: Domain{M,C}
    surfaces::LittleDict{Symbol,AbstractSurface{M,C}}

    #basic constructor starting from surfaces
    function PointBoundary(
        surfaces::LittleDict{Symbol,AbstractSurface{M,C}},
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

"""
    PointBoundary(mesh::SimpleMesh)
Create a `PointBoundary` from a `SimpleMesh` 
by taking the centroids of its elements as points, 
and computing normals and areas accordingly.

(IMPORTANT: does not use any fancy node sampling, 
depends on the mesh's discretization, and is not guaranteed 
to be a good representation of the original geometry. 
Use with caution.)
"""
function PointBoundary(mesh::SimpleMesh)
    points = map(centroid, elements(mesh))
    normals = compute_normals(points)
    normals = map(x -> x / norm(x), normals) # normalize normals
    areas = map(Meshes.area, elements(mesh))
    return PointBoundary(points, normals, areas)
end

to(boundary::PointBoundary) = to.(points(boundary))
centroid(boundary::PointBoundary) = centroid(points(boundary))
boundingbox(boundary::PointBoundary) = boundingbox(points(boundary))

boundary(boundary::PointBoundary) = boundary
namedsurfaces(boundary::PointBoundary) = boundary.surfaces
surfaces(boundary::PointBoundary) = values(boundary.surfaces)
normal(boundary::PointBoundary) = mapreduce(normal, vcat, surfaces(boundary))
area(boundary::PointBoundary) = mapreduce(area, vcat, surfaces(boundary))

hassurface(boundary::PointBoundary, name) = haskey(namedsurfaces(boundary), name)

"""
    points(boundary::PointBoundary)

Return vector of all points from all surfaces in the boundary.
"""
points(boundary::PointBoundary) = mapreduce(points, vcat, surfaces(boundary))

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
        index <= (length(surf) + offset) && return surf[index-offset]
        offset += length(surf)
    end
    return
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
    return if !isnothing(namedsurfaces(boundary))
        println(io, "└─Surfaces")
        N = length(namedsurfaces(boundary))
        for (i, name) in enumerate(names(boundary))
            i < N ? println(io, "  ├─$(name)") : println(io, "  └─$(name)")
        end
    end
end

Base.show(io::IO, ::PointBoundary) = println(io, "PointBoundary")
