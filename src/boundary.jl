"""
Sentinel type indicating no source mesh is stored.
"""
struct NoMesh end

"""
    struct PointBoundary{M,C,SM}

A boundary of points.

# Fields
- `surfaces`: Named surfaces forming the boundary
- `source_mesh`: Source mesh (e.g., from STL import) for octree construction, or `NoMesh()`

# Type Parameters
- `M <: Manifold`: The manifold type
- `C <: CRS`: The coordinate reference system
- `SM`: Source mesh type (`NoMesh` or a mesh type like `SimpleMesh`)
"""
struct PointBoundary{M <: Manifold, C <: CRS, SM} <: Domain{M, C}
    surfaces::LittleDict{Symbol, AbstractSurface{M, C}}
    source_mesh::SM
    function PointBoundary(
            surfaces::LittleDict{Symbol, AbstractSurface{M, C}},
            source_mesh::SM = NoMesh(),
        ) where {M <: Manifold, C <: CRS, SM}
        return new{M, C, SM}(surfaces, source_mesh)
    end
end

function PointBoundary(points, normals, areas)
    surf = PointSurface(points, normals, areas)
    M = manifold(surf)
    C = crs(surf)
    surfaces = LittleDict{Symbol, AbstractSurface{M, C}}(:surface1 => surf)
    return PointBoundary(surfaces, NoMesh())
end

function PointBoundary(points)
    normals = compute_normals(points)
    areas = zeros(length(points)) * Unitful.m^2
    return PointBoundary(points, normals, areas)
end

function PointBoundary(filepath::String)
    println("Importing surface from $filepath")
    points, normals, areas, mesh = import_surface(filepath)
    surf = PointSurface(points, normals, areas)
    M = manifold(surf)
    C = crs(surf)
    surfaces = LittleDict{Symbol, AbstractSurface{M, C}}(:surface1 => surf)
    return PointBoundary(surfaces, mesh)
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
source_mesh(boundary::PointBoundary) = boundary.source_mesh
has_source_mesh(::PointBoundary{M, C, NoMesh}) where {M, C} = false
has_source_mesh(::PointBoundary) = true

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
        index <= (length(surf) + offset) && return surf[index - offset]
        offset += length(surf)
    end
    return
end
function Base.setindex!(boundary::PointBoundary, surf::PointSurface, name::Symbol)
    hassurface(boundary, name) && throw(ArgumentError("surface name already exists."))
    namedsurfaces(boundary)[name] = surf
    return nothing
end

function Base.iterate(boundary::PointBoundary, state = 1)
    return state > length(boundary) ? nothing : (boundary[state], state + 1)
end

Base.delete!(boundary::PointBoundary, name::Symbol) = delete!(namedsurfaces(boundary), name)

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", boundary::PointBoundary{Dim, T}) where {Dim, T}
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

# TriangleOctree constructor from PointBoundary (defined here since PointBoundary must be defined first)
"""
    TriangleOctree(boundary::PointBoundary; h_min, kwargs...) -> TriangleOctree

Build an octree spatial index from a PointBoundary's stored source mesh.

This avoids reloading the STL file when the boundary was imported from file.
Requires the boundary to have been created from a file (has stored mesh).

# Arguments
- `boundary::PointBoundary`: Boundary with stored source mesh
- `h_min`: Minimum octree box size
- `kwargs...`: Additional arguments passed to TriangleOctree(mesh; ...)

# Example
```julia
boundary = PointBoundary("model.stl")  # Loads and stores mesh
octree = TriangleOctree(boundary; h_min=0.01, classify_leaves=true)
```
"""
function TriangleOctree(::PointBoundary{M, C, NoMesh}; kwargs...) where {M, C}
    throw(ArgumentError(
        "PointBoundary has no stored source mesh. " *
        "Create boundary from file (PointBoundary(\"file.stl\")) or provide mesh directly."
    ))
end

function TriangleOctree(boundary::PointBoundary; h_min, kwargs...)
    return TriangleOctree(source_mesh(boundary); h_min, kwargs...)
end
