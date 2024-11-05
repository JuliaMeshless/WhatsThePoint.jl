"""
    struct PointPart{Dim,T,P}

A CAD part.
"""
struct PointPart{M<:Manifold,C<:CRS} <: Domain{M,C}
    points::Domain{M,C}
    surfaces::Dict{Symbol,AbstractSurface{M,C}}
    function PointPart(
        points::Domain{M,C}, surfaces::Dict{Symbol,AbstractSurface{M,C}}
    ) where {M<:Manifold,C<:CRS}
        return new{M,C}(points, surfaces)
    end
end

function PointPart(points, normals, areas)
    surf = PointSurface(points, normals, areas)
    M = manifold(surf)
    C = crs(surf)
    surfaces = Dict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    return PointPart(PointSet(points), surfaces)
end

function PointPart(points)
    normals = compute_normals(points)
    areas = zeros(length(points))
    return PointPart(points, normals, areas)
end

function PointPart(filepath::String)
    points, normals, areas, _ = import_surface(filepath)
    surf = PointSurface(points, normals, areas)
    pointset = PointSet(points)
    M = manifold(pointset)
    C = crs(pointset)
    surfaces = Dict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    return PointPart(pointset, surfaces)
end

Meshes.nelements(part::PointPart) = Meshes.nelements(part.points)
Base.size(part::PointPart) = (length(part),)
Base.getindex(part::PointPart, name::Symbol) = part.surfaces[name]
Base.getindex(part::PointPart, index::Int) = part.points[index]
function Base.setindex!(part::PointPart, surf::PointSurface, name::Symbol)
    haskey(part.surfaces, name) && throw(ArgumentError("surface name already exists."))
    part.surfaces[name] = surf
    return nothing
end

Base.view(part::PointPart, range::UnitRange) = view(part.points, range)
Base.view(part::PointPart, range::StepRange) = view(part.points, range)

function rename!(part::PointPart, oldname::Symbol, newname::Symbol)
    error("this is bugged, do not use.")
    haskey(part.surfaces, oldname) || throw(ArgumentError("surface name does not exist."))
    haskey(part.surfaces, newname) && throw(ArgumentError("surface name already exists."))
    part.surfaces[newname] = part.surfaces[oldname]
    delete!(part.surfaces, oldname)
    return nothing
end
function Base.iterate(part::PointPart, state=1)
    return state > length(part) ? nothing : (part[state], state + 1)
end
to(part::PointPart) = to.(part.points)
centroid(part::PointPart) = centroid(PointSet(part.points))
boundingbox(part::PointPart) = boundingbox(part.points)

surfaces(part::PointPart) = values(part.surfaces)

names(part::PointPart) = keys(part.surfaces)
normals(part::PointPart) = mapreduce(normals, vcat, surfaces(part))
areas(part::PointPart) = mapreduce(areas, vcat, surfaces(part))

hassurface(part::PointPart, name) = haskey(part.surfaces, name)

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", part::PointPart{Dim,T}) where {Dim,T}
    println(io, "PointPart{$Dim, $T}")
    println(io, "├─Number of points: $(length(part.points))")
    if !isnothing(part.surfaces)
        println(io, "└─Surfaces")
        N = length(part.surfaces)
        for (i, name) in enumerate(keys(part.surfaces))
            i < N ? println(io, "  ├─$(name)") : println(io, "  └─$(name)")
        end
    end
end

Base.show(io::IO, ::PointPart) = println(io, "PointPart")
