"""
    struct PointCloud{M,C} <: Domain{M,C}

A point cloud.
"""
mutable struct PointCloud{M<:Manifold,C<:CRS} <: Domain{M,C}
    points::Domain{M,C}
    surfaces::Dict{Symbol,AbstractSurface{M,C}}
    volume::PointVolume{M,C}
end

function PointCloud(part::PointPart{M,C}) where {M,C}
    vol = PointVolume{M,C}()
    return PointCloud((deepcopy(part.points)), deepcopy(part.surfaces), vol)
end

function PointCloud(filepath::String)
    points, normals, areas, _ = import_surface(filepath)
    surf = PointSurface(points, normals, areas)
    points = PointSet(points)
    M = manifold(points)
    C = crs(points)
    surfaces = Dict{Symbol,AbstractSurface{M,C}}(:surface1 => surf)
    vol = PointVolume{M,C}()
    return PointCloud(points, surfaces, vol)
end

Meshes.nelements(cloud::PointCloud) = Meshes.nelements(cloud.points)
Base.size(cloud::PointCloud) = (length(cloud),)
Base.getindex(cloud::PointCloud, name::Symbol) = cloud.surfaces[name]
Base.getindex(cloud::PointCloud, index::Int) = cloud.points[index]
function Base.setindex!(cloud::PointCloud, surf::PointSurface, name::Symbol)
    haskey(cloud.surfaces, name) && throw(ArgumentError("surface name already exists."))
    cloud.surfaces[name] = surf
    return nothing
end
function rename!(part::PointCloud, oldname::Symbol, newname::Symbol)
    error("this is bugged, do not use.")
    haskey(part.surfaces, oldname) || throw(ArgumentError("surface name does not exist."))
    haskey(part.surfaces, newname) && throw(ArgumentError("surface name already exists."))
    part.surfaces[newname] = part.surfaces[oldname]
    delete!(part.surfaces, oldname)
    return nothing
end
function Base.iterate(cloud::PointCloud, state=1)
    return state > length(cloud) ? nothing : (cloud[state], state + 1)
end

Base.view(cloud::PointCloud, range::UnitRange) = view(cloud.points, range)
Base.view(cloud::PointCloud, range::StepRange) = view(cloud.points, range)

to(cloud::PointCloud) = to.(cloud.points)
function to(surfaces::Dict{Symbol,<:AbstractSurface})
    return mapreduce(to, vcat, values(surfaces))
end
centroid(cloud::PointCloud) = centroid(PointSet(cloud.points))
boundingbox(cloud::PointCloud) = boundingbox(cloud.points)

surfaces(cloud::PointCloud) = values(cloud.surfaces)
volume(cloud::PointCloud) = cloud.volume
attributes(cloud::PointCloud) = values(cloud.attributes)

names(cloud::PointCloud) = keys(cloud.surfaces)
normals(cloud::PointCloud) = mapreduce(normals, vcat, surfaces(cloud))
areas(cloud::PointCloud) = mapreduce(areas, vcat, surfaces(cloud))

hassurface(cloud::PointCloud, name) = haskey(cloud.surfaces, name)

function generate_shadows(cloud::PointCloud, shadow::ShadowPoints)
    return cloud.shadow = mapreduce(s -> generate_shadows(s, shadow), vcat, surfaces(cloud))
end

# memory alignment
function make_memory_contiguous!(cloud::PointCloud, permutations)
    ranges = ranges_from_permutation(permutations)
    make_memory_contiguous!(cloud, permutations, ranges)
    return nothing
end

function make_memory_contiguous!(
    cloud::PointCloud, permutations, ranges::Vector{UnitRange{Int}}
)
    many_permute!(cloud.points, permutations, ranges)
    return nothing
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", cloud::PointCloud{Dim,T}) where {Dim,T}
    println(io, "PointCloud{$Dim, $T}")
    println(io, "├─Number of points: $(length(cloud))")
    has_vol = length(cloud.volume) != 0
    vert = has_vol ? "│ " : "  "
    if !isnothing(cloud.surfaces)
        char = has_vol ? "├" : "└"
        println(io, char * "─Surfaces")
        N = length(cloud.surfaces)
        for (i, name) in enumerate(keys(cloud.surfaces))
            char = i < N ? "├" : "└"
            println(io, vert * char * "─$(name)")
        end
    end
    if has_vol
        println(io, "└─Volume")
        println(io, "  └─Number of points: $(length(cloud.volume))")
    end
end

Base.show(io::IO, ::PointCloud) = println(io, "PointCloud")
