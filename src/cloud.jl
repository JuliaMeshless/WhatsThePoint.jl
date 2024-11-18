"""
    struct PointCloud{M,C} <: Domain{M,C}

A point cloud.
"""
mutable struct PointCloud{M<:Manifold,C<:CRS} <: Domain{M,C}
    boundary::PointBoundary{M,C}
    volume::PointVolume{M,C}
end

function PointCloud(boundary::PointBoundary{M,C}) where {M,C}
    vol = PointVolume{M,C}()
    return PointCloud(deepcopy(boundary), vol)
end

PointCloud(filepath::String) = PointCloud(PointBoundary(filepath))

Base.length(cloud::PointCloud) = length(boundary(cloud)) + length(volume(cloud))
Base.size(cloud::PointCloud) = (length(cloud),)
Base.getindex(cloud::PointCloud, name::Symbol) = boundary(cloud)[name]
function Base.getindex(cloud::PointCloud, index::Int)
    b = boundary(cloud)
    v = volume(cloud)
    if index > length(cloud)
        throw(
            BoundsError(
                "attempt to access PointCloud at index [$index], but there are only $(length(cloud)) points.",
            ),
        )
    end
    if index <= length(b)
        return b[index]
    elseif index > length(b)
        return v[index - length(b)]
    end
end
function Base.setindex!(cloud::PointCloud, surf::PointSurface, name::Symbol)
    hassurface(boundary(cloud), name) &&
        throw(ArgumentError("surface name already exists."))
    boundary(cloud)[name] = surf
    return nothing
end
function Base.iterate(cloud::PointCloud, state=1)
    return state > length(cloud) ? nothing : (cloud[state], state + 1)
end

to(cloud::PointCloud) = to.(pointify(cloud))
function to(surfaces::Dict{Symbol,<:AbstractSurface})
    return mapreduce(to, vcat, values(surfaces))
end
boundary(cloud::PointCloud) = cloud.boundary
volume(cloud::PointCloud) = cloud.volume
surfaces(cloud::PointCloud) = surfaces(boundary(cloud))
names(cloud::PointCloud) = names(boundary(cloud))
normals(cloud::PointCloud) = mapreduce(normals, vcat, surfaces(cloud))
areas(cloud::PointCloud) = mapreduce(areas, vcat, surfaces(cloud))

hassurface(cloud::PointCloud, name) = hassurface(boundary(cloud), name)

function Meshes.pointify(cloud::PointCloud)
    return vcat(Meshes.pointify(boundary(cloud)), Meshes.pointify(volume(cloud)))
end
function Meshes.nelements(cloud::PointCloud)
    return Meshes.nelements(boundary(cloud)) + Meshes.nelements(volume(cloud))
end

function generate_shadows(cloud::PointCloud, shadow::ShadowPoints)
    return cloud.shadow = mapreduce(s -> generate_shadows(s, shadow), vcat, surfaces(cloud))
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", cloud::PointCloud{Dim,T}) where {Dim,T}
    println(io, "PointCloud{$Dim, $T}")
    println(io, "├─Number of points: $(length(cloud))")
    has_vol = length(cloud.volume) != 0
    vert = has_vol ? "│ " : "  "
    if !isnothing(surfaces(cloud))
        char = has_vol ? "├" : "└"
        println(io, char * "─Surfaces")
        N = length(surfaces(cloud))
        for (i, name) in enumerate(names(boundary(cloud)))
            char = i < N ? "├" : "└"
            println(io, vert * char * "─$(name)")
        end
    end
    if has_vol
        println(io, "└─Volume")
        println(io, "  └─Number of points: $(length(volume(cloud)))")
    end
end

Base.show(io::IO, ::PointCloud) = println(io, "PointCloud")
