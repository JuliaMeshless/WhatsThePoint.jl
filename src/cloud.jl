"""
    struct PointCloud{M,C,T} <: Domain{M,C}

A point cloud with optional topology (connectivity).

# Type Parameters
- `M<:Manifold` - manifold type
- `C<:CRS` - coordinate reference system
- `T<:AbstractTopology` - topology type for cloud-level connectivity
"""
struct PointCloud{M <: Manifold, C <: CRS, T <: AbstractTopology} <: Domain{M, C}
    boundary::PointBoundary{M, C}
    volume::PointVolume{M, C}
    topology::T
end

function PointCloud(boundary::PointBoundary{M, C}, volume::PointVolume{M, C}) where {M, C}
    return PointCloud(boundary, volume, NoTopology())
end

function PointCloud(boundary::PointBoundary{M, C}) where {M, C}
    vol = PointVolume{M, C}()
    return PointCloud(deepcopy(boundary), vol, NoTopology())
end

PointCloud(filepath::String) = PointCloud(PointBoundary(filepath))

Base.length(cloud::PointCloud) = length(boundary(cloud)) + length(volume(cloud))
Base.size(cloud::PointCloud) = (length(cloud),)
Base.getindex(cloud::PointCloud, name::Symbol) = boundary(cloud)[name]
function Base.setindex!(cloud::PointCloud, surf::PointSurface, name::Symbol)
    boundary(cloud)[name] = surf
    rebuild_topology!(cloud)
    return nothing
end
function Base.getindex(cloud::PointCloud, index::Int)
    if index > length(cloud)
        throw(
            BoundsError(
                "attempt to access PointCloud at index [$index], but there are only $(length(cloud)) points.",
            ),
        )
    end
    component, local_idx = global_to_local(cloud, index)
    if component === :volume
        return volume(cloud)[local_idx]
    else
        return point(cloud[component])[local_idx]
    end
end
function Base.iterate(cloud::PointCloud, state = 1)
    return state > length(cloud) ? nothing : (cloud[state], state + 1)
end
Base.names(cloud::PointCloud) = names(boundary(cloud))

to(cloud::PointCloud) = to.(points(cloud))
function to(namedsurfaces::LittleDict{Symbol, <:PointSurface})
    return mapreduce(to, vcat, values(namedsurfaces))
end
boundary(cloud::PointCloud) = cloud.boundary
volume(cloud::PointCloud) = cloud.volume
namedsurfaces(cloud::PointCloud) = namedsurfaces(boundary(cloud))
surfaces(cloud::PointCloud) = surfaces(boundary(cloud))
normal(cloud::PointCloud) = mapreduce(normal, vcat, surfaces(cloud))
area(cloud::PointCloud) = mapreduce(area, vcat, surfaces(cloud))

hassurface(cloud::PointCloud, name) = hassurface(boundary(cloud), name)

# Index-space conversion utilities

"""
    local_to_global(cloud::PointCloud, name::Symbol, local_idx::Int) -> Int

Convert a surface-local index to a cloud-global index.
"""
function local_to_global(cloud::PointCloud, name::Symbol, local_idx::Int)
    return local_to_global(boundary(cloud), name, local_idx)
end

"""
    volume_to_global(cloud::PointCloud, local_idx::Int) -> Int

Convert a volume-local index to a cloud-global index.
"""
volume_to_global(cloud::PointCloud, local_idx::Int) = length(boundary(cloud)) + local_idx

"""
    global_to_local(cloud::PointCloud, global_idx::Int) -> (Symbol, Int)

Convert a cloud-global index to a `(component, local_index)` tuple.
Returns `(:volume, local_idx)` for volume indices, or
`(surface_name, local_idx)` for boundary indices.
"""
function global_to_local(cloud::PointCloud, global_idx::Int)
    b = boundary(cloud)
    if global_idx <= length(b)
        return global_to_local(b, global_idx)
    else
        return (:volume, global_idx - length(b))
    end
end

# Topology accessors
"""
    topology(cloud::PointCloud)

Return the topology of the point cloud.
"""
topology(cloud::PointCloud) = cloud.topology

"""
    hastopology(cloud::PointCloud)

Check if point cloud has a topology (not NoTopology).
"""
hastopology(cloud::PointCloud) = !isa(topology(cloud), NoTopology)

"""
    neighbors(cloud::PointCloud)

Return all neighbor lists from the topology. Throws error if no topology or invalid.
"""
neighbors(cloud::PointCloud) = neighbors(topology(cloud))

"""
    neighbors(cloud::PointCloud, i::Int)

Return neighbors of point `i`. Throws error if no topology or invalid.
"""
neighbors(cloud::PointCloud, i::Int) = neighbors(topology(cloud), i)

# Topology operations (functional style - return new clouds)
"""
    set_topology(cloud::PointCloud, ::Type{KNNTopology}, k::Int)

Build and return new cloud with k-nearest neighbor topology.
"""
function set_topology(cloud::PointCloud, ::Type{KNNTopology}, k::Int)
    pts = points(cloud)
    adj = _build_knn_neighbors(pts, k)
    topo = KNNTopology(adj, k)
    return PointCloud(boundary(cloud), volume(cloud), topo)
end

"""
    set_topology(cloud::PointCloud, ::Type{RadiusTopology}, radius)

Build and return new cloud with radius-based topology.
"""
function set_topology(cloud::PointCloud, ::Type{RadiusTopology}, radius)
    pts = points(cloud)
    adj = _build_radius_neighbors(pts, radius)
    topo = RadiusTopology(adj, radius)
    return PointCloud(boundary(cloud), volume(cloud), topo)
end

"""
    rebuild_topology!(cloud::PointCloud)

Rebuild topology in place using same parameters. No-op if NoTopology.
"""
function rebuild_topology!(cloud::PointCloud)
    pts = points(cloud)
    rebuild_topology!(topology(cloud), pts)
    return nothing
end

"""
    points(cloud::PointCloud)

Return vector of all points (boundary + volume).
"""
function points(cloud::PointCloud)
    return vcat(points(boundary(cloud)), points(volume(cloud)))
end
function Meshes.nelements(cloud::PointCloud)
    return Meshes.nelements(boundary(cloud)) + Meshes.nelements(volume(cloud))
end
function Meshes.boundingbox(cloud::PointCloud)
    return boundingbox(points(cloud))
end

function generate_shadows(cloud::PointCloud, shadow::ShadowPoints)
    return mapreduce(s -> generate_shadows(s, shadow), vcat, surfaces(cloud))
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", cloud::PointCloud)
    M, C = manifold(cloud), crs(cloud)
    println(io, "PointCloud{$M, $C}")
    println(io, "├─$(length(cloud)) points")
    has_vol = !iszero(length(cloud.volume))
    if !isnothing(namedsurfaces(cloud))
        println(io, "├─Boundary: $(length(boundary(cloud))) points")
        N = length(namedsurfaces(cloud))
        vert = "│ "
        for (i, name) in enumerate(names(boundary(cloud)))
            char = i < N ? "├" : "└"
            println(io, vert * char * "─$(name)")
        end
    end
    if has_vol
        println(io, "├─Volume: $(length(volume(cloud))) points")
    end
    topo = topology(cloud)
    topo_name = nameof(typeof(topo))
    return println(io, "└─Topology: $topo_name")
end

Base.show(io::IO, ::PointCloud) = println(io, "PointCloud")
