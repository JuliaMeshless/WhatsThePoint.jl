"""
    struct PointVolume{M,C,T,V} <: Domain{M,C}

Interior volume points with optional topology.

# Type Parameters
- `M<:Manifold` - manifold type
- `C<:CRS` - coordinate reference system
- `T<:AbstractTopology` - topology type for volume-local connectivity
- `V<:AbstractVector{Point{M,C}}` - storage type (allows GPU arrays)
"""
struct PointVolume{M<:Manifold,C<:CRS,T<:AbstractTopology,V<:AbstractVector{Point{M,C}}} <: Domain{M,C}
    points::V
    topology::T
end

function PointVolume{M,C}(;
    topology::T=NoTopology()
) where {M<:Manifold,C<:CRS,T<:AbstractTopology}
    return PointVolume(Point{M,C}[], topology)
end

function PointVolume(pts::AbstractVector{<:Point}; topology=NoTopology())
    return PointVolume(pts, topology)
end

Base.length(vol::PointVolume) = length(vol.points)
Base.size(vol::PointVolume) = (length(vol),)
Base.getindex(vol::PointVolume, index::Int) = vol.points[index]
Base.getindex(vol::PointVolume, index::AbstractVector) = vol.points[index]
function Base.iterate(vol::PointVolume, state=1)
    return state > length(vol) ? nothing : (vol[state], state + 1)
end
Base.isempty(vol::PointVolume) = isempty(vol.points)
Base.parent(vol::PointVolume) = vol.points

"""
    filter(f::Function, vol::PointVolume)

Return new PointVolume with only points satisfying predicate `f`.
Topology is stripped since point indices change.
"""
function Base.filter(f::Function, vol::PointVolume)
    pts = collect(vol.points)
    filtered = filter(f, pts)
    return PointVolume(filtered)
end

Meshes.nelements(vol::PointVolume) = length(vol.points)

to(vol::PointVolume) = to.(vol.points)
centroid(vol::PointVolume) = centroid(vol.points)
boundingbox(vol::PointVolume) = boundingbox(vol.points)

"""
    points(vol::PointVolume)

Return vector of points from volume.
"""
points(vol::PointVolume) = vol.points

# Topology accessors
"""
    topology(vol::PointVolume)

Return the topology of the volume.
"""
topology(vol::PointVolume) = vol.topology

"""
    hastopology(vol::PointVolume)

Check if volume has a topology (not NoTopology).
"""
hastopology(vol::PointVolume) = !isa(topology(vol), NoTopology)

"""
    neighbors(vol::PointVolume)

Return all neighbor lists from the volume topology. Throws error if no topology.
"""
neighbors(vol::PointVolume) = neighbors(topology(vol))

"""
    neighbors(vol::PointVolume, i::Int)

Return neighbors of point `i` in volume-local indices. Throws error if no topology.
"""
neighbors(vol::PointVolume, i::Int) = neighbors(topology(vol), i)

"""
    set_topology(vol::PointVolume, ::Type{KNNTopology}, k::Int)

Build and return new volume with k-nearest neighbor topology.
"""
function set_topology(vol::PointVolume, ::Type{KNNTopology}, k::Int)
    pts = points(vol)
    adj = _build_knn_neighbors(pts, k)
    topo = KNNTopology(adj, k)
    return PointVolume(vol.points; topology=topo)
end

"""
    set_topology(vol::PointVolume, ::Type{RadiusTopology}, radius)

Build and return new volume with radius-based topology.
"""
function set_topology(vol::PointVolume, ::Type{RadiusTopology}, radius)
    pts = points(vol)
    adj = _build_radius_neighbors(pts, radius)
    topo = RadiusTopology(adj, radius)
    return PointVolume(vol.points; topology=topo)
end

"""
    rebuild_topology!(vol::PointVolume)

Rebuild topology in place using same parameters. No-op if NoTopology.
"""
function rebuild_topology!(vol::PointVolume)
    pts = points(vol)
    rebuild_topology!(topology(vol), pts)
    return nothing
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", vol::PointVolume{M,C}) where {M,C}
    println(io, "PointVolume{$M,$C}")
    println(io, "├─Number of points: $(length(vol.points))")
    topo = topology(vol)
    topo_name = nameof(typeof(topo))
    println(io, "└─Topology: $topo_name")
    return nothing
end

Base.show(io::IO, ::PointVolume) = println(io, "PointVolume")
