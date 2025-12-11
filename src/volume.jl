"""
    struct PointVolume{M,C,T} <: Domain{M,C}

Interior volume points with optional topology.

# Type Parameters
- `M<:Manifold` - manifold type
- `C<:CRS` - coordinate reference system
- `T<:AbstractTopology` - topology type for volume-local connectivity
"""
struct PointVolume{M<:Manifold,C<:CRS,T<:AbstractTopology} <: Domain{M,C}
    points::Domain{M,C}
    topology::T
end

function PointVolume{M,C}(; topology::T=NoTopology()) where {M<:Manifold,C<:CRS,T<:AbstractTopology}
    return PointVolume(PointSet(Point{M,C}[]), topology)
end

function PointVolume(points::AbstractVector; topology=NoTopology())
    return PointVolume(PointSet(points), topology)
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
centroid(vol::PointVolume) = centroid(PointSet(vol.points))
boundingbox(vol::PointVolume) = boundingbox(vol.points)

Meshes.pointify(vol::PointVolume) = Meshes.pointify(vol.points)

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
    points = pointify(vol)
    adj = _build_knn_neighbors(points, k)
    topo = KNNTopology(adj, k)
    return PointVolume(collect(vol.points); topology=topo)
end

"""
    set_topology(vol::PointVolume, ::Type{RadiusTopology}, radius)

Build and return new volume with radius-based topology.
"""
function set_topology(vol::PointVolume, ::Type{RadiusTopology}, radius)
    points = pointify(vol)
    adj = _build_radius_neighbors(points, radius)
    topo = RadiusTopology(adj, radius)
    return PointVolume(collect(vol.points); topology=topo)
end

"""
    rebuild_topology(vol::PointVolume)

Rebuild topology using same parameters. Returns new volume.
"""
function rebuild_topology(vol::PointVolume)
    topo = topology(vol)
    topo isa NoTopology && throw(ArgumentError("Cannot rebuild NoTopology"))
    points = pointify(vol)
    if topo isa KNNTopology
        new_adj = _build_knn_neighbors(points, topo.k)
        new_topo = KNNTopology(new_adj, topo.k)
    elseif topo isa RadiusTopology
        new_adj = _build_radius_neighbors(points, topo.radius)
        new_topo = RadiusTopology(new_adj, topo.radius)
    end
    return PointVolume(collect(vol.points); topology=new_topo)
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", vol::PointVolume{M,C}) where {M,C}
    println(io, "PointVolume{$M,$C}")
    has_topo = hastopology(vol)
    points_char = has_topo ? "├" : "└"
    println(io, "$points_char─Number of points: $(length(vol.points))")
    if has_topo
        topo = topology(vol)
        topo_name = nameof(typeof(topo))
        println(io, "└─Topology: $topo_name")
    end
    return nothing
end

Base.show(io::IO, ::PointVolume) = println(io, "PointVolume")
