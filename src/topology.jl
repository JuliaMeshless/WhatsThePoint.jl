"""
    abstract type AbstractTopology{S}

Abstract base type for point cloud topology (connectivity).
Type parameter `S` is the storage format for neighbor indices.
"""
abstract type AbstractTopology{S} end

"""
    struct NoTopology <: AbstractTopology{Nothing}

Singleton type representing no topology. Default for PointCloud.
"""
struct NoTopology <: AbstractTopology{Nothing} end

"""
    mutable struct KNNTopology{S} <: AbstractTopology{S}

k-nearest neighbors topology.

# Fields
- `neighbors::S` - neighbor indices storage
- `k::Int` - number of neighbors per point
"""
mutable struct KNNTopology{S} <: AbstractTopology{S}
    neighbors::S
    k::Int
end

"""
    mutable struct RadiusTopology{S,R} <: AbstractTopology{S}

Radius-based topology where neighbors are all points within a given radius.

# Fields
- `neighbors::S` - neighbor indices storage
- `radius::R` - search radius (scalar or function of position)
"""
mutable struct RadiusTopology{S, R} <: AbstractTopology{S}
    neighbors::S
    radius::R
end

# Adjacency list type alias
const AdjacencyList = Vector{Vector{Int}}

"""
    neighbors(t::AbstractTopology)

Return the neighbor storage from a topology.
"""
neighbors(t::AbstractTopology) = t.neighbors

"""
    neighbors(t::AbstractTopology, i::Int)

Return neighbors of point `i`.
"""
neighbors(t::AbstractTopology, i::Int) = t.neighbors[i]

neighbors(::NoTopology) = throw(ArgumentError("NoTopology has no neighbors"))
neighbors(::NoTopology, ::Int) = throw(ArgumentError("NoTopology has no neighbors"))

"""
    isvalid(t::AbstractTopology)

Check if topology is valid. With immutable design, topology is always valid if it exists.
"""
Base.isvalid(::Union{KNNTopology, RadiusTopology}) = true
Base.isvalid(::NoTopology) = true

# Build functions for creating topology from points

"""
    _build_knn_neighbors(points, k::Int) -> Vector{Vector{Int}}

Build k-nearest neighbor adjacency list from points.
"""
function _build_knn_neighbors(points, k::Int)
    method = KNearestSearch(points, k + 1)  # +1 because point is its own nearest neighbor
    all_neighbors = search.(points, Ref(method))
    # Remove self from neighbors (first element)
    return [n[2:end] for n in all_neighbors]
end

"""
    _build_radius_neighbors(points, radius) -> Vector{Vector{Int}}

Build radius-based adjacency list from points.
"""
function _build_radius_neighbors(points, radius)
    r = _get_radius(radius, points)
    method = BallSearch(points, MetricBall(r))
    all_neighbors = search.(points, Ref(method))
    # Remove self from neighbors
    return [filter(!=(i), n) for (i, n) in enumerate(all_neighbors)]
end

_get_radius(r::Number, ::Any) = r
_get_radius(f::Function, points) = f(points)

# In-place rebuild functions

"""
    rebuild_topology!(topo::NoTopology, points)

No-op for NoTopology (nothing to rebuild).
"""
rebuild_topology!(::NoTopology, points) = nothing

"""
    rebuild_topology!(topo::KNNTopology, points)

Rebuild k-nearest neighbor topology in place.
"""
function rebuild_topology!(topo::KNNTopology, points)
    topo.neighbors = _build_knn_neighbors(points, topo.k)
    return nothing
end

"""
    rebuild_topology!(topo::RadiusTopology, points)

Rebuild radius-based topology in place.
"""
function rebuild_topology!(topo::RadiusTopology, points)
    topo.neighbors = _build_radius_neighbors(points, topo.radius)
    return nothing
end

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", t::KNNTopology)
    n = length(t.neighbors)
    println(io, "KNNTopology")
    println(io, "├─k: $(t.k)")
    return println(io, "└─points: $n")
end

function Base.show(io::IO, ::MIME"text/plain", t::RadiusTopology)
    n = length(t.neighbors)
    println(io, "RadiusTopology")
    println(io, "├─radius: $(t.radius)")
    return println(io, "└─points: $n")
end

Base.show(io::IO, ::NoTopology) = print(io, "NoTopology()")
Base.show(io::IO, t::KNNTopology) = print(io, "KNNTopology(k=$(t.k))")
Base.show(io::IO, t::RadiusTopology) = print(io, "RadiusTopology(r=$(t.radius))")
