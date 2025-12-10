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
    struct KNNTopology{S} <: AbstractTopology{S}

k-nearest neighbors topology.

# Fields
- `neighbors::S` - neighbor indices storage
- `k::Int` - number of neighbors per point
- `valid::Base.RefValue{Bool}` - staleness flag
"""
mutable struct KNNTopology{S} <: AbstractTopology{S}
    neighbors::S
    k::Int
    valid::Bool
end

"""
    struct RadiusTopology{S,R} <: AbstractTopology{S}

Radius-based topology where neighbors are all points within a given radius.

# Fields
- `neighbors::S` - neighbor indices storage
- `radius::R` - search radius (scalar or function of position)
- `valid::Base.RefValue{Bool}` - staleness flag
"""
mutable struct RadiusTopology{S,R} <: AbstractTopology{S}
    neighbors::S
    radius::R
    valid::Bool
end

# Adjacency list type alias
const AdjacencyList = Vector{Vector{Int}}

"""
    neighbors(t::AbstractTopology)

Return the neighbor storage from a topology. Throws error if topology is invalid.
"""
function neighbors(t::AbstractTopology)
    isvalid(t) || throw(InvalidTopologyError())
    return t.neighbors
end

"""
    neighbors(t::AbstractTopology, i::Int)

Return neighbors of point `i`. Throws error if topology is invalid.
"""
function neighbors(t::AbstractTopology, i::Int)
    isvalid(t) || throw(InvalidTopologyError())
    return t.neighbors[i]
end

neighbors(::NoTopology) = throw(ArgumentError("NoTopology has no neighbors"))
neighbors(::NoTopology, ::Int) = throw(ArgumentError("NoTopology has no neighbors"))

"""
    isvalid(t::AbstractTopology)

Check if topology is valid (not stale).
"""
Base.isvalid(t::Union{KNNTopology,RadiusTopology}) = t.valid
Base.isvalid(::NoTopology) = true

"""
    invalidate!(t::AbstractTopology)

Mark topology as invalid/stale.
"""
function invalidate!(t::Union{KNNTopology,RadiusTopology})
    t.valid = false
    return t
end
invalidate!(t::NoTopology) = t

"""
    validate!(t::AbstractTopology)

Mark topology as valid.
"""
function validate!(t::Union{KNNTopology,RadiusTopology})
    t.valid = true
    return t
end
validate!(t::NoTopology) = t

"""
    InvalidTopologyError

Error thrown when accessing an invalid (stale) topology.
"""
struct InvalidTopologyError <: Exception end

function Base.showerror(io::IO, ::InvalidTopologyError)
    print(io, "InvalidTopologyError: topology is stale. Call rebuild_topology! to rebuild.")
end

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

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", t::KNNTopology{S}) where {S}
    status = isvalid(t) ? "valid" : "INVALID"
    n = length(t.neighbors)
    println(io, "KNNTopology{$S}")
    println(io, "├─k: $(t.k)")
    println(io, "├─points: $n")
    println(io, "└─status: $status")
end

function Base.show(io::IO, ::MIME"text/plain", t::RadiusTopology{S,R}) where {S,R}
    status = isvalid(t) ? "valid" : "INVALID"
    n = length(t.neighbors)
    println(io, "RadiusTopology{$S,$R}")
    println(io, "├─radius: $(t.radius)")
    println(io, "├─points: $n")
    println(io, "└─status: $status")
end

Base.show(io::IO, ::NoTopology) = print(io, "NoTopology()")
Base.show(io::IO, t::KNNTopology) = print(io, "KNNTopology(k=$(t.k))")
Base.show(io::IO, t::RadiusTopology) = print(io, "RadiusTopology(r=$(t.radius))")
