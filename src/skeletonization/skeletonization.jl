# Skeletonization module
# Laplacian-Based Contraction (LBC) for point cloud skeletonization

# ============================================================================
# Algorithm Types
# ============================================================================

"""
    abstract type AbstractSkeletonizationAlgorithm end

Base type for skeletonization algorithms.
"""
abstract type AbstractSkeletonizationAlgorithm end

"""
    LBCSkeletonization <: AbstractSkeletonizationAlgorithm

Laplacian-Based Contraction skeletonization parameters.

# Fields
- `k::Int` - Number of neighbors for adaptive Laplacian (default: 15)
- `WL_init::Float64` - Initial Laplacian weight (default: 1.0)
- `WL_factor::Float64` - WL increase factor per iteration (default: 2.0)
- `max_iters::Int` - Maximum iterations (default: 50)
- `tol::Float64` - Convergence tolerance (default: 1e-4)
"""
Base.@kwdef struct LBCSkeletonization <: AbstractSkeletonizationAlgorithm
    k::Int = 15
    WL_init::Float64 = 1.0
    WL_factor::Float64 = 2.0
    max_iters::Int = 50
    tol::Float64 = 1e-4
end

"""
    GraphExtractionParams

Parameters for skeleton graph extraction from contracted points.

# Fields
- `voxel_size::Union{Nothing,<:Unitful.Length}` - Voxel size for clustering (auto if nothing)
- `min_branch_length::Int` - Minimum branch length to keep (default: 3 nodes)
- `connectivity_k::Int` - Number of neighbors for connectivity graph (default: 6)
"""
Base.@kwdef struct GraphExtractionParams
    voxel_size::Union{Nothing,Unitful.Length} = nothing
    min_branch_length::Int = 3
    connectivity_k::Int = 6
end

# ============================================================================
# Output Types
# ============================================================================

"""
    ContractedSurface{M<:Manifold, C<:CRS}

Intermediate result of LBC contraction before graph extraction.

# Fields
- `points::Vector{Point{M,C}}` - Contracted point positions
- `original_surface::PointSurface{M,C}` - Reference to source surface
- `volumes::Vector{Float64}` - Local volume estimates at each point
- `convergence::Vector{Float64}` - Convergence history (max displacement per iteration)
"""
struct ContractedSurface{M<:Manifold,C<:CRS}
    points::Vector{Point{M,C}}
    original_surface::PointSurface{M,C}
    volumes::Vector{Float64}
    convergence::Vector{Float64}
end

# Accessors for ContractedSurface
points(cs::ContractedSurface) = cs.points
Base.length(cs::ContractedSurface) = length(cs.points)

"""
    SkeletonNode{M<:Manifold, C<:CRS}

A node in the skeleton graph representing a point on the medial axis.

# Fields
- `point::Point{M,C}` - Position of the skeleton node
- `original_indices::Vector{Int}` - Indices of contracted points that merged here
"""
struct SkeletonNode{M<:Manifold,C<:CRS}
    point::Point{M,C}
    original_indices::Vector{Int}
end

# Accessor
point(sn::SkeletonNode) = sn.point

"""
    SkeletonGraph{M<:Manifold, C<:CRS, G<:AbstractGraph}

Skeleton representation as a graph structure.

# Fields
- `graph::G` - Underlying Graphs.jl graph (SimpleWeightedGraph for edge lengths)
- `nodes::Vector{SkeletonNode{M,C}}` - Node positions and metadata
"""
struct SkeletonGraph{M<:Manifold,C<:CRS,G<:AbstractGraph}
    graph::G
    nodes::Vector{SkeletonNode{M,C}}
end

# Accessors for SkeletonGraph
nodes(sg::SkeletonGraph) = sg.nodes
graph(sg::SkeletonGraph) = sg.graph
Base.length(sg::SkeletonGraph) = length(sg.nodes)
points(sg::SkeletonGraph) = [point(n) for n in sg.nodes]

# Delegate graph operations to Graphs.jl
Graphs.nv(sg::SkeletonGraph) = nv(sg.graph)
Graphs.ne(sg::SkeletonGraph) = ne(sg.graph)
Graphs.edges(sg::SkeletonGraph) = edges(sg.graph)
Graphs.neighbors(sg::SkeletonGraph, i::Int) = Graphs.neighbors(sg.graph, i)

"""
    skeleton_length(sg::SkeletonGraph)

Compute the total skeleton length (sum of all edge weights).
"""
function skeleton_length(sg::SkeletonGraph)
    total = 0.0
    for e in edges(sg.graph)
        total += sg.graph.weights[src(e), dst(e)]
    end
    return total
end

"""
    branch_points(sg::SkeletonGraph) -> Vector{Int}

Return indices of nodes with degree > 2 (branch points).
"""
function branch_points(sg::SkeletonGraph)
    return [i for i in 1:nv(sg) if Graphs.degree(sg.graph, i) > 2]
end

"""
    end_points(sg::SkeletonGraph) -> Vector{Int}

Return indices of nodes with degree == 1 (endpoints/leaf nodes).
"""
function end_points(sg::SkeletonGraph)
    return [i for i in 1:nv(sg) if Graphs.degree(sg.graph, i) == 1]
end

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", cs::ContractedSurface{M,C}) where {M,C}
    println(io, "ContractedSurface{$M, $C}")
    println(io, "  $(length(cs.points)) contracted points")
    println(io, "  $(length(cs.convergence)) iterations")
    if !isempty(cs.convergence)
        println(io, "  final convergence: $(cs.convergence[end])")
    end
end

function Base.show(io::IO, ::MIME"text/plain", sg::SkeletonGraph{M,C}) where {M,C}
    println(io, "SkeletonGraph{$M, $C}")
    println(io, "  $(nv(sg)) nodes, $(ne(sg)) edges")
    println(io, "  $(length(branch_points(sg))) branch points")
    println(io, "  $(length(end_points(sg))) endpoints")
end

# Include sub-modules
include("laplacian.jl")
include("contraction.jl")
include("graph_extraction.jl")

# ============================================================================
# Main API
# ============================================================================

"""
    skeletonize(surf::PointSurface{𝔼{N},C}; alg=LBCSkeletonization(),
                extract_graph=true, graph_params=GraphExtractionParams()) where {N,C}

Compute the skeleton of a point surface using Laplacian-Based Contraction.

Returns `(skeleton, convergence)` where:
- `skeleton` is `SkeletonGraph` if `extract_graph=true`, else `ContractedSurface`
- `convergence` is a vector of max displacement per iteration

Requires Euclidean manifold (`𝔼{2}` or `𝔼{3}`).

# Example
```julia
surf = PointSurface("vessel.stl")
skeleton, conv = skeletonize(surf; alg=LBCSkeletonization(k=20, max_iters=100))
println("Skeleton has \$(nv(skeleton)) nodes")
```
"""
function skeletonize(
    surf::PointSurface{𝔼{N},C};
    alg::AbstractSkeletonizationAlgorithm=LBCSkeletonization(),
    extract_graph::Bool=true,
    graph_params::GraphExtractionParams=GraphExtractionParams(),
) where {N,C<:CRS}
    # Perform contraction
    contracted = contract_lbc(surf, alg)

    if extract_graph
        # Extract graph structure
        skeleton = extract_skeleton_graph(contracted, graph_params)
        return (skeleton, contracted.convergence)
    else
        return (contracted, contracted.convergence)
    end
end
