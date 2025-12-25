# Graph extraction from contracted point clouds

"""
    extract_skeleton_graph(contracted::ContractedSurface{M,C}, params::GraphExtractionParams) where {M,C}

Extract a graph structure from contracted points.

The algorithm:
1. Clusters points into voxels based on spatial proximity
2. Computes cluster centers as graph nodes
3. Builds connectivity from original neighborhood relationships
4. Extracts minimum spanning tree for clean topology
5. Optionally prunes short branches (noise)

Returns `SkeletonGraph` with nodes at cluster centers and edges based on proximity.
"""
function extract_skeleton_graph(
    contracted::ContractedSurface{𝔼{N},C}, params::GraphExtractionParams
) where {N,C<:CRS}
    pts = contracted.points

    if isempty(pts)
        return SkeletonGraph(SimpleWeightedGraph(0), SkeletonNode{𝔼{N},C}[])
    end

    # Determine voxel size (auto-compute if not specified)
    voxel_size = _get_voxel_size(params.voxel_size, contracted)

    # Cluster points into voxels
    clusters, point_to_cluster = _voxel_cluster(pts, voxel_size)

    if isempty(clusters)
        return SkeletonGraph(SimpleWeightedGraph(0), SkeletonNode{𝔼{N},C}[])
    end

    # Compute cluster centers and create nodes
    nodes = _create_skeleton_nodes(pts, clusters)

    # Build connectivity graph based on original neighborhoods
    g = _build_skeleton_adjacency(pts, point_to_cluster, params.connectivity_k)

    # Extract minimum spanning tree to get clean tree structure
    if ne(g) > 0
        mst_edges = kruskal_mst(g)
        g_mst = SimpleWeightedGraph(nv(g))
        for e in mst_edges
            add_edge!(g_mst, src(e), dst(e), g.weights[src(e), dst(e)])
        end
    else
        g_mst = g
    end

    # Prune short branches
    _prune_short_branches!(g_mst, nodes, params.min_branch_length)

    return SkeletonGraph(g_mst, nodes)
end

"""
    _get_voxel_size(specified_size, contracted)

Determine voxel size for clustering. If not specified, use median local scale.
"""
function _get_voxel_size(::Nothing, contracted::ContractedSurface)
    # Auto-compute based on local scales (cube root of volumes)
    if isempty(contracted.volumes)
        return 1.0  # Fallback
    end
    scales = contracted.volumes .^ (1 / 3)
    return median(scales) * 2  # Use 2× median scale for reasonable clustering
end

function _get_voxel_size(specified_size::Unitful.Length, ::ContractedSurface)
    return ustrip(specified_size)
end

"""
    _voxel_cluster(points, voxel_size) -> (clusters, point_to_cluster)

Cluster points into voxel grid cells.

Returns:
- `clusters::Vector{Vector{Int}}` - For each cluster, indices of points in that cluster
- `point_to_cluster::Vector{Int}` - For each point, which cluster it belongs to
"""
function _voxel_cluster(
    points::Vector{<:Point{𝔼{N}}}, voxel_size::Float64
) where {N}
    n = length(points)

    # Map each point to its voxel ID
    voxel_to_points = Dict{NTuple{N,Int},Vector{Int}}()

    for i in 1:n
        coords = ustrip.(to(points[i]))
        voxel_id = ntuple(j -> floor(Int, coords[j] / voxel_size), N)

        if !haskey(voxel_to_points, voxel_id)
            voxel_to_points[voxel_id] = Int[]
        end
        push!(voxel_to_points[voxel_id], i)
    end

    # Convert to vector format
    clusters = collect(values(voxel_to_points))

    # Build reverse mapping
    point_to_cluster = zeros(Int, n)
    for (cluster_idx, point_indices) in enumerate(clusters)
        for i in point_indices
            point_to_cluster[i] = cluster_idx
        end
    end

    return clusters, point_to_cluster
end

"""
    _create_skeleton_nodes(points, clusters) -> Vector{SkeletonNode}

Create skeleton nodes at cluster centers.
"""
function _create_skeleton_nodes(
    points::Vector{<:Point{𝔼{N}}}, clusters::Vector{Vector{Int}}
) where {N}
    coord_unit = unit(to(first(points))[1])

    nodes = map(clusters) do cluster_indices
        # Compute centroid
        center_coords = zeros(N)
        for idx in cluster_indices
            center_coords .+= ustrip.(to(points[idx]))
        end
        center_coords ./= length(cluster_indices)

        # Create point with units (let constructor infer type)
        center_point = Point(ntuple(j -> center_coords[j] * coord_unit, N)...)

        SkeletonNode(center_point, cluster_indices)
    end

    return nodes
end

"""
    _build_skeleton_adjacency(points, point_to_cluster, k) -> SimpleWeightedGraph

Build adjacency graph between clusters based on original point neighborhoods.

Two clusters are connected if any of their points were neighbors in the original cloud.
Edge weight is the distance between cluster centers.
"""
function _build_skeleton_adjacency(
    points::Vector{<:Point{𝔼{N}}}, point_to_cluster::Vector{Int}, k::Int
) where {N}
    n_clusters = maximum(point_to_cluster; init=0)

    if n_clusters == 0
        return SimpleWeightedGraph(0)
    end

    # Find neighbors in original point cloud
    method = KNearestSearch(points, min(k + 1, length(points)))
    all_neighbors = search.(points, Ref(method))

    # Compute cluster centers for edge weights
    cluster_centers = [zeros(N) for _ in 1:n_clusters]
    cluster_counts = zeros(Int, n_clusters)

    for (i, p) in enumerate(points)
        c = point_to_cluster[i]
        cluster_centers[c] .+= ustrip.(to(p))
        cluster_counts[c] += 1
    end

    for c in 1:n_clusters
        if cluster_counts[c] > 0
            cluster_centers[c] ./= cluster_counts[c]
        end
    end

    # Build graph
    g = SimpleWeightedGraph(n_clusters)

    for i in 1:length(points)
        ci = point_to_cluster[i]
        for j in all_neighbors[i]
            if j != i
                cj = point_to_cluster[j]
                if ci != cj && !has_edge(g, ci, cj)
                    # Edge weight = distance between cluster centers
                    dist = sqrt(sum((cluster_centers[ci] .- cluster_centers[cj]) .^ 2))
                    add_edge!(g, ci, cj, dist)
                end
            end
        end
    end

    return g
end

"""
    _prune_short_branches!(g, nodes, min_length)

Remove branches (chains ending in degree-1 nodes) shorter than min_length.
Modifies graph in place.
"""
function _prune_short_branches!(
    g::SimpleWeightedGraph, ::Vector{<:SkeletonNode}, min_length::Int
)
    if nv(g) == 0 || min_length <= 0
        return nothing
    end

    changed = true
    while changed
        changed = false
        endpoints = [v for v in 1:nv(g) if Graphs.degree(g, v) == 1]

        for v in endpoints
            if Graphs.degree(g, v) != 1  # May have changed
                continue
            end

            # Trace back along the branch
            branch = [v]
            current = v
            prev = 0

            while Graphs.degree(g, current) <= 2 && length(branch) < min_length
                neighs = Graphs.neighbors(g, current)
                next_node = findfirst(n -> n != prev, neighs)

                if isnothing(next_node)
                    break
                end

                prev = current
                current = neighs[next_node]
                push!(branch, current)

                # Stop if we hit a branch point (degree > 2)
                if Graphs.degree(g, current) > 2
                    break
                end
            end

            # If branch is too short, remove it (except for the branch point)
            if length(branch) < min_length && Graphs.degree(g, branch[end]) > 2
                for node in branch[1:end-1]
                    # Remove all edges from this node
                    for n in collect(Graphs.neighbors(g, node))
                        rem_edge!(g, node, n)
                    end
                end
                changed = true
            end
        end
    end

    return nothing
end
