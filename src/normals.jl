"""
    compute_normals(points::PointSurface; k::Int=5)

Estimate the normals of a set of points that form a surface. Uses the PCA approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

"""
function compute_normals(surf::PointSurface; k::Int=5)
    k = k > length(surf) ? length(surf) : k
    points = point(surf)
    return compute_normals(points; k=k)
end

function compute_normals(points::AbstractVector{<:Point}; k::Int=5)
    search_method = KNearestSearch(points, k)
    return compute_normals(search_method, points)
end

# TODO do not include points near edge. use map_edges() function to find edge points.
"""
    compute_normals(search_method::KNearestSearch, points::PointSurface)

Estimate the normals of a set of points that form a surface. Uses the PCA approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

"""
function compute_normals(search_method::KNearestSearch, surf::PointSurface)
    return compute_normals(search_method, point(surf))
end

function compute_normals(search_method::KNearestSearch, points::AbstractVector{<:Point})
    neighbors = search.(points, Ref(search_method))
    normals = tmap(neighbors) do neighborhood
        _compute_normal(points[neighborhood])
    end
    return normals
end

"""
    update_normals!(surf::PointCloud; k::Int=5)

Update the normals of the surfaces of a surf. This is necessary whenever the points change for any reason.

"""
function update_normals!(surf::PointSurface; k::Int=5)
    k = k > length(points) ? length(points) : k
    neighbors = search(surf, KNearestSearch(surf, k))
    Threads.@threads for i in eachindex(normals)
        parent(surf).normal[i] = _compute_normal(parent(surf).point[neighbors[i]])
    end
end

function _compute_normal(points::AbstractVector{<:Point})
    # from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).
    v = ustrip.(to.(points))
    _, Q = eigen(Symmetric(cov(v)))
    return SVector(Q[:, 1])
end

"""
    orient_normals!(search_method::KNearestSearch, normals::AbstractVector{<:AbstractVector}, points)

Correct the orientation of normals on a surface as the [compute_normals](@ref) function does not guarantee if the normal is inward or outward facing. Uses the approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

"""
function orient_normals!(
    search_method::KNearestSearch, normals::AbstractVector{<:AbstractVector}, points
)
    # build minimum spanning tree based on angle between normals
    neighbors = search.(points, Ref(search_method))

    g = build_normal_weighted_graph(normals, neighbors)

    # TODO below is slow
    mst = kruskal_mst(g)
    g_mst = SimpleGraph(nv(g)) #Create a new graph
    for ew in mst
        add_edge!(g_mst, ew.src, ew.dst)
    end

    # check if highest point (in Z direction) faces up, correct if not
    start = argmax(last.(to.(points)))
    if last(normals[start]) < 0
        normals[start] = -normals[start]
    end

    # Depth-first traversal of minimum spanning tree
    parents = dfs_parents(g_mst, start)
    visited = falses(length(parents))
    function visit(ivertex)
        visited[ivertex] = true
        if normals[ivertex] ⋅ normals[parents[ivertex]] < 0
            normals[ivertex] = -normals[ivertex]
        end
        for nb in Graphs.neighbors(g_mst, ivertex)
            !visited[nb] && visit(nb)
        end
    end
    visit(start)

    return nothing
end

"""
    orient_normals!(normals::Vector{<:AbstractVector}, points; k::Int=5)

Correct the orientation of normals on a surface as the [compute_normals](@ref) function does not guarantee if the normal is inward or outward facing. Uses the approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

"""
function orient_normals!(normals::AbstractVector{<:AbstractVector}, points; k::Int=5)
    k = k > length(points) ? length(points) : k
    # build minimum spanning tree based on angle between normals
    search_method = KNearestSearch(points, k)
    return orient_normals!(search_method, normals, points)
end

function orient_normals!(surf::PointSurface; k::Int=5)
    k = k > length(surf) ? length(surf) : k
    return orient_normals!(normal(surf), point(surf); k=k)
end

function orient_normals!(cloud::PointCloud; k::Int=5)
    for surf in cloud
        orient_normals!(surf; k=k)
    end
end

# TODO optimize
function build_normal_weighted_graph(normals::AbstractVector{<:AbstractVector}, neighbors)
    g = SimpleWeightedGraph(length(normals))
    T = eltype(first(normals))
    epsilon = eps(T) * 1e2 # offset because edge weight cannot be 0
    unitless_normals = ustrip.(normals)
    for n in neighbors, v in n[2:end]
        weight = one(T) - abs(unitless_normals[n[1]] ⋅ unitless_normals[v]) + epsilon
        add_edge!(g, n[1], v, weight)
    end
    return g
end

function map_edge(points)
    λ, _ = eigen(Symmetric(cov(to.(points))))
    return λ[1] / sum(λ)
end

function map_edges(points::Domain; k::Int=5)
    k = k > length(points) ? length(points) : k
    # find neighbors of each point
    neighbors = search(points, KNearestSearch(points, k))
    edges = tmap(neighbors) do neighborhood
        map_edge(points[neighborhood])
    end
    return edges
end
