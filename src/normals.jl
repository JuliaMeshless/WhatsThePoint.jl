"""
    compute_normals(surf::PointSurface{𝔼{N},C}; k::Int=5) where {N,C<:CRS}

Estimate the normals of a set of points that form a surface. Uses the PCA approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

Requires Euclidean manifold (`𝔼{2}` or `𝔼{3}`). This function assumes flat space geometry.

"""
function compute_normals(surf::PointSurface{𝔼{N}, C}; k::Int = 5) where {N, C <: CRS}
    k = k > length(surf) ? length(surf) : k
    points = point(surf)
    return compute_normals(points; k = k)
end

function compute_normals(points::AbstractVector{<:Point{𝔼{N}}}; k::Int = 5) where {N}
    k = k > length(points) ? length(points) : k
    search_method = KNearestSearch(points, k)
    return compute_normals(search_method, points)
end

# TODO do not include points near edge.
"""
    compute_normals(search_method::KNearestSearch, surf::PointSurface{𝔼{N},C}) where {N,C<:CRS}

Estimate the normals of a set of points that form a surface. Uses the PCA approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

Requires Euclidean manifold (`𝔼{2}` or `𝔼{3}`). This function assumes flat space geometry.

"""
function compute_normals(search_method::KNearestSearch, surf::PointSurface{𝔼{N}, C}) where {N, C <: CRS}
    return compute_normals(search_method, point(surf))
end

function compute_normals(search_method::KNearestSearch, points::AbstractVector{<:Point{𝔼{N}}}) where {N}
    neighbors = search.(points, Ref(search_method))
    normals = tmap(n -> _compute_normal(points[n]), neighbors)
    return normals
end

"""
    update_normals!(surf::PointSurface{𝔼{N},C}; k::Int=5) where {N,C<:CRS}

Update the normals of the boundary of a surf. This is necessary whenever the points change for any reason.

Requires Euclidean manifold (`𝔼{2}` or `𝔼{3}`). This function assumes flat space geometry.

"""
function update_normals!(surf::PointSurface{𝔼{N}, C}; k::Int = 5) where {N, C <: CRS}
    k = k > length(surf) ? length(surf) : k
    neighbors = search(surf, KNearestSearch(surf, k))
    normals = normal(surf)
    points = point(surf)
    return tmap!(n -> _compute_normal(points[n]), normals, neighbors)
end

function _compute_normal(points::AbstractVector{<:Point{𝔼{N}}}) where {N}
    # from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).
    v = map(x -> ustrip.(x), to.(points))
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
        return
    end
    visit(start)

    return nothing
end

"""
    orient_normals!(normals::AbstractVector{<:AbstractVector}, points::AbstractVector{<:Point{𝔼{N}}}; k::Int=5) where {N}

Correct the orientation of normals on a surface as the [compute_normals](@ref) function does not guarantee if the normal is inward or outward facing. Uses the approach from "Surface Reconstruction from Unorganized Points" - Hoppe (1992).

Requires Euclidean manifold (`𝔼{2}` or `𝔼{3}`). This function uses Euclidean dot products for orientation consistency.

"""
function orient_normals!(normals::AbstractVector{<:AbstractVector}, points::AbstractVector{<:Point{𝔼{N}}}; k::Int = 5) where {N}
    k = k > length(points) ? length(points) : k
    # build minimum spanning tree based on angle between normals
    search_method = KNearestSearch(points, k)
    return orient_normals!(search_method, normals, points)
end

function orient_normals!(surf::PointSurface{𝔼{N}, C}; k::Int = 5) where {N, C <: CRS}
    k = k > length(surf) ? length(surf) : k
    return orient_normals!(normal(surf), point(surf); k = k)
end

function orient_normals!(cloud::PointCloud{𝔼{N}, C}; k::Int = 5) where {N, C <: CRS}
    for surf in surfaces(cloud)
        orient_normals!(surf; k = k)
    end
    return
end

# TODO optimize
function build_normal_weighted_graph(normals::AbstractVector{<:AbstractVector}, neighbors)
    g = SimpleWeightedGraph(length(normals))
    T = eltype(first(normals))
    epsilon = eps(T) * 1.0e2 # offset because edge weight cannot be 0
    unitless_normals = map(n -> ustrip.(n), normals)
    for n in neighbors, v in n[2:end]
        weight = one(T) - abs(unitless_normals[n[1]] ⋅ unitless_normals[v]) + epsilon
        add_edge!(g, n[1], v, weight)
    end
    return g
end
