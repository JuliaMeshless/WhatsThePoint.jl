# Delaunay tetrahedralization for centerline extraction
# Wraps MiniQhull.jl for 3D Delaunay computation

"""
    DelaunayResult{T}

Result of Delaunay tetrahedralization containing tetrahedra and their properties.

# Fields
- `points::Matrix{T}` - Original points (3 × n matrix)
- `simplices::Matrix{Int}` - Tetrahedra vertex indices (4 × m matrix, 1-indexed)
"""
struct DelaunayResult{T<:AbstractFloat}
    points::Matrix{T}
    simplices::Matrix{Int}
end

"""
    tetrahedralize_points(points::AbstractVector{<:Point{𝔼{3}}}) -> DelaunayResult

Compute the Delaunay tetrahedralization of a set of 3D points.

Uses MiniQhull.jl (MIT license, wraps BSD-licensed Qhull) for O(n log n) computation.

# Arguments
- `points` - Vector of 3D points (must be `𝔼{3}` Euclidean)

# Returns
- `DelaunayResult` containing tetrahedra vertex indices

# Example
```julia
points = [Point(rand(3)...) for _ in 1:100]
result = tetrahedralize_points(points)
println("Found \$(size(result.simplices, 2)) tetrahedra")
```
"""
function tetrahedralize_points(points::AbstractVector{<:Point{𝔼{3}}})
    n = length(points)

    if n < 4
        error("Need at least 4 points for tetrahedralization, got $n")
    end

    # Convert to matrix format (3 × n) for MiniQhull
    pts_matrix = Matrix{Float64}(undef, 3, n)
    for (i, p) in enumerate(points)
        coords = to(p)
        pts_matrix[1, i] = ustrip(coords[1])
        pts_matrix[2, i] = ustrip(coords[2])
        pts_matrix[3, i] = ustrip(coords[3])
    end

    # Compute Delaunay tetrahedralization using MiniQhull
    # MiniQhull.delaunay returns a (dim+1, nsimplices) matrix with 1-indexed vertex indices
    simplices = qhull_delaunay(pts_matrix)

    return DelaunayResult(pts_matrix, simplices)
end

"""
    tetrahedralize_points(pts_matrix::Matrix{<:AbstractFloat}) -> DelaunayResult

Compute the Delaunay tetrahedralization from a 3×n matrix of coordinates.
"""
function tetrahedralize_points(pts_matrix::Matrix{T}) where {T<:AbstractFloat}
    if size(pts_matrix, 1) != 3
        error("Expected 3×n matrix, got $(size(pts_matrix, 1))×$(size(pts_matrix, 2))")
    end

    n = size(pts_matrix, 2)
    if n < 4
        error("Need at least 4 points for tetrahedralization, got $n")
    end

    # Convert to Float64 if needed
    pts_f64 = convert(Matrix{Float64}, pts_matrix)

    # Compute Delaunay tetrahedralization
    simplices = qhull_delaunay(pts_f64)

    return DelaunayResult(pts_f64, simplices)
end

"""
    num_tetrahedra(result::DelaunayResult) -> Int

Return the number of tetrahedra in the Delaunay result.
"""
num_tetrahedra(result::DelaunayResult) = size(result.simplices, 2)

"""
    num_points(result::DelaunayResult) -> Int

Return the number of points in the Delaunay result.
"""
num_points(result::DelaunayResult) = size(result.points, 2)

"""
    get_tetrahedron_vertices(result::DelaunayResult, tet_idx::Int) -> Matrix{Float64}

Get the 4 vertices of tetrahedron `tet_idx` as a 3×4 matrix.
"""
function get_tetrahedron_vertices(result::DelaunayResult, tet_idx::Int)
    indices = result.simplices[:, tet_idx]
    return result.points[:, indices]
end

"""
    compute_circumsphere(vertices::AbstractMatrix{<:Real}) -> (center, radius)

Compute the circumsphere of a tetrahedron given its 4 vertices.

# Arguments
- `vertices` - 3×4 matrix where each column is a vertex

# Returns
- `center::SVector{3,Float64}` - Circumcenter position
- `radius::Float64` - Circumradius

The circumsphere is the unique sphere passing through all 4 vertices.
Its center is equidistant from all vertices.
"""
function compute_circumsphere(vertices::AbstractMatrix{T}) where {T<:Real}
    # Extract vertices
    a = SVector{3,Float64}(vertices[:, 1])
    b = SVector{3,Float64}(vertices[:, 2])
    c = SVector{3,Float64}(vertices[:, 3])
    d = SVector{3,Float64}(vertices[:, 4])

    # Translate so that 'a' is at origin
    ba = b - a
    ca = c - a
    da = d - a

    # Compute squared lengths
    ba_sq = dot(ba, ba)
    ca_sq = dot(ca, ca)
    da_sq = dot(da, da)

    # Compute the determinant of the matrix formed by ba, ca, da
    # This is 2 * volume of tetrahedron * 6
    det_m = dot(ba, cross(ca, da))

    if abs(det_m) < 1e-14
        # Degenerate tetrahedron (coplanar points)
        # Return center at centroid with zero radius as fallback
        centroid = (a + b + c + d) / 4
        return centroid, 0.0
    end

    # Circumcenter relative to 'a'
    # Using the formula: center = (1 / (2*det)) * [ba_sq*(ca×da) + ca_sq*(da×ba) + da_sq*(ba×ca)]
    cross_ca_da = cross(ca, da)
    cross_da_ba = cross(da, ba)
    cross_ba_ca = cross(ba, ca)

    center_rel = (ba_sq * cross_ca_da + ca_sq * cross_da_ba + da_sq * cross_ba_ca) / (2 * det_m)

    # Translate back
    center = a + center_rel

    # Circumradius is distance from center to any vertex
    radius = norm(center - a)

    return center, radius
end

"""
    compute_all_circumspheres(result::DelaunayResult) -> (centers, radii)

Compute circumspheres for all tetrahedra in the Delaunay result.

# Returns
- `centers::Vector{SVector{3,Float64}}` - Circumcenter of each tetrahedron
- `radii::Vector{Float64}` - Circumradius of each tetrahedron
"""
function compute_all_circumspheres(result::DelaunayResult)
    n_tets = num_tetrahedra(result)

    centers = Vector{SVector{3,Float64}}(undef, n_tets)
    radii = Vector{Float64}(undef, n_tets)

    for i in 1:n_tets
        verts = get_tetrahedron_vertices(result, i)
        centers[i], radii[i] = compute_circumsphere(verts)
    end

    return centers, radii
end
