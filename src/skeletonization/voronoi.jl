# Embedded Voronoi diagram extraction for centerline computation
# The Voronoi diagram is the dual of the Delaunay tetrahedralization

"""
    VoronoiVertex{T}

A vertex in the Voronoi diagram, corresponding to the circumcenter of a Delaunay tetrahedron.

# Fields
- `position::SVector{3,T}` - Position in 3D space (circumcenter)
- `radius::T` - Maximal inscribed sphere radius (circumradius)
- `tet_index::Int` - Index of source tetrahedron in Delaunay result
"""
struct VoronoiVertex{T<:AbstractFloat}
    position::SVector{3,T}
    radius::T
    tet_index::Int
end

"""
    EmbeddedVoronoi{T}

The embedded Voronoi diagram - the portion of the Voronoi diagram internal to the surface.

# Fields
- `vertices::Vector{VoronoiVertex{T}}` - Voronoi vertices (internal circumcenters)
- `adjacency::Vector{Vector{Int}}` - Adjacency list: neighbors[i] = indices of vertices adjacent to vertex i
- `original_to_embedded::Dict{Int,Int}` - Maps original tetrahedron index to embedded vertex index
- `embedded_to_original::Vector{Int}` - Maps embedded vertex index to original tetrahedron index
"""
struct EmbeddedVoronoi{T<:AbstractFloat}
    vertices::Vector{VoronoiVertex{T}}
    adjacency::Vector{Vector{Int}}
    original_to_embedded::Dict{Int,Int}
    embedded_to_original::Vector{Int}
end

# Accessors
Base.length(ev::EmbeddedVoronoi) = length(ev.vertices)
vertices(ev::EmbeddedVoronoi) = ev.vertices

"""
    neighbors(ev::EmbeddedVoronoi, i::Int) -> Vector{Int}

Return indices of Voronoi vertices adjacent to vertex `i`.
"""
neighbors(ev::EmbeddedVoronoi, i::Int) = ev.adjacency[i]

"""
    extract_embedded_voronoi(delaunay::DelaunayResult, boundary::PointBoundary) -> EmbeddedVoronoi

Extract the embedded Voronoi diagram from a Delaunay tetrahedralization.

The embedded Voronoi consists of only those Voronoi vertices (circumcenters) that
lie INSIDE the surface boundary.

# Arguments
- `delaunay` - Result from `tetrahedralize_points`
- `boundary` - PointBoundary for inside/outside testing

# Returns
- `EmbeddedVoronoi` containing internal Voronoi vertices and their connectivity
"""
function extract_embedded_voronoi(delaunay::DelaunayResult, boundary::PointBoundary)
    # Compute all circumspheres
    centers, radii = compute_all_circumspheres(delaunay)

    # Filter to internal vertices
    internal_indices = Int[]
    for (i, center) in enumerate(centers)
        # Skip degenerate tetrahedra (radius ≈ 0)
        if radii[i] < 1e-12
            continue
        end

        # Create a Point for inside testing
        p = Point(center[1], center[2], center[3])

        if isinside(p, boundary)
            push!(internal_indices, i)
        end
    end

    if isempty(internal_indices)
        @warn "No internal Voronoi vertices found - surface may be too sparse or poorly oriented"
        return EmbeddedVoronoi(
            VoronoiVertex{Float64}[],
            Vector{Int}[],
            Dict{Int,Int}(),
            Int[]
        )
    end

    # Build mapping between original and embedded indices
    original_to_embedded = Dict{Int,Int}()
    embedded_to_original = Int[]

    for (embedded_idx, orig_idx) in enumerate(internal_indices)
        original_to_embedded[orig_idx] = embedded_idx
        push!(embedded_to_original, orig_idx)
    end

    # Create Voronoi vertices
    voronoi_vertices = [
        VoronoiVertex(centers[i], radii[i], i)
        for i in internal_indices
    ]

    # Build adjacency from shared faces
    # Two tetrahedra share a face if they have 3 vertices in common
    # Their Voronoi vertices are then adjacent
    tet_neighbors = _compute_tet_neighbors(delaunay)

    adjacency = [Int[] for _ in 1:length(voronoi_vertices)]

    for (embedded_i, orig_i) in enumerate(internal_indices)
        for neighbor_tet in tet_neighbors[orig_i]
            # Check if neighbor is also internal
            if haskey(original_to_embedded, neighbor_tet)
                embedded_j = original_to_embedded[neighbor_tet]
                if embedded_j != embedded_i && !(embedded_j in adjacency[embedded_i])
                    push!(adjacency[embedded_i], embedded_j)
                end
            end
        end
    end

    return EmbeddedVoronoi(
        voronoi_vertices,
        adjacency,
        original_to_embedded,
        embedded_to_original
    )
end

"""
    _compute_tet_neighbors(delaunay::DelaunayResult) -> Vector{Vector{Int}}

Compute neighbor relationships between tetrahedra.
Two tetrahedra are neighbors if they share exactly 3 vertices (a face).
"""
function _compute_tet_neighbors(delaunay::DelaunayResult)
    n_tets = num_tetrahedra(delaunay)

    # Build face -> tetrahedra mapping
    # Each face is represented by sorted vertex indices
    face_to_tets = Dict{NTuple{3,Int}, Vector{Int}}()

    for tet_idx in 1:n_tets
        verts = delaunay.simplices[:, tet_idx]

        # Each tetrahedron has 4 faces (each omitting one vertex)
        for skip in 1:4
            face_verts = sort([verts[i] for i in 1:4 if i != skip])
            face_key = (face_verts[1], face_verts[2], face_verts[3])

            if !haskey(face_to_tets, face_key)
                face_to_tets[face_key] = Int[]
            end
            push!(face_to_tets[face_key], tet_idx)
        end
    end

    # Build neighbor list for each tetrahedron
    neighbors = [Int[] for _ in 1:n_tets]

    for (_, tets) in face_to_tets
        if length(tets) == 2
            # Two tetrahedra share this face
            t1, t2 = tets[1], tets[2]
            push!(neighbors[t1], t2)
            push!(neighbors[t2], t1)
        end
        # If length == 1, it's a boundary face (no neighbor)
    end

    return neighbors
end

"""
    extract_embedded_voronoi(delaunay::DelaunayResult, surf::PointSurface{𝔼{3}}) -> EmbeddedVoronoi

Extract embedded Voronoi diagram using a PointSurface for inside testing.
"""
function extract_embedded_voronoi(delaunay::DelaunayResult, surf::PointSurface{𝔼{3}})
    boundary = PointBoundary(surf)
    return extract_embedded_voronoi(delaunay, boundary)
end

"""
    extract_embedded_voronoi(delaunay::DelaunayResult, mesh::Meshes.Mesh) -> EmbeddedVoronoi

Extract embedded Voronoi diagram using a Meshes.jl mesh for inside testing.

Uses the mesh face normals to determine inside/outside via solid angle / winding number.
"""
function extract_embedded_voronoi(delaunay::DelaunayResult, mesh::Meshes.Mesh)
    # Compute all circumspheres
    centers, radii = compute_all_circumspheres(delaunay)

    # Filter to internal vertices using mesh
    internal_indices = Int[]
    for (i, center) in enumerate(centers)
        # Skip degenerate tetrahedra
        if radii[i] < 1e-12
            continue
        end

        # Create a Point for inside testing
        p = Point(center[1], center[2], center[3])

        # Use solid angle test for mesh
        if _is_inside_mesh(p, mesh)
            push!(internal_indices, i)
        end
    end

    if isempty(internal_indices)
        @warn "No internal Voronoi vertices found - surface may be too sparse or poorly oriented"
        return EmbeddedVoronoi(
            VoronoiVertex{Float64}[],
            Vector{Int}[],
            Dict{Int,Int}(),
            Int[]
        )
    end

    # Build mapping
    original_to_embedded = Dict{Int,Int}()
    embedded_to_original = Int[]

    for (embedded_idx, orig_idx) in enumerate(internal_indices)
        original_to_embedded[orig_idx] = embedded_idx
        push!(embedded_to_original, orig_idx)
    end

    # Create Voronoi vertices
    voronoi_vertices = [
        VoronoiVertex(centers[i], radii[i], i)
        for i in internal_indices
    ]

    # Build adjacency from shared faces
    tet_neighbors = _compute_tet_neighbors(delaunay)

    adjacency = [Int[] for _ in 1:length(voronoi_vertices)]

    for (embedded_i, orig_i) in enumerate(internal_indices)
        for neighbor_tet in tet_neighbors[orig_i]
            if haskey(original_to_embedded, neighbor_tet)
                embedded_j = original_to_embedded[neighbor_tet]
                if embedded_j != embedded_i && !(embedded_j in adjacency[embedded_i])
                    push!(adjacency[embedded_i], embedded_j)
                end
            end
        end
    end

    return EmbeddedVoronoi(
        voronoi_vertices,
        adjacency,
        original_to_embedded,
        embedded_to_original
    )
end

"""
    _is_inside_mesh(point::Point, mesh::Meshes.Mesh) -> Bool

Test if a point is inside a closed mesh using ray casting.
"""
function _is_inside_mesh(point::Point{𝔼{3}}, mesh::Meshes.Mesh)
    # Use Meshes.jl's built-in point-in-polyhedron test if available
    # Otherwise fall back to a simple approach

    # Try using Meshes.jl's `in` operator
    try
        return point in mesh
    catch
        # Fallback: use solid angle / winding number approach
        # For a point inside a closed surface, the solid angle is 4π
        # For a point outside, it's 0

        coords = to(point)
        px, py, pz = ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3])
        p = SVector{3,Float64}(px, py, pz)

        solid_angle = 0.0

        for elem in elements(mesh)
            # Get triangle vertices
            verts = Meshes.vertices(elem)
            if length(verts) != 3
                continue  # Skip non-triangular faces
            end

            v1 = to(verts[1])
            v2 = to(verts[2])
            v3 = to(verts[3])

            a = SVector{3,Float64}(ustrip(v1[1]), ustrip(v1[2]), ustrip(v1[3])) - p
            b = SVector{3,Float64}(ustrip(v2[1]), ustrip(v2[2]), ustrip(v2[3])) - p
            c = SVector{3,Float64}(ustrip(v3[1]), ustrip(v3[2]), ustrip(v3[3])) - p

            # Compute solid angle contribution using the formula:
            # Ω = 2*atan(a·(b×c) / (|a||b||c| + |a|(b·c) + |b|(a·c) + |c|(a·b)))
            na, nb, nc = norm(a), norm(b), norm(c)

            if na < 1e-12 || nb < 1e-12 || nc < 1e-12
                continue  # Point on vertex
            end

            numerator = dot(a, cross(b, c))
            denominator = na * nb * nc + na * dot(b, c) + nb * dot(a, c) + nc * dot(a, b)

            solid_angle += 2 * atan(numerator, denominator)
        end

        # Inside if solid angle is approximately 4π
        return abs(solid_angle) > 2π
    end
end

"""
    edge_length(ev::EmbeddedVoronoi, i::Int, j::Int) -> Float64

Compute the Euclidean distance between Voronoi vertices i and j.
"""
function edge_length(ev::EmbeddedVoronoi, i::Int, j::Int)
    return norm(ev.vertices[i].position - ev.vertices[j].position)
end

"""
    find_nearest_vertex(ev::EmbeddedVoronoi, point) -> Int

Find the index of the Voronoi vertex nearest to the given point.

# Arguments
- `ev` - Embedded Voronoi diagram
- `point` - Target point as Point, tuple (x,y,z), or SVector{3}
"""
function find_nearest_vertex(ev::EmbeddedVoronoi, point::Point{𝔼{3}})
    coords = to(point)
    target = SVector{3,Float64}(ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3]))
    return find_nearest_vertex(ev, target)
end

function find_nearest_vertex(ev::EmbeddedVoronoi, point::Tuple{<:Real,<:Real,<:Real})
    target = SVector{3,Float64}(point...)
    return find_nearest_vertex(ev, target)
end

function find_nearest_vertex(ev::EmbeddedVoronoi, target::SVector{3,<:Real})
    if isempty(ev.vertices)
        error("Cannot find nearest vertex in empty Voronoi diagram")
    end

    min_dist = Inf
    nearest_idx = 1

    for (i, v) in enumerate(ev.vertices)
        dist = norm(v.position - target)
        if dist < min_dist
            min_dist = dist
            nearest_idx = i
        end
    end

    return nearest_idx
end

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", ev::EmbeddedVoronoi)
    n_verts = length(ev.vertices)
    n_edges = sum(length.(ev.adjacency)) ÷ 2  # Each edge counted twice

    println(io, "EmbeddedVoronoi")
    println(io, "  $(n_verts) vertices")
    println(io, "  $(n_edges) edges")

    if n_verts > 0
        radii = [v.radius for v in ev.vertices]
        println(io, "  radius range: $(minimum(radii)) - $(maximum(radii))")
    end
end
