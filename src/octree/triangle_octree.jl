"""
    TriangleOctree{M<:Manifold, C<:CRS, T<:Real}

An octree-based spatial index for efficient triangle mesh queries.

# Fields
- `tree::SpatialOctree{Int,T}`: Underlying octree storing triangle indices
- `mesh::SimpleMesh{M,C}`: Original Meshes.jl mesh
- `triangles::StructArray`: Cached triangle data (v1, v2, v3, normal as SVectors)
- `leaf_classification::Union{Nothing, Vector{Int8}}`: Classification of empty leaves
  - `0`: Exterior (outside mesh)
  - `1`: Boundary (near surface but empty)
  - `2`: Interior (inside mesh)
- `leaf_nearest_triangle::Union{Nothing, Vector{Int}}`: Index of nearest triangle for each leaf.
  Used by boundary leaves (classification == 1) that have no local triangles to avoid
  expensive global signed distance computation at query time.

# Performance
For a mesh with M triangles:
- Construction: O(M log M) - distribute triangles to boxes
- Query: O(log M + k) where k ≈ 10-50 triangles per leaf
- Memory: O(M) for triangle storage + O(N) for octree nodes

# Example
```julia
# Option 1: From STL file (simplest)
octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)

# Option 2: From SimpleMesh
mesh = GeoIO.load("model.stl").geometry
octree = TriangleOctree(mesh; h_min=0.01, classify_leaves=true)

# Fast queries (100-1000x speedup over brute force)
point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)
```
"""
struct TriangleOctree{M<:Manifold,C<:CRS,T<:Real}
    tree::SpatialOctree{Int,T}
    mesh::SimpleMesh{M,C}
    triangles::StructArray{
        @NamedTuple{
            v1::SVector{3,T},
            v2::SVector{3,T},
            v3::SVector{3,T},
            normal::SVector{3,T},
        },
        1,
        @NamedTuple{
            v1::Vector{SVector{3,T}},
            v2::Vector{SVector{3,T}},
            v3::Vector{SVector{3,T}},
            normal::Vector{SVector{3,T}},
        },
    }
    leaf_classification::Union{Nothing,Vector{Int8}}
    leaf_nearest_triangle::Union{Nothing,Vector{Int}}
end

"""
    _normalize_normal(n_vec::Vec) -> SVector{3,Float64}

Extract and normalize a Meshes.jl Vec normal to a unit SVector{3,Float64}.
"""
@inline function _normalize_normal(n_vec)
    # Direct component access avoids tuple overhead
    n = SVector{3,Float64}(ustrip(n_vec[1]), ustrip(n_vec[2]), ustrip(n_vec[3]))
    n_mag = norm(n)
    if n_mag < eps(Float64) * 100
        error("Degenerate triangle: zero normal")
    end
    return n / n_mag
end

"""
    _extract_vertex(vert) -> SVector{3,Float64}

Extract coordinates from a Meshes.jl vertex to an SVector{3,Float64}.
"""
@inline function _extract_vertex(vert)
    coords = Meshes.to(vert)
    return SVector{3,Float64}(ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3]))
end

"""
    _extract_triangle_data(mesh::SimpleMesh) -> StructArray

Extract and cache triangle vertex and normal data from a SimpleMesh.
Returns a StructArray with v1, v2, v3, normal fields as SVectors.

Normals are computed using Meshes.normal(elem) which uses the proper
geometric normal of the triangle. This is more reliable than computing
from vertex winding order, which can be inconsistent in STL files.

Performance: Parallelized extraction using OhMyThreads for large meshes.
"""
function _extract_triangle_data(mesh::SimpleMesh{M,C}) where {M,C}
    # Collect elements once to enable indexed parallel access
    elems = collect(elements(mesh))

    # Parallel extraction using OhMyThreads tmap
    data = tmap(elems) do elem
        verts = Meshes.vertices(elem)

        if length(verts) != 3
            @warn "Skipping non-triangular element with $(length(verts)) vertices"
            return (
                v1 = SVector{3,Float64}(0.0, 0.0, 0.0),
                v2 = SVector{3,Float64}(0.0, 0.0, 0.0),
                v3 = SVector{3,Float64}(0.0, 0.0, 0.0),
                normal = SVector{3,Float64}(0.0, 0.0, 1.0),
            )
        end

        v1 = _extract_vertex(verts[1])
        v2 = _extract_vertex(verts[2])
        v3 = _extract_vertex(verts[3])

        # Compute normal from element
        n_vec = Meshes.normal(elem)
        n = SVector{3,Float64}(ustrip(n_vec[1]), ustrip(n_vec[2]), ustrip(n_vec[3]))
        n_mag = norm(n)

        normal = if n_mag < eps(Float64) * 100
            SVector{3,Float64}(0.0, 0.0, 1.0)  # Degenerate triangle
        else
            n / n_mag
        end

        return (v1 = v1, v2 = v2, v3 = v3, normal = normal)
    end

    return StructArray(data)
end

"""
    _compute_bbox(triangles::StructArray) -> (bbox_min, bbox_max)

Compute bounding box from cached triangle data.

Performance: Single-pass min/max accumulation without intermediate allocations.
"""
function _compute_bbox(triangles::StructArray)
    T = eltype(eltype(triangles.v1))
    n = length(triangles)

    # Initialize with first triangle's vertices
    v = triangles.v1[1]
    min_x, min_y, min_z = v[1], v[2], v[3]
    max_x, max_y, max_z = v[1], v[2], v[3]

    # Single-pass accumulation over all vertices
    @inbounds for i = 1:n
        # Vertex 1
        v = triangles.v1[i]
        min_x = min(min_x, v[1])
        min_y = min(min_y, v[2])
        min_z = min(min_z, v[3])
        max_x = max(max_x, v[1])
        max_y = max(max_y, v[2])
        max_z = max(max_z, v[3])

        # Vertex 2
        v = triangles.v2[i]
        min_x = min(min_x, v[1])
        min_y = min(min_y, v[2])
        min_z = min(min_z, v[3])
        max_x = max(max_x, v[1])
        max_y = max(max_y, v[2])
        max_z = max(max_z, v[3])

        # Vertex 3
        v = triangles.v3[i]
        min_x = min(min_x, v[1])
        min_y = min(min_y, v[2])
        min_z = min(min_z, v[3])
        max_x = max(max_x, v[1])
        max_y = max(max_y, v[2])
        max_z = max(max_z, v[3])
    end

    # Handle degenerate cases (zero extent in any dimension)
    eps_val = max(eps(T) * 100, T(1.0e-10))
    if min_x == max_x
        min_x -= eps_val
        max_x += eps_val
    end
    if min_y == max_y
        min_y -= eps_val
        max_y += eps_val
    end
    if min_z == max_z
        min_z -= eps_val
        max_z += eps_val
    end

    return SVector{3,T}(min_x, min_y, min_z), SVector{3,T}(max_x, max_y, max_z)
end

"""
    _edge_key(v1::SVector{3,T}, v2::SVector{3,T}) where {T} -> Tuple

Create a canonical edge key from two vertices (order-independent).
Uses component-wise comparison for consistent ordering.
"""
@inline function _edge_key(v1::SVector{3,T}, v2::SVector{3,T}) where {T}
    # Order vertices lexicographically for canonical key
    if v1[1] < v2[1] ||
       (v1[1] == v2[1] && v1[2] < v2[2]) ||
       (v1[1] == v2[1] && v1[2] == v2[2] && v1[3] < v2[3])
        return (v1, v2)
    else
        return (v2, v1)
    end
end

"""
    has_consistent_normals(triangles::StructArray; alignment_threshold=-0.3) -> Bool

Check if triangle normals are consistently oriented.

Checks if edge-adjacent triangles (sharing 2 vertices) have normals pointing
in similar directions. Inconsistent normals indicate a mesh quality issue that
can cause incorrect inside/outside queries.

# Arguments
- `triangles::StructArray`: Cached triangle data with v1, v2, v3, normal fields
- `alignment_threshold::Float64=-0.3`: Minimum dot product for consistent normals.
  - `-1.0` to `1.0` range: `-1.0` = opposite directions, `1.0` = same direction
  - Default `-0.3` allows some surface curvature while detecting major flips

# Performance
O(n) complexity using edge hash map instead of O(n²) pairwise comparison.

# Examples
```julia
octree = TriangleOctree("model.stl"; h_min=0.01)
if !has_consistent_normals(octree.triangles)
    @warn "Mesh has inconsistent normals - may need repair"
end
```
"""
function has_consistent_normals(triangles::StructArray; alignment_threshold::Float64 = -0.3)
    n = length(triangles)
    n <= 1 && return true

    # Build edge → first triangle index map in O(n)
    # When we encounter an edge again, check normal alignment immediately
    edge_map = Dict{Tuple{SVector{3,Float64},SVector{3,Float64}},Int}()
    sizehint!(edge_map, 3 * n)  # Each triangle has 3 edges

    @inbounds for i = 1:n
        v1 = triangles.v1[i]
        v2 = triangles.v2[i]
        v3 = triangles.v3[i]
        normal_i = triangles.normal[i]

        # Process each edge of triangle i
        for (ea, eb) in ((v1, v2), (v2, v3), (v3, v1))
            key = _edge_key(ea, eb)
            existing = get(edge_map, key, 0)

            if existing != 0
                # Edge shared with another triangle - check alignment
                alignment = dot(normal_i, triangles.normal[existing])
                if alignment < alignment_threshold
                    return false
                end
            else
                # First time seeing this edge
                edge_map[key] = i
            end
        end
    end

    return true
end

"""
    TriangleOctree(mesh::SimpleMesh{M,C};
                   h_min,
                   max_triangles_per_box::Int=50,
                   classify_leaves::Bool=true,
                   verify_orientation::Bool=true) -> TriangleOctree{M,C,Float64}

Build an octree spatial index for a SimpleMesh.

# Arguments
- `mesh::SimpleMesh{M,C}`: Meshes.jl SimpleMesh to index
- `h_min`: Minimum octree box size (stopping criterion)
- `max_triangles_per_box::Int=50`: Maximum triangles per leaf before subdivision
- `classify_leaves::Bool=true`: Whether to classify empty leaves as interior/exterior
- `verify_orientation::Bool=true`: Check if triangle normals are consistently oriented

# Algorithm
1. (Optional) Verify triangle normal consistency
2. Extract and cache triangle vertex/normal data
3. Create root octree covering mesh bounding box
4. Distribute triangles to boxes using SAT intersection tests
5. Recursively subdivide boxes exceeding triangle threshold
6. Balance octree to ensure 2:1 refinement constraint
7. (Optional) Classify empty leaves via BFS from boundary

# Returns
`TriangleOctree{M,C,Float64}` ready for fast spatial queries

# Example
```julia
mesh = GeoIO.load("box.stl").geometry
octree = TriangleOctree(mesh; h_min=0.01, max_triangles_per_box=50)
println("Built octree with ", num_leaves(octree), " leaves")
```
"""
function TriangleOctree(
    mesh::SimpleMesh{M,C};
    h_min,
    max_triangles_per_box::Int = 50,
    classify_leaves::Bool = true,
    verify_orientation::Bool = true,
) where {M<:Manifold,C<:CRS}
    T = Float64
    h_min_val = T(ustrip(h_min))

    triangles = _extract_triangle_data(mesh)

    if verify_orientation && !has_consistent_normals(triangles)
        @warn """
        Triangle mesh has inconsistent normal orientations!

        Some triangles may have normals pointing inward while others point outward.
        This can lead to incorrect isinside() and signed distance calculations.
        """
    end

    bbox_min, bbox_max = _compute_bbox(triangles)
    bbox_sz = bbox_max - bbox_min
    root_size = maximum(bbox_sz)

    estimated_boxes = max(1000, length(triangles) * 2)
    tree = SpatialOctree{Int,T}(bbox_min, root_size; initial_capacity = estimated_boxes)

    root_elements = tree.element_lists[1]
    for tri_idx = 1:length(triangles)
        push!(root_elements, tri_idx)
    end

    size_criterion = SizeCriterion(h_min_val)
    criterion = AndCriterion((MaxElementsCriterion(max_triangles_per_box), size_criterion))

    _subdivide_triangle_octree!(tree, triangles, 1, criterion)

    balance_octree!(tree, size_criterion)

    classification, nearest_triangle = if classify_leaves
        _classify_leaves(tree, triangles)
    else
        (nothing, nothing)
    end

    return TriangleOctree(tree, mesh, triangles, classification, nearest_triangle)
end

"""
    TriangleOctree(filepath::String; h_min, kwargs...) -> TriangleOctree

Build an octree spatial index from an STL file.

# Arguments
- `filepath::String`: Path to STL file
- `h_min`: Minimum octree box size
- `kwargs...`: Additional arguments passed to TriangleOctree(mesh; ...)

# Example
```julia
octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)
```
"""
function TriangleOctree(filepath::String; h_min, kwargs...)
    geo = GeoIO.load(filepath)
    return TriangleOctree(geo.geometry; h_min, kwargs...)
end

"""
    _subdivide_triangle_octree!(tree, triangles, box_idx, criterion)

Recursively subdivide a box in the triangle octree.

For each box that needs subdivision:
1. Create 8 child boxes
2. For each triangle in parent, find which children it intersects
3. Distribute triangle indices to children
4. Recursively subdivide children if they meet criteria
"""
function _subdivide_triangle_octree!(
    tree::SpatialOctree{Int,T},
    triangles::StructArray,
    box_idx::Int,
    criterion,
) where {T<:Real}
    if !should_subdivide(criterion, tree, box_idx)
        return
    end

    parent_triangles = tree.element_lists[box_idx]
    isempty(parent_triangles) && return

    subdivide!(tree, box_idx)

    children = tree.children[box_idx]

    for tri_idx in parent_triangles
        v1 = triangles.v1[tri_idx]
        v2 = triangles.v2[tri_idx]
        v3 = triangles.v3[tri_idx]

        for child_idx in children
            child_idx == 0 && continue

            child_min, child_max = box_bounds(tree, child_idx)

            if triangle_box_intersection(v1, v2, v3, child_min, child_max)
                push!(tree.element_lists[child_idx], tri_idx)
            end
        end
    end

    for child_idx in children
        child_idx == 0 && continue
        _subdivide_triangle_octree!(tree, triangles, child_idx, criterion)
    end
    return
end

"""
    _classify_leaves(tree::SpatialOctree{Int,T}, triangles::StructArray) -> (Vector{Int8}, Vector{Int})

Classify octree leaves as exterior (0), boundary (1), or interior (2).

# Algorithm
1. Mark leaves containing triangles as boundary (1)
2. Breadth-first propagation from boundary leaves
3. Compute signed distance at empty leaf centers
4. Classify based on sign: negative = interior (2), positive = exterior (0)
5. Store nearest triangle index for each leaf (used at query time)

# Returns
Tuple of:
- `classification::Vector{Int8}`: Classification for each box (0=exterior, 1=boundary, 2=interior)
- `nearest_triangle::Vector{Int}`: Index of nearest triangle for each box (0 if not computed)
"""
function _classify_leaves(
    tree::SpatialOctree{Int,T},
    triangles::StructArray,
) where {T<:Real}
    n_boxes = length(tree.element_lists)
    classification = zeros(Int8, n_boxes)
    nearest_triangle = zeros(Int, n_boxes)
    distances = fill(T(Inf), n_boxes)
    visited = falses(n_boxes)

    queue = Int[]
    for leaf_idx in all_leaves(tree)
        tri_list = tree.element_lists[leaf_idx]
        if !isempty(tri_list)
            classification[leaf_idx] = 1
            distances[leaf_idx] = T(1.0e-6)
            nearest_triangle[leaf_idx] = first(tri_list)
            visited[leaf_idx] = true
            push!(queue, leaf_idx)
        end
    end

    while !isempty(queue)
        next_queue = Int[]

        for box_idx in queue
            for direction = 1:6
                neighbors = find_neighbor(tree, box_idx, direction)

                for neighbor_idx in neighbors
                    (visited[neighbor_idx] || !is_leaf(tree, neighbor_idx)) && continue

                    neighbor_center = box_center(tree, neighbor_idx)
                    signed_dist, closest_tri_idx =
                        _compute_signed_distance_with_index(neighbor_center, triangles)

                    distances[neighbor_idx] = signed_dist
                    nearest_triangle[neighbor_idx] = closest_tri_idx
                    visited[neighbor_idx] = true

                    cell_radius = box_size(tree, neighbor_idx) * sqrt(3) / 2
                    if abs(signed_dist) < cell_radius
                        classification[neighbor_idx] = 1
                    elseif signed_dist < 0
                        classification[neighbor_idx] = 2
                    else
                        classification[neighbor_idx] = 0
                    end

                    if abs(signed_dist) < 10 * box_size(tree, neighbor_idx)
                        push!(next_queue, neighbor_idx)
                    end
                end
            end
        end

        queue = next_queue
    end

    return classification, nearest_triangle
end

"""
    _compute_signed_distance(point::SVector{3,T}, triangles::StructArray) -> T

Compute signed distance from a point to the closest triangle.

Returns:
- Positive if point is on the side of the normal (outside)
- Negative if point is on the opposite side of the normal (inside)
- Zero if point is exactly on the surface
"""
function _compute_signed_distance(
    point::SVector{3,T},
    triangles::StructArray,
) where {T<:Real}
    min_dist = T(Inf)
    closest_idx = 0

    for i = 1:length(triangles)
        closest_pt = closest_point_on_triangle(
            point,
            triangles.v1[i],
            triangles.v2[i],
            triangles.v3[i],
        )
        dist = norm(point - closest_pt)

        if dist < abs(min_dist)
            to_point = point - closest_pt
            sign = dot(to_point, triangles.normal[i]) >= 0 ? 1 : -1
            min_dist = sign * dist
            closest_idx = i
        end
    end

    return min_dist
end

"""
    _compute_signed_distance_with_index(point::SVector{3,T}, triangles::StructArray) -> (T, Int)

Compute signed distance from a point to the closest triangle, returning both the
distance and the index of the closest triangle.

Returns:
- `(distance, closest_idx)` tuple where:
  - `distance > 0`: point is on the side of the normal (outside)
  - `distance < 0`: point is on the opposite side of the normal (inside)
  - `distance == 0`: point is exactly on the surface
  - `closest_idx`: 1-based index of the nearest triangle
"""
function _compute_signed_distance_with_index(
    point::SVector{3,T},
    triangles::StructArray,
) where {T<:Real}
    min_dist = T(Inf)
    closest_idx = 0

    for i = 1:length(triangles)
        closest_pt = closest_point_on_triangle(
            point,
            triangles.v1[i],
            triangles.v2[i],
            triangles.v3[i],
        )
        dist = norm(point - closest_pt)

        if dist < abs(min_dist)
            to_point = point - closest_pt
            sign = dot(to_point, triangles.normal[i]) >= 0 ? 1 : -1
            min_dist = sign * dist
            closest_idx = i
        end
    end

    return min_dist, closest_idx
end

Base.length(octree::TriangleOctree) = length(octree.triangles)
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))
num_triangles(octree::TriangleOctree) = length(octree.triangles)

#=============================================================================
Fast Spatial Queries (Layer 4)
=============================================================================#

"""
    isinside(point::SVector{3,T}, octree::TriangleOctree) -> Bool

Fast interior/exterior test using octree spatial index.

Returns `true` if point is inside the closed surface defined by the mesh.

# Algorithm
1. Find leaf box containing point (O(log M) via octree traversal)
2. If leaf contains triangles: compute signed distance to k local triangles
3. If leaf is empty: use pre-computed classification (O(1) lookup)
4. Return: distance < 0 indicates interior

# Performance
- Complexity: O(log M + k) where M = number of triangles, k ≈ 10-50
- Speedup: 100-1000× faster than brute-force O(M) approach

# Example
```julia
using WhatsThePoint, StaticArrays

octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)

# Fast query
point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)  # ~0.05ms vs ~50ms brute-force
```
"""
function isinside(point::SVector{3,T}, octree::TriangleOctree) where {T<:Real}
    tree = octree.tree
    triangles = octree.triangles

    leaf_idx = find_leaf(tree, point)

    if isnothing(octree.leaf_classification)
        error(
            """Cannot query without classification!
            Rebuild TriangleOctree with classify_leaves=true to enable full isinside() queries.""",
        )
    end

    classification = octree.leaf_classification[leaf_idx]

    if classification == Int8(2)
        return true
    elseif classification == Int8(0)
        return false
    else
        tri_indices = tree.element_lists[leaf_idx]

        if !isempty(tri_indices)
            dist = _compute_local_signed_distance(point, triangles, tri_indices)
            return dist < 0
        else
            neighbor_triangles = Int[]
            for direction = 1:6
                neighbors = find_neighbor(tree, leaf_idx, direction)
                for neighbor_idx in neighbors
                    if is_leaf(tree, neighbor_idx)
                        append!(neighbor_triangles, tree.element_lists[neighbor_idx])
                    end
                end
            end

            if !isempty(neighbor_triangles)
                dist = _compute_local_signed_distance(point, triangles, neighbor_triangles)
                return dist < 0
            else
                if !isnothing(octree.leaf_nearest_triangle)
                    tri_idx = octree.leaf_nearest_triangle[leaf_idx]
                    if tri_idx > 0
                        dist = _compute_local_signed_distance(point, triangles, [tri_idx])
                        return dist < 0
                    end
                end
                dist = _compute_signed_distance(point, triangles)
                @warn "Falling back to global signed distance computation - this indicates a bug"
                return dist < 0
            end
        end
    end
end

"""
    _compute_local_signed_distance(
        point::SVector{3,T},
        triangles::StructArray,
        tri_indices::Vector{Int}
    ) -> T

Compute signed distance from point to nearest triangle in the local set.

This is the key performance optimization: instead of checking all M triangles,
we only check the k≈10-50 triangles in the point's octree leaf.

Performance: Fused computation avoids redundant closest_point calculation.
"""
function _compute_local_signed_distance(
    point::SVector{3,T},
    triangles::StructArray,
    tri_indices::Vector{Int},
) where {T<:Real}
    min_dist_sq = typemax(T)
    closest_idx = 0
    closest_pt = point  # Will be overwritten

    @inbounds for tri_idx in tri_indices
        v1 = triangles.v1[tri_idx]
        v2 = triangles.v2[tri_idx]
        v3 = triangles.v3[tri_idx]

        # Compute closest point on this triangle
        cp = closest_point_on_triangle(point, v1, v2, v3)
        diff = point - cp
        dist_sq = dot(diff, diff)  # Squared distance avoids sqrt

        if dist_sq < min_dist_sq
            min_dist_sq = dist_sq
            closest_idx = tri_idx
            closest_pt = cp
        end
    end

    # Compute sign using normal of closest triangle
    to_point = point - closest_pt
    sign = dot(to_point, triangles.normal[closest_idx]) < 0 ? -1 : 1

    return sign * sqrt(min_dist_sq)
end

"""
    isinside(points::Vector{SVector{3,T}}, octree::TriangleOctree) -> Vector{Bool}

Batch interior test for multiple points.

# Example
```julia
test_points = [SVector(randn(3)...) for _ in 1:1000]
results = isinside(test_points, octree)  # Returns Vector{Bool}
```
"""
function isinside(points::Vector{SVector{3,T}}, octree::TriangleOctree) where {T<:Real}
    return [isinside(p, octree) for p in points]
end
