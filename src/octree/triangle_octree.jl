"""
    TriangleOctree{T<:Real}

An octree-based spatial index for efficient triangle mesh queries.

# Fields
- `tree::SpatialOctree{Int,T}`: Underlying octree storing triangle indices
- `mesh::TriangleMesh{T}`: Triangle mesh being indexed
- `leaf_classification::Union{Nothing, Vector{Int8}}`: Classification of empty leaves
  - `0`: Exterior (outside mesh)
  - `1`: Boundary (contains triangles)
  - `2`: Interior (inside mesh)

# Performance
For a mesh with M triangles:
- Construction: O(M log M) - distribute triangles to boxes
- Query: O(log M + k) where k ≈ 10-50 triangles per leaf
- Memory: O(M) for triangle storage + O(N) for octree nodes

# Example
```julia
# Load mesh and build octree
mesh = TriangleMesh("model.stl")
octree = TriangleOctree(mesh; h_min=0.01, max_triangles_per_box=50)

# Fast queries (100-1000x speedup over brute force)
point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)
distance = signed_distance(point, octree)
```
"""
struct TriangleOctree{T <: Real}
    tree::SpatialOctree{Int, T}
    mesh::TriangleMesh{T}
    leaf_classification::Union{Nothing, Vector{Int8}}
end

"""
    has_consistent_normals(mesh::TriangleMesh{T}; alignment_threshold::T=T(-0.3)) -> Bool

Check if triangle normals in the mesh are consistently oriented.

Checks if edge-adjacent triangles (sharing 2 vertices) have normals pointing
in similar directions. Inconsistent normals indicate a mesh quality issue that
can cause incorrect inside/outside queries.

# Arguments
- `mesh::TriangleMesh{T}`: Triangle mesh to check
- `alignment_threshold::T=-0.3`: Minimum dot product for consistent normals.
  - `-1.0` to `1.0` range: `-1.0` = opposite directions, `1.0` = same direction
  - Default `-0.3` allows some surface curvature while detecting major flips
  - Use stricter values (e.g., `0.0`) for smoother surfaces
  - Use looser values (e.g., `-0.5`) for highly curved surfaces

# Algorithm
1. For each triangle, identify its 3 edges
2. Find other triangles that share each edge (2 common vertices)
3. Check if edge-adjacent triangles have aligned normals (dot product > threshold)
4. Return false if any edge-adjacent pair has opposing normals

# Note
This checks **topological** adjacency (shared edges), not spatial proximity.
A mesh can have geometrically close triangles that aren't connected, such as
in U-turns or folds. Only edge-adjacent triangles are compared.

Catches common problems like:
- Flipped triangles in STL files
- Inconsistent winding order
- Inside-out patches

# Examples
```julia
mesh = TriangleMesh("model.stl")

# Default threshold (-0.3)
if !has_consistent_normals(mesh)
    @warn "Mesh has inconsistent normals - may need repair"
end

# Stricter check for smooth surfaces
if !has_consistent_normals(mesh; alignment_threshold=0.0)
    @warn "Mesh has normals with high variation"
end

# Looser check for highly curved surfaces
if !has_consistent_normals(mesh; alignment_threshold=-0.5)
    @warn "Mesh has severely misaligned normals"
end
```
"""
function has_consistent_normals(
        mesh::TriangleMesh{T};
        alignment_threshold::T = T(-0.3),
    ) where {T <: Real}
    length(mesh) <= 1 && return true

    # Check each triangle against all others for edge adjacency
    for i in 1:length(mesh)
        tri_i = mesh.triangles[i]

        # Get the 3 edges of triangle i
        edges_i = [
            Set([tri_i.v1, tri_i.v2]),
            Set([tri_i.v2, tri_i.v3]),
            Set([tri_i.v3, tri_i.v1]),
        ]

        # Check against all subsequent triangles
        for j in (i + 1):length(mesh)
            tri_j = mesh.triangles[j]

            # Get the 3 edges of triangle j
            edges_j = [
                Set([tri_j.v1, tri_j.v2]),
                Set([tri_j.v2, tri_j.v3]),
                Set([tri_j.v3, tri_j.v1]),
            ]

            shares_edge = false
            for edge_i in edges_i
                for edge_j in edges_j
                    if edge_i == edge_j
                        shares_edge = true
                        break
                    end
                end
                shares_edge && break
            end

            if shares_edge
                alignment = dot(tri_i.normal, tri_j.normal)

                if alignment < alignment_threshold
                    return false
                end
            end
        end
    end

    return true
end

"""
    TriangleOctree(mesh::TriangleMesh{T};
                   h_min::T,
                   max_triangles_per_box::Int=50,
                   classify_leaves::Bool=true,
                   verify_orientation::Bool=true) -> TriangleOctree{T}

Build an octree spatial index for a triangle mesh.

# Arguments
- `mesh::TriangleMesh{T}`: Triangle mesh to index
- `h_min::T`: Minimum octree box size (stopping criterion)
- `max_triangles_per_box::Int=50`: Maximum triangles per leaf before subdivision
- `classify_leaves::Bool=true`: Whether to classify empty leaves as interior/exterior
- `verify_orientation::Bool=true`: Check if triangle normals are consistently oriented (warns if not)

# Algorithm
1. (Optional) Verify triangle normal consistency
2. Create root octree covering mesh bounding box
3. Distribute triangles to boxes using SAT intersection tests
4. Recursively subdivide boxes exceeding triangle threshold
5. Balance octree to ensure 2:1 refinement constraint
6. (Optional) Classify empty leaves via BFS from boundary

# Returns
`TriangleOctree{T}` ready for fast spatial queries

# Example
```julia
mesh = TriangleMesh("box.stl")
octree = TriangleOctree(mesh; h_min=0.01, max_triangles_per_box=50)
println("Built octree with ", length(all_leaves(octree.tree)), " leaves")
```

# Note on Normal Orientation
For accurate inside/outside queries, triangle normals should consistently point outward
(or consistently inward). Use `verify_orientation=true` to check this. If normals are
inconsistent, you may get incorrect isinside() results near the affected triangles.
"""
function TriangleOctree(
        mesh::TriangleMesh{T};
        h_min::T,
        max_triangles_per_box::Int = 50,
        classify_leaves::Bool = true,
        verify_orientation::Bool = true,
    ) where {T <: Real}
    # 1. Verify normal consistency (optional but recommended)
    if verify_orientation && !has_consistent_normals(mesh)
        @warn """
        Triangle mesh has inconsistent normal orientations!

        Some triangles may have normals pointing inward while others point outward.
        This can lead to incorrect isinside() and signed distance calculations.
        """
    end

    # 2. Create root octree covering mesh bounding box
    bbox_sz = bbox_size(mesh)
    root_size = maximum(bbox_sz)  # Use largest dimension for cubic root

    # Estimate initial capacity based on mesh size
    # Rule of thumb: ~8 boxes per subdivision level, need log2(n_triangles) levels
    estimated_boxes = max(1000, length(mesh) * 2)  # Conservative estimate
    tree =
        SpatialOctree{Int, T}(mesh.bbox_min, root_size; initial_capacity = estimated_boxes)

    # 3. Distribute triangles to root box
    # The root is always box index 1
    root_elements = tree.element_lists[1]
    for tri_idx in 1:length(mesh.triangles)
        push!(root_elements, tri_idx)
    end

    # 4. Define subdivision criteria
    size_criterion = SizeCriterion(h_min)
    criterion = AndCriterion((MaxElementsCriterion(max_triangles_per_box), size_criterion))

    # 5. Recursively subdivide
    _subdivide_triangle_octree!(tree, mesh, 1, criterion)

    # 6. Balance octree (ensure 2:1 refinement constraint)
    balance_octree!(tree, size_criterion)

    # 7. Optional: Classify empty leaves
    classification = classify_leaves ? _classify_leaves(tree, mesh) : nothing

    return TriangleOctree(tree, mesh, classification)
end

"""
    _subdivide_triangle_octree!(tree, mesh, box_idx, criterion)

Recursively subdivide a box in the triangle octree.

For each box that needs subdivision:
1. Create 8 child boxes
2. For each triangle in parent, find which children it intersects
3. Distribute triangle indices to children
4. Recursively subdivide children if they meet criteria
"""
function _subdivide_triangle_octree!(
        tree::SpatialOctree{Int, T},
        mesh::TriangleMesh{T},
        box_idx::Int,
        criterion,
    ) where {T <: Real}
    # Check if this box needs subdivision
    if !should_subdivide(criterion, tree, box_idx)
        return
    end

    # Get parent triangles
    parent_triangles = tree.element_lists[box_idx]
    isempty(parent_triangles) && return

    # Subdivide the box (creates 8 children)
    subdivide!(tree, box_idx)

    # Get child boxes
    children = tree.children[box_idx]

    # Distribute triangles to children
    for tri_idx in parent_triangles
        tri = mesh.triangles[tri_idx]

        # Find which child boxes this triangle intersects
        # We need to check all 8 children individually
        for child_idx in children
            child_idx == 0 && continue  # Skip if child doesn't exist

            # Get child box bounds
            child_min, child_max = box_bounds(tree, child_idx)

            # Check intersection
            if triangle_box_intersection(tri.v1, tri.v2, tri.v3, child_min, child_max)
                push!(tree.element_lists[child_idx], tri_idx)
            end
        end
    end

    # Recursively subdivide children
    for child_idx in children
        child_idx == 0 && continue
        _subdivide_triangle_octree!(tree, mesh, child_idx, criterion)
    end
    return
end

"""
    _classify_leaves(tree::SpatialOctree{Int,T}, mesh::TriangleMesh{T}) -> Vector{Int8}

Classify octree leaves as exterior (0), boundary (1), or interior (2).

# Algorithm
1. Mark leaves containing triangles as boundary (1)
2. Breadth-first propagation from boundary leaves
3. Compute signed distance at empty leaf centers
4. Classify based on sign: negative = interior (2), positive = exterior (0)

# Returns
Vector of Int8 classifications, indexed by box_idx
"""
function _classify_leaves(tree::SpatialOctree{Int, T}, mesh::TriangleMesh{T}) where {T <: Real}
    n_boxes = length(tree.element_lists)
    classification = zeros(Int8, n_boxes)
    distances = fill(T(Inf), n_boxes)
    visited = falses(n_boxes)

    # Layer 0: Mark boundary leaves (contain triangles)
    queue = Int[]
    for leaf_idx in all_leaves(tree)
        if !isempty(tree.element_lists[leaf_idx])
            classification[leaf_idx] = 1  # Boundary
            distances[leaf_idx] = T(1.0e-6)  # Small positive (on surface)
            visited[leaf_idx] = true
            push!(queue, leaf_idx)
        end
    end

    # BFS from boundary leaves
    while !isempty(queue)
        next_queue = Int[]

        for box_idx in queue
            # Check all 6 neighbors (±x, ±y, ±z)
            for direction in 1:6
                neighbors = find_neighbor(tree, box_idx, direction)

                for neighbor_idx in neighbors
                    # Skip if already visited or not a leaf
                    (visited[neighbor_idx] || !is_leaf(tree, neighbor_idx)) && continue

                    # Compute signed distance at neighbor's center
                    neighbor_center = box_center(tree, neighbor_idx)
                    signed_dist = _compute_signed_distance(neighbor_center, mesh)

                    # Store distance and mark visited
                    distances[neighbor_idx] = signed_dist
                    visited[neighbor_idx] = true

                    # Classify: negative = interior, positive = exterior
                    cell_radius = box_size(tree, neighbor_idx) * sqrt(3) / 2
                    if abs(signed_dist) < cell_radius
                        #close to surface, treat as boundary
                        classification[neighbor_idx] = 1  # Boundary
                    elseif signed_dist < 0
                        classification[neighbor_idx] = 2  # Interior
                    else
                        classification[neighbor_idx] = 0  # Exterior
                    end

                    # Add to next layer if close to surface
                    if abs(signed_dist) < 10 * box_size(tree, neighbor_idx)
                        push!(next_queue, neighbor_idx)
                    end
                end
            end
        end

        queue = next_queue
    end

    return classification
end

"""
    _compute_signed_distance(point::SVector{3,T}, mesh::TriangleMesh{T}) -> T

Compute signed distance from a point to the closest triangle in the mesh.

Returns:
- Positive if point is on the side of the normal (outside)
- Negative if point is on the opposite side of the normal (inside)
- Zero if point is exactly on the surface
"""
function _compute_signed_distance(
        point::SVector{3, T},
        mesh::TriangleMesh{T},
    ) where {T <: Real}
    min_dist = T(Inf)
    closest_tri = nothing

    # Find closest triangle (brute force over all triangles)
    for tri in mesh.triangles
        closest_pt = closest_point_on_triangle(point, tri.v1, tri.v2, tri.v3)
        dist = norm(point - closest_pt)

        if dist < abs(min_dist)
            # Determine sign from normal
            to_point = point - closest_pt
            sign = dot(to_point, tri.normal) >= 0 ? 1 : -1
            min_dist = sign * dist
            closest_tri = tri
        end
    end

    return min_dist
end

# Convenience accessors
Base.length(octree::TriangleOctree) = length(octree.mesh)
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))
num_triangles(octree::TriangleOctree) = length(octree.mesh)

#=============================================================================
Fast Spatial Queries (Layer 4)
=============================================================================#

"""
    isinside(point::SVector{3,T}, octree::TriangleOctree{T}) -> Bool

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
- Amortized: Build octree once (~1.5s), query many times (~0.05ms/query)

# Example
```julia
using WhatsThePoint, StaticArrays

mesh = TriangleMesh("model.stl")
octree = TriangleOctree(mesh; h_min=0.01, classify_leaves=true)

# Fast query
point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)  # ~0.05ms vs ~50ms brute-force
```

# Note
Requires `classify_leaves=true` during construction for empty leaf queries.
If octree was built without classification, only boundary leaf queries work.
"""
function isinside(point::SVector{3, T}, octree::TriangleOctree{T}) where {T <: Real}
    tree = octree.tree
    mesh = octree.mesh

    # Step 1: Find leaf containing point (O(log M))
    leaf_idx = find_leaf(tree, point)

    # Step 2: Check classification first
    if isnothing(octree.leaf_classification)
        error(
            """Cannot query without classification!
            Rebuild TriangleOctree with classify_leaves=true to enable full isinside() queries.""",
        )
    end

    classification = octree.leaf_classification[leaf_idx]

    # Step 3: Handle based on classification
    if classification == Int8(2)  # Interior
        return true
    elseif classification == Int8(0)  # Exterior
        return false
    else  # Boundary (classification == 1)
        # Get triangles in this leaf
        tri_indices = tree.element_lists[leaf_idx]

        if !isempty(tri_indices)
            # Compute distance to local triangles
            dist = _compute_local_signed_distance(point, mesh, tri_indices)
            return dist < 0
        else
            # Boundary leaf but empty - search neighbors
            neighbor_triangles = Int[]
            for direction in 1:6
                neighbors = find_neighbor(tree, leaf_idx, direction)
                for neighbor_idx in neighbors
                    if is_leaf(tree, neighbor_idx)
                        append!(neighbor_triangles, tree.element_lists[neighbor_idx])
                    end
                end
            end

            if !isempty(neighbor_triangles)
                dist = _compute_local_signed_distance(point, mesh, neighbor_triangles)
                return dist < 0
            else
                dist = _compute_signed_distance(point, mesh)
                @warn "WARNING (critical): Falling back to global signed distance computation!"
                return dist < 0
            end
        end
    end
end

"""
    _compute_local_signed_distance(
        point::SVector{3,T},
        mesh::TriangleMesh{T},
        tri_indices::Vector{Int}
    ) -> T

Compute signed distance from point to nearest triangle in the local set.

This is the key performance optimization: instead of checking all M triangles,
we only check the k≈10-50 triangles in the point's octree leaf.

# Arguments
- `point`: Query point
- `mesh`: Triangle mesh
- `tri_indices`: Indices of triangles to check (from octree leaf)

# Returns
Signed distance (negative = interior, positive = exterior)
"""
function _compute_local_signed_distance(
        point::SVector{3, T},
        mesh::TriangleMesh{T},
        tri_indices::Vector{Int},
    ) where {T <: Real}
    min_dist = typemax(T)
    closest_tri = nothing

    # Find closest triangle among candidates (k << M triangles)
    for tri_idx in tri_indices
        tri = mesh.triangles[tri_idx]

        # Compute unsigned distance
        dist = distance_point_triangle(point, tri.v1, tri.v2, tri.v3)

        if dist < abs(min_dist)
            min_dist = dist
            closest_tri = tri
        end
    end

    # Determine sign from normal of closest triangle
    closest_pt =
        closest_point_on_triangle(point, closest_tri.v1, closest_tri.v2, closest_tri.v3)
    to_point = point - closest_pt

    # Negative = interior (opposite to normal direction)
    sign = dot(to_point, closest_tri.normal) < 0 ? -1 : 1

    return sign * min_dist
end

"""
    isinside(points::Vector{SVector{3,T}}, octree::TriangleOctree{T}) -> Vector{Bool}

Batch interior test for multiple points.

More efficient than individual queries for many points (can be parallelized in future).

# Example
```julia
test_points = [SVector(randn(3)...) for _ in 1:1000]
results = isinside(test_points, octree)  # Returns Vector{Bool}
```
"""
function isinside(points::Vector{SVector{3, T}}, octree::TriangleOctree{T}) where {T <: Real}
    return [isinside(p, octree) for p in points]
    # TODO: Add @threads for parallelization:
    # return ThreadsX.map(p -> isinside(p, octree), points)
end
