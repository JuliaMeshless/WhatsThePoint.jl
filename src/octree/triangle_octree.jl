"""
    TriangleOctree{M<:Manifold, C<:CRS, T<:Real}

An octree-based spatial index for efficient triangle mesh queries.

# Fields
- `tree::SpatialOctree{Int,T}`: Underlying octree storing triangle indices
- `mesh::SimpleMesh{M,C}`: Original Meshes.jl mesh (single source of truth for triangle data)
- `leaf_classification::Union{Nothing, Vector{Int8}}`: Classification of empty leaves
  - `0`: Exterior (outside mesh)
  - `1`: Boundary (near surface but empty)
  - `2`: Interior (inside mesh)

# Performance
For a mesh with M triangles:
- Construction: O(M log M) - distribute triangles to boxes
- Query: O(log M + k) where k ≈ 10-50 triangles per leaf
- Memory: O(N) for octree nodes (mesh data accessed on-the-fly)

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
    leaf_classification::Union{Nothing,Vector{Int8}}
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
    _get_triangle_vertices(mesh::SimpleMesh, tri_idx::Int) -> (SVector{3}, SVector{3}, SVector{3})

Extract triangle vertices from mesh as SVectors. Accesses mesh data on-the-fly.
"""
@inline function _get_triangle_vertices(mesh::SimpleMesh, tri_idx::Int)
    elem = mesh[tri_idx]
    verts = Meshes.vertices(elem)
    v1 = _extract_vertex(verts[1])
    v2 = _extract_vertex(verts[2])
    v3 = _extract_vertex(verts[3])
    return v1, v2, v3
end

"""
    _get_triangle_normal(mesh::SimpleMesh, tri_idx::Int) -> SVector{3,Float64}

Extract and normalize triangle normal from mesh. Accesses mesh data on-the-fly.
"""
@inline function _get_triangle_normal(mesh::SimpleMesh, tri_idx::Int)
    elem = mesh[tri_idx]
    return _normalize_normal(Meshes.normal(elem))
end

"""
    _compute_bbox(mesh::SimpleMesh) -> (bbox_min, bbox_max)

Compute bounding box from mesh triangle data.

Performance: Single-pass min/max accumulation over all triangles.
"""
function _compute_bbox(mesh::SimpleMesh)
    T = Float64
    n = Meshes.nelements(mesh)

    # Initialize with first triangle's vertices
    v1, v2, v3 = _get_triangle_vertices(mesh, 1)
    min_x = min(v1[1], v2[1], v3[1])
    min_y = min(v1[2], v2[2], v3[2])
    min_z = min(v1[3], v2[3], v3[3])
    max_x = max(v1[1], v2[1], v3[1])
    max_y = max(v1[2], v2[2], v3[2])
    max_z = max(v1[3], v2[3], v3[3])

    # Single-pass accumulation over all triangles
    for i in 2:n
        v1, v2, v3 = _get_triangle_vertices(mesh, i)

        min_x = min(min_x, v1[1], v2[1], v3[1])
        min_y = min(min_y, v1[2], v2[2], v3[2])
        min_z = min(min_z, v1[3], v2[3], v3[3])
        max_x = max(max_x, v1[1], v2[1], v3[1])
        max_y = max(max_y, v1[2], v2[2], v3[2])
        max_z = max(max_z, v1[3], v2[3], v3[3])
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
    has_consistent_normals(mesh::SimpleMesh) -> Bool

Check if triangle faces are consistently oriented (manifold orientation test).

This function verifies that all shared edges between adjacent triangles are
traversed in OPPOSITE directions, which is the geometric requirement for a
properly oriented manifold surface. This test is independent of surface curvature.

# Algorithm
For each shared edge between two triangles:
- Triangle A has edge v1→v2
- Triangle B (adjacent) must have edge v2→v1 (opposite direction)
- If both traverse in the same direction → faces are incorrectly oriented

# Returns
- `true` if all triangles are correctly oriented (manifold surface)
- `false` if any triangles have flipped faces (orientation errors)

# Performance
O(n) complexity using edge hash map instead of O(n²) pairwise comparison.

# Examples
```julia
octree = TriangleOctree("model.stl"; h_min=0.01)
if !has_consistent_normals(octree.mesh)
    @warn "Mesh has flipped triangles - will cause incorrect isinside() results"
end
```

# Note
This test uses GEOMETRIC edge orientation, not algebraic normal dot products.
It will correctly validate meshes with high curvature (sharp edges, creases).
"""
function has_consistent_normals(mesh::SimpleMesh)
    n = Meshes.nelements(mesh)
    n <= 1 && return true

    # Build edge → (triangle index, edge vertices) map in O(n)
    # When we encounter an edge again, check if it's traversed in opposite direction
    edge_map = Dict{Tuple{SVector{3,Float64},SVector{3,Float64}},
        Tuple{Int,SVector{3,Float64},SVector{3,Float64}}}()
    sizehint!(edge_map, 3 * n)  # Each triangle has 3 edges

    for i in 1:n
        v1, v2, v3 = _get_triangle_vertices(mesh, i)

        # Process each edge of triangle i
        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            key = _edge_key(va, vb)
            existing = get(edge_map, key, nothing)

            if existing !== nothing
                # Edge shared with another triangle - check orientation
                (_, other_va, other_vb) = existing

                # For correct manifold orientation, the edge should be traversed
                # in OPPOSITE directions: if this triangle goes va→vb,
                # the other should go vb→va
                same_direction = (va ≈ other_va && vb ≈ other_vb)

                if same_direction
                    # Both triangles traverse edge in SAME direction → FLIPPED FACE!
                    return false
                end
                # If opposite direction → correct orientation, continue
            else
                # First time seeing this edge - store triangle index and edge vertices
                edge_map[key] = (i, va, vb)
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
2. Create root octree covering mesh bounding box
3. Distribute triangles to boxes using SAT intersection tests
4. Recursively subdivide boxes exceeding triangle threshold
5. Balance octree to ensure 2:1 refinement constraint
6. (Optional) Classify empty leaves via BFS from boundary

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
    max_triangles_per_box::Int=50,
    classify_leaves::Bool=true,
    verify_orientation::Bool=true,
) where {M<:Manifold,C<:CRS}
    T = Float64
    h_min_val = T(ustrip(h_min))
    n_triangles = Meshes.nelements(mesh)

    if verify_orientation && !has_consistent_normals(mesh)
        @warn """
        Triangle mesh has orientation errors (flipped faces)!

        Some triangles have their faces oriented incorrectly (shared edges
        traversed in the same direction instead of opposite directions).
        This will cause incorrect isinside() and signed distance calculations.

        The mesh needs to be repaired before use with the octree.
        """
    end

    bbox_min, bbox_max = _compute_bbox(mesh)

    # Expand bounding box by 2% to avoid numerical precision issues
    # and ensure geometry is fully contained with buffer zone
    bbox_sz = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) * T(0.5)
    expansion_factor = T(1.02)  # 2% expansion
    expanded_half_size = (bbox_sz * T(0.5)) * expansion_factor
    bbox_min = bbox_center - expanded_half_size
    bbox_max = bbox_center + expanded_half_size

    root_size = maximum(bbox_max - bbox_min)

    estimated_boxes = max(1000, n_triangles * 2)
    tree = SpatialOctree{Int,T}(bbox_min, root_size; initial_capacity=estimated_boxes)

    root_elements = tree.element_lists[1]
    for tri_idx in 1:n_triangles
        push!(root_elements, tri_idx)
    end

    size_criterion = SizeCriterion(h_min_val)
    criterion = AndCriterion((MaxElementsCriterion(max_triangles_per_box), size_criterion))

    _subdivide_triangle_octree!(tree, mesh, 1, criterion)

    # Balance octree to ensure 2:1 refinement constraint
    balance_octree!(tree, size_criterion)

    # Classify leaves after balancing
    classification = if classify_leaves
        _classify_leaves(tree, mesh)
    else
        nothing
    end

    return TriangleOctree(tree, mesh, classification)
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
    _subdivide_triangle_octree!(tree, mesh, box_idx, criterion)

Recursively subdivide a box in the triangle octree.

For each box that needs subdivision:
1. Create 8 child boxes
2. For each triangle in parent, find which children it intersects
3. Distribute triangle indices to children
4. Recursively subdivide children if they meet criteria
"""
function _subdivide_triangle_octree!(
    tree::SpatialOctree{Int,T},
    mesh::SimpleMesh,
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
        v1, v2, v3 = _get_triangle_vertices(mesh, tri_idx)

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
        _subdivide_triangle_octree!(tree, mesh, child_idx, criterion)
    end
    return
end

"""
    _find_nearby_triangles_for_classification(
        tree::SpatialOctree{Int,T},
        leaf_idx::Int,
        search_radius::T
    ) -> Vector{Int}

Find all triangles within search_radius of a leaf's center.

Uses breadth-first search through octree neighbors to collect triangles from
leaves that intersect the search sphere. This is much faster than brute-force
search and naturally excludes far-side triangles in thin features.

# Arguments
- `tree`: The octree structure
- `leaf_idx`: Index of the leaf to search from
- `search_radius`: Radius to search (typically 2-3× leaf size)

# Returns
Vector of unique triangle indices within the search radius
"""
function _find_nearby_triangles_for_classification(
    tree::SpatialOctree{Int,T},
    leaf_idx::Int,
    search_radius::T,
) where {T<:Real}
    leaf_center = box_center(tree, leaf_idx)
    triangles = Set{Int}()
    visited = Set{Int}()
    to_visit = [leaf_idx]

    while !isempty(to_visit)
        current_idx = popfirst!(to_visit)
        current_idx ∈ visited && continue
        push!(visited, current_idx)

        # Check if this box intersects the search sphere
        box_min, box_max = box_bounds(tree, current_idx)
        closest_pt_in_box = clamp.(leaf_center, box_min, box_max)
        dist_to_box = norm(leaf_center - closest_pt_in_box)

        if dist_to_box > search_radius
            continue  # Box too far, skip it and its children
        end

        # Collect triangles from this leaf
        if is_leaf(tree, current_idx)
            for tri_idx in tree.element_lists[current_idx]
                push!(triangles, tri_idx)
            end
        end

        # Add neighbors to search queue (face neighbors only for efficiency)
        for direction in 1:6
            neighbor_indices = find_neighbor(tree, current_idx, direction)
            for neighbor_idx in neighbor_indices
                if neighbor_idx != 0 && neighbor_idx ∉ visited
                    push!(to_visit, neighbor_idx)
                end
            end
        end
    end

    return collect(triangles)
end

"""
    _classify_leaves(tree::SpatialOctree{Int,T}, mesh::SimpleMesh) -> Vector{Int8}

Classify octree leaves as exterior (0), boundary (1), or interior (2).

Uses **restricted radius search** to avoid thin-feature misclassification:
- For leaves near boundaries: only search triangles within 3× leaf size
- For leaves far from boundaries: use global search

This prevents grabbing triangles on the "far side" of thin features like bunny ears.

# Returns
- `classification::Vector{Int8}`: Classification for each box (0=exterior, 1=boundary, 2=interior)
"""
function _classify_leaves(tree::SpatialOctree{Int,T}, mesh::SimpleMesh) where {T<:Real}
    n_boxes = length(tree.element_lists)
    classification = zeros(Int8, n_boxes)

    for leaf_idx in all_leaves(tree)
        triangles_in_leaf = tree.element_lists[leaf_idx]

        if !isempty(triangles_in_leaf)
            # Leaf contains triangles → boundary
            classification[leaf_idx] = 1
        else
            # Empty leaf → classify based on signed distance
            leaf_center = box_center(tree, leaf_idx)
            leaf_size = box_size(tree, leaf_idx)

            # Strategy: Use restricted radius search for robustness
            # Search radius = 3× leaf size (enough to find nearby surface, but excludes far-side triangles)
            search_radius = T(3.0) * leaf_size
            nearby_triangles = _find_nearby_triangles_for_classification(
                tree, leaf_idx, search_radius
            )

            # Compute signed distance using nearby triangles if found
            if !isempty(nearby_triangles)
                signed_dist = _compute_signed_distance_local(leaf_center, mesh, nearby_triangles)
            else
                # Fallback to global search if no triangles nearby (far from surface)
                signed_dist = _compute_signed_distance(leaf_center, mesh)
            end

            # Classify based on signed distance
            leaf_radius = leaf_size * T(0.5)
            if abs(signed_dist) < leaf_radius
                classification[leaf_idx] = 1  # Boundary
            elseif signed_dist < 0
                classification[leaf_idx] = 2  # Interior
            else
                classification[leaf_idx] = 0  # Exterior
            end
        end
    end

    return classification
end

"""
    _compute_signed_distance(point::SVector{3,T}, mesh::SimpleMesh) -> T

Compute signed distance from a point to the closest triangle.

Returns:
- Positive if point is on the side of the normal (outside)
- Negative if point is on the opposite side of the normal (inside)
- Zero if point is exactly on the surface
"""
function _compute_signed_distance(point::SVector{3,T}, mesh::SimpleMesh) where {T<:Real}
    min_dist = T(Inf)
    n = Meshes.nelements(mesh)

    for i in 1:n
        v1, v2, v3 = _get_triangle_vertices(mesh, i)
        closest_pt = closest_point_on_triangle(point, v1, v2, v3)
        dist = norm(point - closest_pt)

        if dist < abs(min_dist)
            to_point = point - closest_pt
            normal_i = _get_triangle_normal(mesh, i)
            sign = dot(to_point, normal_i) >= 0 ? 1 : -1
            min_dist = sign * dist
        end
    end

    return min_dist
end

"""
    _compute_signed_distance_local(
        point::SVector{3,T},
        mesh::SimpleMesh,
        triangle_indices::Vector{Int}
    ) -> T

Compute signed distance from a point to the closest triangle in a local set.

This is the key to avoiding thin-feature misclassification: instead of searching
all M triangles globally, we only check a local subset (e.g., triangles within
3× leaf size). This excludes far-side triangles that would give wrong signs.

# Arguments
- `point`: Query point
- `mesh`: Triangle mesh
- `triangle_indices`: Subset of triangle indices to search

# Returns
Signed distance (positive = outside, negative = inside)
"""
function _compute_signed_distance_local(
    point::SVector{3,T},
    mesh::SimpleMesh,
    triangle_indices::Vector{Int},
) where {T<:Real}
    min_dist = T(Inf)

    for i in triangle_indices
        v1, v2, v3 = _get_triangle_vertices(mesh, i)
        closest_pt = closest_point_on_triangle(point, v1, v2, v3)
        dist = norm(point - closest_pt)

        if dist < abs(min_dist)
            to_point = point - closest_pt
            normal_i = _get_triangle_normal(mesh, i)
            sign = dot(to_point, normal_i) >= 0 ? 1 : -1
            min_dist = sign * dist
        end
    end

    return min_dist
end

Base.length(octree::TriangleOctree) = Meshes.nelements(octree.mesh)
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))
num_triangles(octree::TriangleOctree) = Meshes.nelements(octree.mesh)

#=============================================================================
Fast Spatial Queries (Layer 4)
=============================================================================#

"""
    isinside(point::SVector{3,T}, octree::TriangleOctree) -> Bool

Fast interior/exterior test using octree spatial index.

Returns `true` if point is inside the closed surface defined by the mesh.

# Performance
- Complexity: O(log M + k) where M = number of triangles, k ≈ 10-50
- Speedup: 100-1000× faster than brute-force O(M) approach

# Example
```julia
using WhatsThePoint, StaticArrays

octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)

point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)
```
"""
function isinside(point::SVector{3,T}, octree::TriangleOctree) where {T<:Real}
    isnothing(octree.leaf_classification) &&
        error("TriangleOctree must be built with classify_leaves=true")

    # Fast rejection: if point is outside bounding box, it's definitely outside
    bbox_min, bbox_max = box_bounds(octree.tree, 1)
    if any(point .< bbox_min) || any(point .> bbox_max)
        return false
    end

    leaf_idx = find_leaf(octree.tree, point)
    classification = octree.leaf_classification[leaf_idx]

    # Fast path for definite exterior
    classification == Int8(0) && return false

    # For interior leaves, verify the point is actually inside
    # (large leaves can span the boundary, so leaf classification alone is insufficient)
    if classification == Int8(2)
        # Check if point is near any boundary by searching nearby leaves for triangles
        # Use depth-2 neighbor search to find boundary triangles
        nearby_triangles = _collect_nearby_triangles(octree.tree, leaf_idx, 2)

        if !isempty(nearby_triangles)
            # Point is near boundary - recompute signed distance for accuracy
            signed_dist = _compute_local_signed_distance(point, octree.mesh, nearby_triangles)
            return signed_dist < 0
        else
            # Point is far from any boundary - trust interior classification
            return true
        end
    end

    # Boundary leaves - recompute signed distance
    triangles_in_leaf = octree.tree.element_lists[leaf_idx]

    if !isempty(triangles_in_leaf)
        signed_dist = _compute_local_signed_distance(point, octree.mesh, triangles_in_leaf)
        return signed_dist < 0
    else
        # Boundary leaf with no local triangles - search neighboring leaves
        # Since boundary neighbor refinement ensures uniform resolution near boundaries,
        # a shallow search (depth=2) should suffice for most cases
        nearby_triangles = _collect_nearby_triangles(octree.tree, leaf_idx, 2)

        if isempty(nearby_triangles)
            error("""
                Cannot compute signed distance: no triangles found in leaf or neighbors.
                Point: $point, Leaf: $leaf_idx

                This can occur in edge cases with very coarse meshes or complex geometry.
                Try reducing h_min for finer octree resolution.
                """)
        end

        signed_dist = _compute_local_signed_distance(point, octree.mesh, nearby_triangles)
        return signed_dist < 0
    end
end

"""
    _compute_local_signed_distance(
        point::SVector{3,T},
        mesh::SimpleMesh,
        tri_indices::Vector{Int}
    ) -> T

Compute signed distance from point to nearest triangle in the local set.

This is the key performance optimization: instead of checking all M triangles,
we only check the k≈10-50 triangles in the point's octree leaf.

Performance: Fused computation avoids redundant closest_point calculation.
"""
function _compute_local_signed_distance(
    point::SVector{3,T},
    mesh::SimpleMesh,
    tri_indices::Vector{Int},
) where {T<:Real}
    min_dist_sq = typemax(T)
    closest_idx = 0
    closest_pt = point

    @inbounds for tri_idx in tri_indices
        v1, v2, v3 = _get_triangle_vertices(mesh, tri_idx)
        cp = closest_point_on_triangle(point, v1, v2, v3)
        diff = point - cp
        dist_sq = dot(diff, diff)

        if dist_sq < min_dist_sq
            min_dist_sq = dist_sq
            closest_idx = tri_idx
            closest_pt = cp
        end
    end

    to_point = point - closest_pt
    normal = _get_triangle_normal(mesh, closest_idx)
    sign = dot(to_point, normal) < 0 ? -1 : 1

    return sign * sqrt(min_dist_sq)
end

"""
    _compute_signed_distance_to_triangle(
        point::SVector{3,T},
        mesh::SimpleMesh,
        tri_idx::Int
    ) -> T

Compute signed distance from point to a single triangle.
"""
function _compute_signed_distance_to_triangle(
    point::SVector{3,T},
    mesh::SimpleMesh,
    tri_idx::Int,
) where {T<:Real}
    v1, v2, v3 = _get_triangle_vertices(mesh, tri_idx)
    closest_pt = closest_point_on_triangle(point, v1, v2, v3)

    to_point = point - closest_pt
    normal = _get_triangle_normal(mesh, tri_idx)
    sign = dot(to_point, normal) < 0 ? -1 : 1

    return sign * norm(to_point)
end

"""
    _collect_nearby_triangles(
        tree::SpatialOctree,
        leaf_idx::Int,
        max_depth::Int=2
    ) -> Vector{Int}

Collect all triangles from the given leaf and its neighbors up to max_depth levels.

This function searches the leaf and neighboring leaves to build a local set of
triangles for accurate signed distance computation near boundaries.

# Arguments
- `tree`: The spatial octree structure
- `leaf_idx`: Index of the leaf to start from
- `max_depth`: How many neighbor levels to search (1=face neighbors, 2=face+edge+corner)

# Returns
Vector of unique triangle indices from nearby leaves
"""
function _collect_nearby_triangles(
    tree::SpatialOctree{Int,T},
    leaf_idx::Int,
    max_depth::Int=2,
) where {T<:Real}
    triangles = Set{Int}()
    visited = Set{Int}()
    to_visit = [(leaf_idx, 0)]  # (leaf_idx, depth)

    while !isempty(to_visit)
        current_idx, depth = popfirst!(to_visit)

        current_idx ∈ visited && continue
        push!(visited, current_idx)

        # Only collect triangles from leaf nodes (internal nodes may have stale triangle lists)
        if is_leaf(tree, current_idx)
            for tri_idx in tree.element_lists[current_idx]
                push!(triangles, tri_idx)
            end
        end

        # If not at max depth, add neighbors
        if depth < max_depth
            # Check all 6 face neighbors (±x, ±y, ±z)
            for direction in 1:6
                neighbor_indices = find_neighbor(tree, current_idx, direction)
                for neighbor_idx in neighbor_indices
                    if neighbor_idx != 0 && neighbor_idx ∉ visited
                        push!(to_visit, (neighbor_idx, depth + 1))
                    end
                end
            end
        end
    end

    return collect(triangles)
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
