#=============================================================================
TYPES AND DATA STRUCTURES
=============================================================================#

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

#=============================================================================
UTILITIES
=============================================================================#

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

#=============================================================================
CONSTRUCTION (BUILD-TIME ONLY)

Ray casting is used here to classify original leaves before balancing.
Performance is acceptable because this happens once during octree build.
The implementation uses AABB-ray intersection to filter candidate triangles
from ~70k to ~200 per ray.
=============================================================================#

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

    # Classify leaves BEFORE balancing (only the original leaves from subdivision)
    pre_balance_classification = if classify_leaves
        _classify_leaves(tree, mesh)
    else
        nothing
    end

    # Balance octree to ensure 2:1 refinement constraint
    # This will subdivide some leaves, creating new empty leaves
    balance_octree!(tree, size_criterion)

    # After balancing: propagate classification from parent to new children
    # This avoids expensive ray casting for the 23k+ new empty leaves!
    classification = if classify_leaves
        _propagate_classification_after_balance(tree, pre_balance_classification)
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
    _collect_triangles_along_ray!(
        candidate_triangles::Set{Int},
        tree::SpatialOctree{Int,T},
        point::SVector{3,T},
        ray_direction::SVector{3,T}
    ) where {T<:Real} -> Nothing

Collect candidate triangles along a ray path using octree spatial acceleration.
Mutates the `candidate_triangles` set to avoid allocations.

Simplified for construction-time use (called once during octree build).
"""
function _collect_triangles_along_ray!(
        candidate_triangles::Set{Int},
        tree::SpatialOctree{Int, T},
        point::SVector{3, T},
        ray_direction::SVector{3, T},
    ) where {T <: Real}
    empty!(candidate_triangles)

    # Check each leaf box for ray intersection
    @inbounds for leaf_idx in all_leaves(tree)
        bbox_min, bbox_max = box_bounds(tree, leaf_idx)

        # Fast AABB-ray intersection test using slab method
        t_min = typemin(T)
        t_max = typemax(T)

        # Check each axis
        for axis in 1:3
            if abs(ray_direction[axis]) > T(1e-10)
                t1 = (bbox_min[axis] - point[axis]) / ray_direction[axis]
                t2 = (bbox_max[axis] - point[axis]) / ray_direction[axis]

                if t1 > t2
                    t1, t2 = t2, t1
                end

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max
                    @goto next_box
                end
            else
                # Ray parallel to axis - check bounds
                if point[axis] < bbox_min[axis] || point[axis] > bbox_max[axis]
                    @goto next_box
                end
            end
        end

        # Ray intersects box if t_max >= 0
        if t_max >= 0
            for tri_idx in tree.element_lists[leaf_idx]
                push!(candidate_triangles, tri_idx)
            end
        end

        @label next_box
    end

    return nothing
end

"""
    _classify_point_ray_casting(
        point::SVector{3,T},
        mesh::SimpleMesh,
        tree::SpatialOctree{Int,T}
    ) where {T<:Real} -> Bool

Classify a point as inside (true) or outside (false) using ray casting.

Simplified for construction-time use (called once during octree build).
Uses ray casting for robust classification that handles thin features,
high curvature, and concave geometry.

# Algorithm
1. Cast ray from point in +X direction
2. Collect candidate triangles along ray using octree acceleration
3. Count ray-triangle intersections: odd = inside, even = outside
"""
function _classify_point_ray_casting(
        point::SVector{3, T},
        mesh::SimpleMesh,
        tree::SpatialOctree{Int, T},
    ) where {T <: Real}
    # Ray direction (along +X axis for simplicity)
    ray_direction = SVector{3, T}(1, 0, 0)

    # Collect candidate triangles using octree acceleration
    candidate_triangles = Set{Int}()
    _collect_triangles_along_ray!(candidate_triangles, tree, point, ray_direction)

    # Count ray-triangle intersections
    intersection_count = 0

    @inbounds for tri_idx in candidate_triangles
        v1, v2, v3 = _get_triangle_vertices(mesh, tri_idx)
        t = ray_triangle_intersection(point, ray_direction, v1, v2, v3)

        if t !== nothing
            intersection_count += 1
        end
    end

    # Odd count = inside, even count = outside
    return isodd(intersection_count)
end

"""
    _propagate_classification_after_balance(
        tree::SpatialOctree{Int,T},
        pre_balance_classification::Vector{Int8}
    ) -> Vector{Int8}

After octree balancing, propagate classification from parents to newly created children.

# Key Insight
When balancing subdivides a leaf, new empty children inherit the parent's classification:
- Interior parent → interior children
- Exterior parent → exterior children
- Only leaves that GAIN triangles become boundary leaves

This simple inheritance rule eliminates expensive ray casting for ~23k new empty leaves!

# Algorithm
For each new leaf:
- Contains triangles → boundary (1)
- Empty → inherit parent classification (0 or 2)
"""
function _propagate_classification_after_balance(
        tree::SpatialOctree{Int, T},
        pre_balance_classification::Vector{Int8},
    ) where {T <: Real}
    n_boxes = length(tree.element_lists)
    classification = zeros(Int8, n_boxes)

    # Copy pre-balance classifications
    n_pre_balance = length(pre_balance_classification)
    classification[1:n_pre_balance] .= pre_balance_classification

    # Classify new leaves created during balancing
    for leaf_idx in all_leaves(tree)
        if leaf_idx <= n_pre_balance
            # Old leaf - already classified
            continue
        end

        triangles_in_leaf = tree.element_lists[leaf_idx]

        if !isempty(triangles_in_leaf)
            # Gained triangles → boundary
            classification[leaf_idx] = 1
        else
            # Empty leaf → inherit from parent
            parent_idx = tree.parent[leaf_idx]
            classification[leaf_idx] = pre_balance_classification[parent_idx]
        end
    end

    return classification
end

"""
    _classify_leaves(tree::SpatialOctree{Int,T}, mesh::SimpleMesh) -> Vector{Int8}

Classify octree leaves as exterior (0), boundary (1), or interior (2).

Uses ray casting for robust classification that handles thin features,
high curvature, and concave geometry.

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
            # Empty leaf → classify using ray casting
            leaf_center = box_center(tree, leaf_idx)
            is_inside = _classify_point_ray_casting(leaf_center, mesh, tree)

            classification[leaf_idx] = is_inside ? Int8(2) : Int8(0)
        end
    end

    return classification
end



Base.length(octree::TriangleOctree) = Meshes.nelements(octree.mesh)
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))
num_triangles(octree::TriangleOctree) = Meshes.nelements(octree.mesh)

#=============================================================================
QUERIES (RUN-TIME, PERFORMANCE CRITICAL)

isinside() is called thousands of times during discretization.
CRITICAL: NO ray casting here! Only cached classification + local signed distance.

Performance:
- Exterior/interior leaves: O(1) - single array lookup
- Boundary leaves: O(k) where k ≈ 10-50 triangles in leaf
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

    # Use leaf classification for clear cases
    classification == Int8(0) && return false  # Exterior
    classification == Int8(2) && return true   # Interior

    # For boundary leaves (classification == 1), compute signed distance
    # This is fast because boundary leaves contain only ~10-50 triangles
    triangles_in_leaf = octree.tree.element_lists[leaf_idx]
    if !isempty(triangles_in_leaf)
        signed_dist = _compute_local_signed_distance(point, octree.mesh, triangles_in_leaf)
        return signed_dist < 0
    end

    # Shouldn't reach here, but default to false
    return false
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
