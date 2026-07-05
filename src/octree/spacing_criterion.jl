# Spacing-driven octree subdivision criterion and node octree construction
#
# This file provides infrastructure for building octrees that subdivide based on
# a prescribed spacing function, enabling spacing-aware point generation.

"""
    SpacingCriterion{T<:Real, F} <: SubdivisionCriterion

Octree subdivision criterion based on local spacing requirements.

Subdivides boxes where `h_box > alpha * h(center)`, ensuring the octree
resolution is fine enough to properly represent the spacing function.

# Fields
- `h::F`: Numerical spacing function `(SVector{3, Float64}) -> Float64` (wrap
  an [`AbstractSpacing`](@ref) via [`numerical_spacing`](@ref))
- `alpha::T`: Subdivision aggressiveness factor
- `absolute_min::T`: Absolute minimum box size (prevents infinite subdivision)

# Algorithm
For each octree box:
1. Query `h_local = h(box_center)`
2. If `h_box > alpha * h_local`, subdivide
3. Stop if `h_box ≤ absolute_min`

Smaller `alpha` values create finer octrees (more aggressive subdivision).
"""
struct SpacingCriterion{T <: Real, F} <: SubdivisionCriterion
    h::F
    alpha::T
    absolute_min::T
end

function SpacingCriterion(h, diagonal::Real; alpha = 2, min_ratio = 1.0e-6)
    T = typeof(float(diagonal))
    return SpacingCriterion{T, typeof(h)}(h, T(alpha), T(diagonal) * T(min_ratio))
end

function should_subdivide(c::SpacingCriterion{T}, tree, box_idx) where {T}
    h_box = box_size(tree, box_idx)
    h_box <= c.absolute_min && return false

    center = box_center(tree, box_idx)
    h_local = c.h(center)
    h_local <= eps(T) && return false

    return h_box > c.alpha * h_local
end

can_subdivide(c::SpacingCriterion, tree, idx) = box_size(tree, idx) > c.absolute_min

# ============================================================================
# Node octree construction
# ============================================================================

"""
    build_node_octree(triangle_octree, h, alpha, node_min_ratio)

Build a spacing-driven node octree from an existing triangle octree.

Creates a new `SpatialOctree` that subdivides based on a spacing function,
enabling spacing-aware point distribution. The node octree is:
1. Recursively subdivided using `SpacingCriterion`
2. Balanced to maintain 2:1 refinement ratio
3. Independent of the triangle octree resolution

# Arguments
- `triangle_octree`: Base `TriangleOctree` for geometry
- `h`: Numerical spacing function `(SVector{3, Float64}) -> Float64` (wrap an
  `AbstractSpacing` via [`numerical_spacing`](@ref))
- `alpha`: Subdivision aggressiveness (`h_box ≤ alpha * h`)
- `node_min_ratio`: Minimum box size ratio relative to domain

# Returns
`SpatialOctree{Int, T}` with spacing-driven subdivision, where `T` is the
triangle octree's coordinate type (the mesh CRS machine type)

# Example
```julia
tri_octree = TriangleOctree(mesh; classify_leaves=true)
spacing = BoundaryLayerSpacing(points; at_wall=0.5m, bulk=5m, layer_thickness=2m)
node_tree = build_node_octree(tri_octree, numerical_spacing(spacing, m), 1.0, 1e-6)
```
"""
function build_node_octree(
        triangle_octree::TriangleOctree{M, C, T}, h, alpha, node_min_ratio
    ) where {M, C, T}
    bbox_min, bbox_max = bounding_box(triangle_octree.tree)
    node_tree = SpatialOctree{Int, T}(bbox_min, triangle_octree.tree.root_size; initial_capacity = 1000)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(h, diagonal; alpha, min_ratio = node_min_ratio)

    _subdivide_node_octree!(node_tree, 1, criterion, triangle_octree)
    balance_octree!(node_tree, criterion)

    return node_tree
end

@inline function _mesh_geometry_query(pt::SVector{3, T}, tol, octree) where {T}
    return _classify_point_octree(pt, octree; tol = T(tol))
end

function _box_may_contain_interior(
        node_tree::SpatialOctree{<:Any, T}, box_idx, triangle_octree
    ) where {T}
    bbox_min, bbox_max = box_bounds(node_tree, box_idx)
    h = box_size(node_tree, box_idx)
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    center = box_center(node_tree, box_idx)
    corners = _box_corners(bbox_min, bbox_max)
    faces = _box_face_centers(bbox_min, bbox_max)
    edges = _box_edge_midpoints(bbox_min, bbox_max)

    for pt in (center, corners..., faces..., edges...)
        _mesh_geometry_query(pt, tol, triangle_octree) != LEAF_EXTERIOR && return true
    end

    # 27-point sampling still misses thin/elongated domains inside large cubic
    # boxes when every sample falls outside the geometry. Fall back to a spatial
    # descent of the triangle octree for O(log L) overlap detection.
    tri_cls = triangle_octree.leaf_classification
    predicate = if isnothing(tri_cls)
        _ -> true
    else
        leaf_idx -> tri_cls[leaf_idx] != LEAF_EXTERIOR
    end
    return any_leaf_overlapping(triangle_octree.tree, bbox_min, bbox_max, predicate)
end

function _subdivide_node_octree!(node_tree, box_idx, criterion, triangle_octree)
    should_subdivide(criterion, node_tree, box_idx) || return
    _box_may_contain_interior(node_tree, box_idx, triangle_octree) || return

    subdivide!(node_tree, box_idx)
    for child_idx in node_tree.children[box_idx]
        child_idx == 0 && continue
        _subdivide_node_octree!(node_tree, child_idx, criterion, triangle_octree)
    end
    return
end

"""
    classify_node_octree(node_tree, triangle_octree)

Classify node octree leaves as interior, boundary, or exterior using the
triangle octree's geometry query. Correctness of downstream sampling
(skipping `isinside` on `LEAF_INTERIOR` points) relies on the mesh-bbox
early return inside `_mesh_geometry_query`, which prevents sign-vote flips
from promoting far-exterior leaves into `LEAF_INTERIOR`.

# Returns
Vector of `Int8` classifications indexed by node-octree box index.
"""
function classify_node_octree(node_tree, triangle_octree)
    query(pt, tol) = _mesh_geometry_query(pt, tol, triangle_octree)
    return classify_leaves!(node_tree, query)
end
