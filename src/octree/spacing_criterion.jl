# Spacing-driven octree subdivision criterion and node octree construction
#
# This file provides infrastructure for building octrees that subdivide based on
# a prescribed spacing function, enabling spacing-aware point generation.

"""
    SpacingCriterion{T<:Real, S} <: SubdivisionCriterion

Octree subdivision criterion based on local spacing requirements.

Subdivides boxes where `h_box > alpha * h_spacing(center)`, ensuring the octree
resolution is fine enough to properly represent the spacing function.

# Fields
- `spacing::S`: Spacing function object
- `alpha::T`: Subdivision aggressiveness factor
- `absolute_min::T`: Absolute minimum box size (prevents infinite subdivision)

# Algorithm
For each octree box:
1. Query `h_local = spacing(box_center)`
2. If `h_box > alpha * h_local`, subdivide
3. Stop if `h_box ≤ absolute_min`

Smaller `alpha` values create finer octrees (more aggressive subdivision).
"""
struct SpacingCriterion{T <: Real, S} <: SubdivisionCriterion
    spacing::S
    alpha::T
    absolute_min::T
end

function SpacingCriterion(spacing, diagonal; alpha = 2.0, min_ratio = 1.0e-6)
    T = Float64
    return SpacingCriterion{T, typeof(spacing)}(spacing, T(alpha), T(diagonal) * T(min_ratio))
end

@inline function _spacing_value(::Type{T}, spacing, p::SVector{3, T}) where {T}
    return T(ustrip(spacing(Point(p...))))
end

function should_subdivide(c::SpacingCriterion{T}, tree, box_idx) where {T}
    h_box = box_size(tree, box_idx)
    h_box <= c.absolute_min && return false

    center = box_center(tree, box_idx)
    h_local = _spacing_value(T, c.spacing, center)
    h_local <= eps(T) && return false

    return h_box > c.alpha * h_local
end

can_subdivide(c::SpacingCriterion, tree, idx) = box_size(tree, idx) > c.absolute_min

# ============================================================================
# Node octree construction
# ============================================================================

"""
    build_node_octree(triangle_octree, spacing, alpha, node_min_ratio)

Build a spacing-driven node octree from an existing triangle octree.

Creates a new `SpatialOctree` that subdivides based on a spacing function,
enabling spacing-aware point distribution. The node octree is:
1. Recursively subdivided using `SpacingCriterion`
2. Balanced to maintain 2:1 refinement ratio
3. Independent of the triangle octree resolution

# Arguments
- `triangle_octree`: Base `TriangleOctree` for geometry
- `spacing`: Spacing function (e.g., `ConstantSpacing`, `BoundaryLayerSpacing`)
- `alpha`: Subdivision aggressiveness (`h_box ≤ alpha * h_spacing`)
- `node_min_ratio`: Minimum box size ratio relative to domain

# Returns
`SpatialOctree{Int, Float64}` with spacing-driven subdivision

# Example
```julia
tri_octree = TriangleOctree(mesh; classify_leaves=true)
spacing = BoundaryLayerSpacing(points; at_wall=0.5m, bulk=5m, layer_thickness=2m)
node_tree = build_node_octree(tri_octree, spacing, 1.0, 1e-6)
```
"""
function build_node_octree(triangle_octree, spacing, alpha, node_min_ratio)
    T = Float64
    bbox_min, bbox_max = bounding_box(triangle_octree.tree)
    node_tree = SpatialOctree{Int, T}(bbox_min, triangle_octree.tree.root_size; initial_capacity = 1000)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(spacing, diagonal; alpha, min_ratio = node_min_ratio)

    _subdivide_node_octree!(node_tree, 1, criterion, triangle_octree)
    balance_octree!(node_tree, criterion)

    return node_tree
end

@inline function _mesh_geometry_query(pt::SVector{3, T}, tol, octree) where {T}
    # Sign voting inside `_compute_signed_distance_octree` can flip for points
    # far outside the triangle octree's cubic root, producing negative signed
    # distances (and therefore LEAF_INTERIOR) above regions of space that are
    # obviously exterior to the mesh. The mesh bbox is the authoritative
    # envelope — anything strictly outside it (by more than the classification
    # tolerance) is exterior, matching the fast-path used by `isinside`.
    if any(pt .< octree.mesh_bbox_min .- tol) || any(pt .> octree.mesh_bbox_max .+ tol)
        return LEAF_EXTERIOR
    end
    sd = _compute_signed_distance_octree(pt, octree.mesh, octree.tree)
    return _leaf_class_from_signed_distance(sd, tol)
end

function _box_may_contain_interior(node_tree, box_idx, triangle_octree)
    T = Float64
    bbox_min, bbox_max = box_bounds(node_tree, box_idx)
    h = box_size(node_tree, box_idx)
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    center = box_center(node_tree, box_idx)
    corners = _box_corners(bbox_min, bbox_max)

    for pt in (center, corners...)
        _mesh_geometry_query(pt, tol, triangle_octree) != LEAF_EXTERIOR && return true
    end

    # Point sampling can miss thin/elongated domains inside large cubic boxes.
    # Fall back to checking whether any non-exterior triangle-octree leaf overlaps this box.
    tri_tree = triangle_octree.tree
    tri_cls = triangle_octree.leaf_classification
    if !isnothing(tri_cls)
        for leaf_idx in all_leaves(tri_tree)
            tri_cls[leaf_idx] == LEAF_EXTERIOR && continue
            leaf_min, leaf_max = box_bounds(tri_tree, leaf_idx)
            if _boxes_overlap(bbox_min, bbox_max, leaf_min, leaf_max)
                return true
            end
        end
    end

    return false
end

@inline function _boxes_overlap(a_min, a_max, b_min, b_max)
    @inbounds for d in 1:3
        a_min[d] > b_max[d] && return false
        a_max[d] < b_min[d] && return false
    end
    return true
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
