# Spacing-driven tree subdivision criterion and node tree construction
#
# This file provides infrastructure for building node trees (octree in 3D,
# quadtree in 2D) that subdivide based on a prescribed spacing function,
# enabling spacing-aware point generation.

"""
    SpacingCriterion{T<:Real, S} <: SubdivisionCriterion

Tree subdivision criterion based on local spacing requirements.

Subdivides boxes where `h_box > alpha * h_spacing(center)`, ensuring the tree
resolution is fine enough to properly represent the spacing function.

# Fields
- `spacing::S`: Spacing function object
- `alpha::T`: Subdivision aggressiveness factor
- `absolute_min::T`: Absolute minimum box size (prevents infinite subdivision)

# Algorithm
For each box:
1. Query `h_local = spacing(box_center)`
2. If `h_box > alpha * h_local`, subdivide
3. Stop if `h_box ≤ absolute_min`

Smaller `alpha` values create finer trees (more aggressive subdivision).
"""
struct SpacingCriterion{T <: Real, S} <: SubdivisionCriterion
    spacing::S
    alpha::T
    absolute_min::T
end

function SpacingCriterion(spacing, diagonal::Real; alpha = 2, min_ratio = 1.0e-6)
    T = typeof(float(diagonal))
    return SpacingCriterion{T, typeof(spacing)}(spacing, T(alpha), T(diagonal) * T(min_ratio))
end

@inline function _spacing_value(::Type{T}, spacing, p::SVector{N, T}) where {N, T}
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
# Node tree construction
# ============================================================================

"""
    build_node_octree(geometry, spacing, alpha, node_min_ratio)

Build a spacing-driven node tree from an existing geometry index.

Creates a new `SpatialTree` that subdivides based on a spacing function,
enabling spacing-aware point distribution. The node tree is:
1. Recursively subdivided using `SpacingCriterion`
2. Balanced to maintain 2:1 refinement ratio
3. Independent of the geometry-index resolution

# Arguments
- `geometry`: Geometry index (`TriangleOctree` in 3D, `SegmentQuadtree` in 2D)
- `spacing`: Spacing function (e.g., `ConstantSpacing`, `BoundaryLayerSpacing`)
- `alpha`: Subdivision aggressiveness (`h_box ≤ alpha * h_spacing`)
- `node_min_ratio`: Minimum box size ratio relative to domain

# Returns
`SpatialTree{N, Int, T}` with spacing-driven subdivision, where `T` is the
geometry index's coordinate type (the source CRS machine type)

# Example
```julia
tri_octree = TriangleOctree(mesh; classify_leaves=true)
spacing = BoundaryLayerSpacing(points; at_wall=0.5m, bulk=5m, layer_thickness=2m)
node_tree = build_node_octree(tri_octree, spacing, 1.0, 1e-6)
```
"""
function build_node_octree(
        geometry::AbstractGeometryIndex, spacing, alpha, node_min_ratio
    )
    geo_tree = geometry_tree(geometry)
    bbox_min, bbox_max = bounding_box(geo_tree)
    node_tree = _node_tree_like(geo_tree)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(spacing, diagonal; alpha, min_ratio = node_min_ratio)

    _subdivide_node_octree!(node_tree, 1, criterion, geometry)
    balance_octree!(node_tree, criterion)

    return node_tree
end

"Empty node tree spanning the same root box as the geometry tree."
_node_tree_like(t::SpatialTree{N, <:Any, T}) where {N, T} =
    SpatialTree{N, Int, T}(t.origin, t.root_size; initial_capacity = 1000)

function _box_may_contain_interior(
        node_tree::SpatialTree{N, <:Any, T}, box_idx, geometry
    ) where {N, T}
    bbox_min, bbox_max = box_bounds(node_tree, box_idx)
    h = box_size(node_tree, box_idx)
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    for pt in _box_probe_points(bbox_min, bbox_max)
        classify_point(geometry, pt, tol) != LEAF_EXTERIOR && return true
    end

    # Probe sampling still misses thin/elongated domains inside large boxes
    # when every sample falls outside the geometry. Fall back to a spatial
    # descent of the geometry tree for O(log L) overlap detection.
    geo_cls = leaf_classes(geometry)
    predicate = if isnothing(geo_cls)
        _ -> true
    else
        leaf_idx -> geo_cls[leaf_idx] != LEAF_EXTERIOR
    end
    return any_leaf_overlapping(geometry_tree(geometry), bbox_min, bbox_max, predicate)
end

"""
    _box_probe_points(bbox_min, bbox_max)

Deterministic point probes covering a box: center, corners, and midpoints of
faces/edges — 27 points in 3D, 9 in 2D (center + 4 corners + 4 edge midpoints).
"""
@inline function _box_probe_points(bbox_min::SVector{3, T}, bbox_max::SVector{3, T}) where {T}
    center = (bbox_min + bbox_max) / 2
    corners = _box_corners(bbox_min, bbox_max)
    faces = _box_face_centers(bbox_min, bbox_max)
    edges = _box_edge_midpoints(bbox_min, bbox_max)
    return (center, corners..., faces..., edges...)
end

@inline function _box_probe_points(bbox_min::SVector{2, T}, bbox_max::SVector{2, T}) where {T}
    cx = (bbox_min[1] + bbox_max[1]) / 2
    cy = (bbox_min[2] + bbox_max[2]) / 2
    return (
        SVector{2, T}(cx, cy),
        _box_corners(bbox_min, bbox_max)...,
        SVector{2, T}(cx, bbox_min[2]),
        SVector{2, T}(cx, bbox_max[2]),
        SVector{2, T}(bbox_min[1], cy),
        SVector{2, T}(bbox_max[1], cy),
    )
end

@inline function _box_face_centers(bbox_min::SVector{3, T}, bbox_max::SVector{3, T}) where {T}
    cx = (bbox_min[1] + bbox_max[1]) / 2
    cy = (bbox_min[2] + bbox_max[2]) / 2
    cz = (bbox_min[3] + bbox_max[3]) / 2
    return (
        SVector{3, T}(bbox_min[1], cy, cz),
        SVector{3, T}(bbox_max[1], cy, cz),
        SVector{3, T}(cx, bbox_min[2], cz),
        SVector{3, T}(cx, bbox_max[2], cz),
        SVector{3, T}(cx, cy, bbox_min[3]),
        SVector{3, T}(cx, cy, bbox_max[3]),
    )
end

@inline function _box_edge_midpoints(bbox_min::SVector{3, T}, bbox_max::SVector{3, T}) where {T}
    x0, x1 = bbox_min[1], bbox_max[1]
    y0, y1 = bbox_min[2], bbox_max[2]
    z0, z1 = bbox_min[3], bbox_max[3]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    cz = (z0 + z1) / 2
    return (
        SVector{3, T}(cx, y0, z0), SVector{3, T}(cx, y1, z0),
        SVector{3, T}(cx, y0, z1), SVector{3, T}(cx, y1, z1),
        SVector{3, T}(x0, cy, z0), SVector{3, T}(x1, cy, z0),
        SVector{3, T}(x0, cy, z1), SVector{3, T}(x1, cy, z1),
        SVector{3, T}(x0, y0, cz), SVector{3, T}(x1, y0, cz),
        SVector{3, T}(x0, y1, cz), SVector{3, T}(x1, y1, cz),
    )
end

function _subdivide_node_octree!(node_tree, box_idx, criterion, geometry)
    should_subdivide(criterion, node_tree, box_idx) || return
    _box_may_contain_interior(node_tree, box_idx, geometry) || return

    subdivide!(node_tree, box_idx)
    for child_idx in children(node_tree, box_idx)
        _subdivide_node_octree!(node_tree, child_idx, criterion, geometry)
    end
    return
end

"""
    classify_node_octree(node_tree, geometry)

Classify node tree leaves as interior, boundary, or exterior using the
geometry index's `classify_point` seam method. Correctness of downstream
sampling (skipping `isinside` on `LEAF_INTERIOR` points) relies on the
geometry-bbox early return inside `classify_point`, which prevents sign-vote
flips from promoting far-exterior leaves into `LEAF_INTERIOR`.

# Returns
Vector of `Int8` classifications indexed by node-tree box index.
"""
function classify_node_octree(node_tree, geometry)
    query(pt, tol) = classify_point(geometry, pt, tol)
    return classify_leaves!(node_tree, query)
end
