# SegmentQuadtree — the 2D geometry index (`AbstractGeometryIndex{𝔼{2}}`).
#
# The 2D counterpart of `TriangleOctree`: a quadtree (`SpatialTree{2,Int,T}`)
# over the segments of one or more closed boundary loops, answering the same
# seam contract (`domain_bounds`, `classify_point`, `isinside`) through signed
# distance to the nearest segment, signed by the feature pseudonormal — the 2D
# specialization of Bærentzen & Aanæs (2005), where a vertex pseudonormal is
# simply the sum of its two adjacent segment normals.
#
# Everything structural is shared with the 3D index: the tree container,
# the branch-and-bound nearest-element descent, `VertexResolutionCriterion`,
# the subdivision/redistribution recursion, and leaf classification all come
# from `spatial_octree.jl` / `triangle_octree.jl` generics — this file only
# supplies the segment-specific geometry kernels.

"""
    SegmentIndex{T}

The package's runtime representation of a 2D boundary: one or more closed
polyline loops, indexed, unit-stripped and machine-typed — the 2D counterpart
of [`TriangleIndex`](@ref).

Loops are oriented at construction so that segment normals point out of the
domain: outer loops counter-clockwise, hole loops clockwise (nesting parity
decides which is which — a loop contained in an odd number of other loops is a
hole). Callers may pass loops in any orientation.

Fields:
- `vertices`/`segments`: indexed representation (unique coords + per-segment
  vertex indices; segment `i` runs `vertices[segments[i][1]] →
  vertices[segments[i][2]]`).
- `normal`: precomputed unit outward segment normals.
- `vertex_normal`: per-vertex pseudonormal (sum of the two adjacent segment
  normals) — the sign-exact feature normal for signed-distance queries.
- `bbox_min`/`bbox_max`: boundary bounding box.
- `len_unit`: the unit stripped from coordinates, re-attached at exits.
"""
struct SegmentIndex{T <: Real}
    vertices::Vector{SVector{2, T}}
    segments::Vector{NTuple{2, Int32}}
    normal::Vector{SVector{2, T}}
    vertex_normal::Vector{SVector{2, T}}
    bbox_min::SVector{2, T}
    bbox_max::SVector{2, T}
    len_unit::Unitful.Units
end

num_segments(index::SegmentIndex) = length(index.segments)

@inline function _get_segment_vertices(index::SegmentIndex, seg_idx::Int)
    s = @inbounds index.segments[seg_idx]
    @inbounds return index.vertices[s[1]], index.vertices[s[2]]
end

# Geometry-index seam for `VertexResolutionCriterion` (see `should_subdivide`).
@inline _element_vertices(index::SegmentIndex, el_idx::Int) =
    _get_segment_vertices(index, el_idx)

_compute_bbox(index::SegmentIndex{T}) where {T} = (index.bbox_min, index.bbox_max)

"Shoelace signed area of a closed loop (positive = counter-clockwise)."
function _loop_signed_area(loop::Vector{SVector{2, T}}) where {T}
    sa = zero(T)
    n = length(loop)
    @inbounds for i in 1:n
        a, b = loop[i], loop[mod1(i + 1, n)]
        sa += a[1] * b[2] - b[1] * a[2]
    end
    return sa / 2
end

"Even-odd crossing test of `p` against a closed loop (loop vertices ordered)."
function _point_in_loop(p::SVector{2, T}, loop::Vector{SVector{2, T}}) where {T}
    c = false
    n = length(loop)
    @inbounds for i in 1:n
        a, b = loop[i], loop[mod1(i + 1, n)]
        ((a[2] > p[2]) != (b[2] > p[2])) || continue
        x = a[1] + (p[2] - a[2]) / (b[2] - a[2]) * (b[1] - a[1])
        x > p[1] && (c = !c)
    end
    return c
end

"""
    SegmentIndex(T, loops, len_unit)

Build a `SegmentIndex` from closed loops of stripped 2D vertices. Validates
each loop (≥ 3 vertices, non-degenerate signed area) and normalizes loop
orientation by nesting parity so normals point out of the domain.
"""
function SegmentIndex(
        ::Type{T},
        loops::Vector{Vector{SVector{2, T}}},
        len_unit::Unitful.Units,
    ) where {T <: Real}
    isempty(loops) && throw(ArgumentError("SegmentIndex requires at least one boundary loop"))

    # Validate and orient: outer loops CCW, holes CW (by nesting parity).
    oriented = Vector{Vector{SVector{2, T}}}(undef, length(loops))
    for (li, loop) in enumerate(loops)
        n = length(loop)
        n >= 3 || throw(ArgumentError("boundary loop $li has $n vertices; a closed loop needs at least 3"))
        sa = _loop_signed_area(loop)
        lo = reduce((a, b) -> min.(a, b), loop)
        hi = reduce((a, b) -> max.(a, b), loop)
        bbox_area = (hi[1] - lo[1]) * (hi[2] - lo[2])
        abs(sa) < T(1.0e-10) * max(bbox_area, eps(T)) && throw(
            ArgumentError(
                "boundary loop $li has (near-)zero signed area — its vertices are " *
                    "collinear or not ordered sequentially around the loop"
            )
        )
        depth = count(
            lj -> lj != li && _point_in_loop(loop[1], loops[lj]), eachindex(loops)
        )
        want_ccw = iseven(depth)
        oriented[li] = (sa > 0) == want_ccw ? loop : reverse(loop)
    end

    n_vertices = sum(length, oriented)
    vertices = Vector{SVector{2, T}}(undef, n_vertices)
    segments = Vector{NTuple{2, Int32}}(undef, n_vertices)
    normal = Vector{SVector{2, T}}(undef, n_vertices)
    vertex_normal = [zero(SVector{2, T}) for _ in 1:n_vertices]

    offset = 0
    for loop in oriented
        n = length(loop)
        @inbounds for i in 1:n
            vertices[offset + i] = loop[i]
        end
        @inbounds for i in 1:n
            vi = offset + i
            vj = offset + mod1(i + 1, n)
            segments[vi] = (Int32(vi), Int32(vj))
            d = vertices[vj] - vertices[vi]
            mag = norm(d)
            # Outward normal: edge direction rotated -90° — for a CCW outer
            # loop this points away from the enclosed area, and for a CW hole
            # loop into the hole (also away from the domain).
            nrm = mag < eps(T) * 100 ? zero(SVector{2, T}) :
                SVector{2, T}(d[2], -d[1]) / mag
            normal[vi] = nrm
            vertex_normal[vi] += nrm
            vertex_normal[vj] += nrm
        end
        offset += n
    end

    bbox_min, bbox_max = _compute_bbox_raw_2d(vertices)
    return SegmentIndex{T}(
        vertices, segments, normal, vertex_normal, bbox_min, bbox_max, len_unit
    )
end

function _compute_bbox_raw_2d(vertices::Vector{SVector{2, T}}) where {T}
    min_x = min_y = typemax(T)
    max_x = max_y = typemin(T)
    @inbounds for v in vertices
        min_x = min(min_x, v[1]); max_x = max(max_x, v[1])
        min_y = min(min_y, v[2]); max_y = max(max_y, v[2])
    end
    eps_val = max(eps(T) * 100, T(_DEGENERATE_EPS))
    min_x == max_x && (min_x -= eps_val; max_x += eps_val)
    min_y == max_y && (min_y -= eps_val; max_y += eps_val)
    return SVector{2, T}(min_x, min_y), SVector{2, T}(max_x, max_y)
end

# ============================================================================
# Segment geometry kernels (2D counterparts of geometric_utils.jl)
# ============================================================================

"""
    closest_point_on_segment_feature(p, a, b) -> (closest_point, feature)

Closest point on segment `a → b`, plus the feature it lies on:
`FEATURE_FACE` (segment interior), `FEATURE_VERTEX_1` (`a`), or
`FEATURE_VERTEX_2` (`b`).
"""
@inline function closest_point_on_segment_feature(
        p::SVector{2, T}, a::SVector{2, T}, b::SVector{2, T}
    ) where {T <: Real}
    ab = b - a
    denom = dot(ab, ab)
    denom < eps(T) && return a, FEATURE_VERTEX_1   # degenerate segment
    t = dot(p - a, ab) / denom
    t <= zero(T) && return a, FEATURE_VERTEX_1
    t >= one(T) && return b, FEATURE_VERTEX_2
    return a + t * ab, FEATURE_FACE
end

"Segment–AABB overlap via Liang–Barsky parameter clipping."
function segment_box_intersection(
        a::SVector{2, T}, b::SVector{2, T},
        bbox_min::SVector{2, T}, bbox_max::SVector{2, T},
    ) where {T <: Real}
    t0, t1 = zero(T), one(T)
    d = b - a
    @inbounds for i in 1:2
        if abs(d[i]) < eps(T)
            (a[i] < bbox_min[i] || a[i] > bbox_max[i]) && return false
        else
            inv_d = inv(d[i])
            tA = (bbox_min[i] - a[i]) * inv_d
            tB = (bbox_max[i] - a[i]) * inv_d
            tA > tB && ((tA, tB) = (tB, tA))
            t0 = max(t0, tA)
            t1 = min(t1, tB)
            t0 > t1 && return false
        end
    end
    return true
end

# ============================================================================
# Quadtree construction
# ============================================================================

"""
Quadtree spatial index for 2D boundary queries — the `𝔼{2}` implementation of
[`AbstractGeometryIndex`](@ref), mirroring [`TriangleOctree`](@ref).
Accelerates `isinside`, signed distance, and leaf classification over the
segments of a [`SegmentIndex`](@ref).
"""
struct SegmentQuadtree{T <: Real} <: AbstractGeometryIndex{𝔼{2}}
    tree::SpatialTree{2, Int, T}
    index::SegmentIndex{T}
    leaf_classification::Union{Nothing, Vector{Int8}}
end

"""
    SegmentQuadtree(loops; len_unit, tolerance_relative=1e-6, min_ratio=1e-6,
                    classify_leaves=true)

Build a geometry-adaptive quadtree over closed 2D boundary loops (each a
`Vector{SVector{2,T}}` of vertices ordered around the loop, in any
orientation). Multiple loops describe multiply-connected domains: outer
boundaries plus holes, resolved automatically by nesting parity.
"""
function SegmentQuadtree(
        loops::Vector{Vector{SVector{2, T}}};
        len_unit::Unitful.Units = Unitful.m,
        tolerance_relative = 1.0e-6,
        min_ratio = 1.0e-6,
        classify_leaves::Bool = true,
    ) where {T <: Real}
    index = SegmentIndex(T, loops, len_unit)
    return SegmentQuadtree(index; tolerance_relative, min_ratio, classify_leaves)
end

"Single-loop convenience constructor."
SegmentQuadtree(loop::Vector{SVector{2, T}}; kwargs...) where {T <: Real} =
    SegmentQuadtree([loop]; kwargs...)

function SegmentQuadtree(
        index::SegmentIndex{T};
        tolerance_relative = 1.0e-6,
        min_ratio = 1.0e-6,
        classify_leaves::Bool = true,
    ) where {T <: Real}
    criterion = VertexResolutionCriterion(index; tolerance_relative, min_ratio)

    tree = _create_root_tree(Val(2), index.bbox_min, index.bbox_max, num_segments(index))
    _subdivide_geometry_tree!(tree, index, 1, criterion)
    balance_octree!(
        tree, criterion;
        redistribute! = (t, box_idx) -> _redistribute_elements!(t, index, box_idx),
    )
    classification = classify_leaves ? _classify_leaves(tree, index) : nothing

    return SegmentQuadtree{T}(tree, index, classification)
end

function VertexResolutionCriterion(
        index::SegmentIndex{T};
        tolerance_relative = 1.0e-6,
        min_ratio = 1.0e-6,
    ) where {T}
    num_segments(index) > 0 ||
        throw(ArgumentError("Boundary must contain at least one segment"))

    bbox_min, bbox_max = _compute_bbox(index)
    diagonal = norm(bbox_max - bbox_min)

    tolerance = diagonal * T(tolerance_relative)
    return VertexResolutionCriterion(tolerance * tolerance, diagonal * T(min_ratio))
end

function _redistribute_elements!(
        tree::SpatialTree{2, Int, T},
        index::SegmentIndex{T},
        box_idx::Int,
    ) where {T <: Real}
    parent_segments = tree.element_lists[box_idx]
    isempty(parent_segments) && return

    kids = children(tree, box_idx)
    for seg_idx in parent_segments
        a, b = _get_segment_vertices(index, seg_idx)
        for child_idx in kids
            child_min, child_max = box_bounds(tree, child_idx)
            if segment_box_intersection(a, b, child_min, child_max)
                push!(tree.element_lists[child_idx], seg_idx)
            end
        end
    end
    return
end

# ============================================================================
# Signed distance and classification
# ============================================================================

# Callable struct (not a closure) so the per-query traversal stays
# allocation-free; see `_nearest_element_tree!`.
struct _SegmentUpdater{T <: Real}
    point::SVector{2, T}
    index::SegmentIndex{T}
    state::NearestElementState{2, T}
end

@inline function (u::_SegmentUpdater{T})(seg_idx::Int) where {T}
    a, b = _get_segment_vertices(u.index, seg_idx)
    cp, feature = closest_point_on_segment_feature(u.point, a, b)
    dvec = u.point - cp
    d2 = dot(dvec, dvec)

    if d2 < u.state.best_dist_sq
        u.state.best_dist_sq = d2
        u.state.closest_idx = seg_idx
        u.state.closest_pt = cp
        u.state.closest_feature = feature
    end

    return nothing
end

function _nearest_segment_quadtree!(
        point::SVector{2, T},
        tree::SpatialTree{2, Int, T},
        index::SegmentIndex{T},
        box_idx::Int,
        state::NearestElementState{2, T},
    ) where {T <: Real}
    return _nearest_element_tree!(
        point, tree, box_idx, state, _SegmentUpdater(point, index, state)
    )
end

"Pseudonormal of the feature (segment / vertex) the closest point lies on."
@inline function _segment_feature_pseudonormal(
        index::SegmentIndex{T}, seg_idx::Int, feature::Int8
    ) where {T}
    feature == FEATURE_FACE && return index.normal[seg_idx]
    s = @inbounds index.segments[seg_idx]
    vi = feature == FEATURE_VERTEX_1 ? s[1] : s[2]
    return @inbounds index.vertex_normal[vi]
end

"""
Signed distance to the boundary loops: distance to the closest point, signed
by the pseudonormal of the closest feature (negative inside the domain) —
the 2D specialization of the 3D triangle query.
"""
function _compute_signed_distance_quadtree(
        point::SVector{2, T},
        index::SegmentIndex{T},
        tree::SpatialTree{2, Int, T},
    ) where {T <: Real}
    state = NearestElementState(point)
    _nearest_segment_quadtree!(point, tree, index, 1, state)

    state.closest_idx == 0 && return typemax(T)

    n = _segment_feature_pseudonormal(index, state.closest_idx, state.closest_feature)
    s = dot(point - state.closest_pt, n)

    nearest_dist = sqrt(state.best_dist_sq)
    s > zero(T) && return nearest_dist
    s == zero(T) && return zero(T)
    return -nearest_dist
end

function _classify_leaves(
        tree::SpatialTree{2, Int, T}, index::SegmentIndex{T}
    ) where {T <: Real}
    function segment_query(point::SVector{2, T2}, tol::T2) where {T2 <: Real}
        sd = _compute_signed_distance_quadtree(point, index, tree)
        return _leaf_class_from_signed_distance(sd, tol)
    end

    return _force_occupied_boundary!(classify_leaves!(tree, segment_query), tree)
end

# ============================================================================
# Seam methods (the AbstractGeometryIndex contract)
# ============================================================================

"Axis-aligned bounds of the boundary geometry."
domain_bounds(g::SegmentQuadtree) = (g.index.bbox_min, g.index.bbox_max)

"""
Shared classification of a point against a SegmentQuadtree — mirrors
`_classify_point_octree`: bbox fast path, cached leaf classification for
INTERIOR/EXTERIOR dispatch, exact signed distance only for BOUNDARY leaves.
"""
@inline function classify_point(
        g::SegmentQuadtree{T}, point::SVector{2, <:Real}, tol
    ) where {T}
    # Seam policy: convert a foreign-precision query once at the entry point;
    # everything below runs strictly in the quadtree's machine type T.
    p = SVector{2, T}(point)
    t = T(tol)
    if any(p .< g.index.bbox_min .- t) || any(p .> g.index.bbox_max .+ t)
        return LEAF_EXTERIOR
    end
    seg_cls = g.leaf_classification
    if !isnothing(seg_cls)
        leaf_idx = find_leaf(g.tree, p)
        cls = seg_cls[leaf_idx]
        cls != LEAF_BOUNDARY && return cls
    end
    sd = _compute_signed_distance_quadtree(p, g.index, g.tree)
    tol_val = isnothing(seg_cls) ? zero(T) : t
    return _leaf_class_from_signed_distance(sd, tol_val)
end

"""
Fast interior/exterior test using the quadtree spatial index.
"""
function isinside(point::SVector{2, T}, quadtree::SegmentQuadtree) where {T <: Real}
    return classify_point(quadtree, point, zero(T)) == LEAF_INTERIOR
end

function isinside(points::Vector{SVector{2, T}}, quadtree::SegmentQuadtree) where {T <: Real}
    return tmap(p -> isinside(p, quadtree), points)
end

@inline function _extract_vertex(::Type{T}, point::Point{𝔼{2}}) where {T}
    coords = Meshes.to(point)
    return SVector{2, T}(ustrip(coords[1]), ustrip(coords[2]))
end

function isinside(point::Point{𝔼{2}}, quadtree::SegmentQuadtree{T}) where {T}
    return isinside(_extract_vertex(T, point), quadtree)
end

function isinside(
        points::AbstractVector{<:Point{𝔼{2}}},
        quadtree::SegmentQuadtree,
    )
    return tmap(p -> isinside(p, quadtree), points)
end

Base.length(quadtree::SegmentQuadtree) = num_segments(quadtree.index)

"""
    num_leaves(quadtree::SegmentQuadtree) -> Int

Return the number of leaf nodes in the quadtree's spatial subdivision.
"""
num_leaves(quadtree::SegmentQuadtree) = length(all_leaves(quadtree.tree))

"""
    num_segments(quadtree::SegmentQuadtree) -> Int

Return the number of boundary segments indexed by the quadtree.
"""
num_segments(quadtree::SegmentQuadtree) = num_segments(quadtree.index)
