"""
    TriangleIndex{T}

The package's runtime representation of a triangle mesh: an indexed,
unit-stripped, machine-typed cache built once from a `SimpleMesh` at the
package boundary. After construction, query paths read contiguous
`SVector{3,T}` arrays — they never touch the Meshes.jl object model.

Fields (all canonical — no derived cache):
- `vertices`/`triangles`: indexed representation (unique coords + per-triangle
  vertex indices). Also the serialization format.
- `face`: precomputed unit face normals.
- `edge`/`vertex`: angle-weighted pseudonormals (Bærentzen & Aanæs 2005)
  keyed by exact coordinates — the sign-exact feature normals for
  signed-distance queries.
- `bbox_min`/`bbox_max`: mesh bounding box (computed once at construction).
- `len_unit`: the unit stripped from mesh coordinates, re-attached at exits.

This struct is self-sufficient: nothing in the package carries a
`SimpleMesh` reference after `TriangleIndex(T, mesh)` runs.
"""
struct TriangleIndex{T <: Real}
    vertices::Vector{SVector{3, T}}
    triangles::Vector{NTuple{3, Int32}}
    face::Vector{SVector{3, T}}
    edge::Dict{Tuple{SVector{3, T}, SVector{3, T}}, SVector{3, T}}
    vertex::Dict{SVector{3, T}, SVector{3, T}}
    bbox_min::SVector{3, T}
    bbox_max::SVector{3, T}
    len_unit::Unitful.Units
end

"""
Octree spatial index for triangle mesh queries. Accelerates isinside(),
signed distance, etc. Carries the mesh as a `TriangleIndex{T}` — no
`SimpleMesh` reference, no Meshes.jl in the runtime.
"""
struct TriangleOctree{T <: Real}
    tree::SpatialOctree{Int, T}
    index::TriangleIndex{T}
    leaf_classification::Union{Nothing, Vector{Int8}}
end

const _BBOX_EXPANSION = 1.02
const _DEGENERATE_EPS = 1.0e-10

"""
Subdivide based on vertex density within box bounds.
"""
struct VertexResolutionCriterion{T <: Real} <: SubdivisionCriterion
    tolerance_sq::T
    absolute_min::T
end

function VertexResolutionCriterion(
        index::TriangleIndex{T};
        tolerance_relative = 1.0e-6,
        min_ratio = 1.0e-6
    ) where {T}
    n_triangles = num_triangles(index)
    n_triangles > 0 || throw(ArgumentError("Mesh must contain at least one triangle"))

    bbox_min, bbox_max = _compute_bbox(index)
    diagonal = norm(bbox_max - bbox_min)

    tolerance = diagonal * T(tolerance_relative)
    tolerance_sq = tolerance * tolerance
    absolute_min = diagonal * T(min_ratio)

    return VertexResolutionCriterion(tolerance_sq, absolute_min)
end

function should_subdivide(
        c::VertexResolutionCriterion{T},
        tree,
        box_idx,
        index::TriangleIndex{T}
    ) where {T}
    box_size(tree, box_idx) <= c.absolute_min && return false
    isempty(tree.element_lists[box_idx]) && return false

    bbox_min, bbox_max = box_bounds(tree, box_idx)
    found = SVector{3, T}[]

    for tri_idx in tree.element_lists[box_idx], v in _get_triangle_vertices(index, tri_idx)
        all(bbox_min .<= v .<= bbox_max) || continue
        all(sum(abs2, v - existing) >= c.tolerance_sq for existing in found) || continue
        push!(found, v)
        length(found) > 1 && return true
    end

    return false
end

function can_subdivide(c::VertexResolutionCriterion, tree, box_idx)
    return box_size(tree, box_idx) > c.absolute_min
end

mutable struct NearestTriangleState{T <: Real}
    best_dist_sq::T
    closest_idx::Int
    closest_pt::SVector{3, T}
    closest_feature::Int8
end

NearestTriangleState{T}(point::SVector{3, T}) where {T <: Real} =
    NearestTriangleState{T}(typemax(T), 0, point, FEATURE_FACE)

@inline function _extract_vertex(::Type{T}, vert) where {T}
    coords = Meshes.to(vert)
    return SVector{3, T}(ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3]))
end

@inline function _get_triangle_vertices(index::TriangleIndex, tri_idx::Int)
    t = @inbounds index.triangles[tri_idx]
    @inbounds return index.vertices[t[1]], index.vertices[t[2]], index.vertices[t[3]]
end

@inline _get_triangle_normal(index::TriangleIndex, tri_idx::Int) =
    (@inbounds index.face[tri_idx])

num_triangles(index::TriangleIndex) = length(index.triangles)

function TriangleIndex(::Type{T}, mesh::SimpleMesh) where {T <: Real}
    verts = Meshes.vertices(mesh)
    vertices = Vector{SVector{3, T}}(undef, length(verts))
    @inbounds for i in eachindex(verts)
        vertices[i] = _extract_vertex(T, verts[i])
    end

    n = Meshes.nelements(mesh)
    triangles = Vector{NTuple{3, Int32}}(undef, n)
    face = Vector{SVector{3, T}}(undef, n)
    for (i, connec) in enumerate(Meshes.elements(Meshes.topology(mesh)))
        idx = Meshes.indices(connec)
        length(idx) == 3 || throw(
            ArgumentError(
                "TriangleIndex requires a pure-triangle mesh; " *
                    "element $i has $(length(idx)) vertices"
            )
        )
        t = (Int32(idx[1]), Int32(idx[2]), Int32(idx[3]))
        triangles[i] = t
        a = vertices[t[1]]
        b = vertices[t[2]]
        c = vertices[t[3]]
        # Precompute unit face normal.
        nrm = cross(b - a, c - a)
        mag = norm(nrm)
        @inbounds face[i] = mag < eps(T) * 100 ? zero(SVector{3, T}) : nrm / mag
    end

    # Angle-weighted pseudonormals (Bærentzen & Aanæs 2005) for exact sign
    # determination. Edges/vertices keyed by exact coordinates — triangle-soup
    # input (binary STL with duplicated vertices) needs no topology cleanup.
    edge = Dict{Tuple{SVector{3, T}, SVector{3, T}}, SVector{3, T}}()
    vertex = Dict{SVector{3, T}, SVector{3, T}}()
    sizehint!(edge, 3 * n ÷ 2 + 1)
    sizehint!(vertex, n ÷ 2 + 2)
    z = zero(SVector{3, T})
    @inbounds for i in 1:n
        t = triangles[i]
        v1, v2, v3 = vertices[t[1]], vertices[t[2]], vertices[t[3]]
        nhat = face[i]
        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            k = _edge_key(va, vb)
            edge[k] = get(edge, k, z) + nhat
        end
        for (vc, va, vb) in ((v1, v2, v3), (v2, v3, v1), (v3, v1, v2))
            vertex[vc] = get(vertex, vc, z) + _corner_angle(vc, va, vb) * nhat
        end
    end

    bbox_min, bbox_max = _compute_bbox_raw(vertices)

    len_unit = Unitful.unit(Meshes.lentype(mesh))
    return TriangleIndex{T}(vertices, triangles, face, edge, vertex, bbox_min, bbox_max, len_unit)
end

# Bounding box from raw vertices (used at construction time before the
# TriangleIndex is assembled).
function _compute_bbox_raw(vertices::Vector{SVector{3, T}}) where {T}
    min_x = min_y = min_z = typemax(T)
    max_x = max_y = max_z = typemin(T)
    @inbounds for v in vertices
        min_x = min(min_x, v[1]); max_x = max(max_x, v[1])
        min_y = min(min_y, v[2]); max_y = max(max_y, v[2])
        min_z = min(min_z, v[3]); max_z = max(max_z, v[3])
    end
    eps_val = max(eps(T) * 100, T(_DEGENERATE_EPS))
    min_x == max_x && (min_x -= eps_val; max_x += eps_val)
    min_y == max_y && (min_y -= eps_val; max_y += eps_val)
    min_z == max_z && (min_z -= eps_val; max_z += eps_val)
    return SVector{3, T}(min_x, min_y, min_z), SVector{3, T}(max_x, max_y, max_z)
end

# Field-access shortcut — bbox is stored on the index.
_compute_bbox(index::TriangleIndex{T}) where {T} = (index.bbox_min, index.bbox_max)

@inline function _edge_key(v1::SVector{3, T}, v2::SVector{3, T}) where {T}
    return v1 < v2 ? (v1, v2) : (v2, v1)
end

@inline function _corner_angle(vc::SVector{3, T}, va::SVector{3, T}, vb::SVector{3, T}) where {T}
    u = va - vc
    w = vb - vc
    den = sqrt(dot(u, u) * dot(w, w))
    den < eps(T) && return zero(T)
    return acos(clamp(dot(u, w) / den, -one(T), one(T)))
end

"""
Pseudonormal of the feature (face / edge / vertex) the closest point lies on.
Falls back to the face normal if the feature key is missing — cannot happen
for features built from this mesh's own triangles, but keeps the query total.
"""
@inline function _feature_pseudonormal(
        index::TriangleIndex{T},
        tri_idx::Int,
        feature::Int8,
        v1::SVector{3, T},
        v2::SVector{3, T},
        v3::SVector{3, T},
    ) where {T}
    f = index.face[tri_idx]
    feature == FEATURE_FACE && return f
    feature == FEATURE_VERTEX_1 && return get(index.vertex, v1, f)
    feature == FEATURE_VERTEX_2 && return get(index.vertex, v2, f)
    feature == FEATURE_VERTEX_3 && return get(index.vertex, v3, f)
    feature == FEATURE_EDGE_12 && return get(index.edge, _edge_key(v1, v2), f)
    feature == FEATURE_EDGE_13 && return get(index.edge, _edge_key(v1, v3), f)
    return get(index.edge, _edge_key(v2, v3), f)
end

"""
Check if triangle faces are consistently oriented (manifold orientation test).
"""
has_consistent_normals(mesh::SimpleMesh{M, C}) where {M, C} =
    has_consistent_normals(TriangleIndex(CoordRefSystems.mactype(C), mesh))

function has_consistent_normals(index::TriangleIndex{T}) where {T}
    n = num_triangles(index)
    n <= 1 && return true

    edge_map = Dict{
        Tuple{SVector{3, T}, SVector{3, T}},
        Tuple{Int, SVector{3, T}, SVector{3, T}},
    }()
    sizehint!(edge_map, 3 * n)

    for i in 1:n
        v1, v2, v3 = _get_triangle_vertices(index, i)

        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            key = _edge_key(va, vb)
            existing = get(edge_map, key, nothing)

            if existing !== nothing
                (_, other_va, other_vb) = existing
                same_direction = (va ≈ other_va && vb ≈ other_vb)
                if same_direction
                    return false
                end
            else
                edge_map[key] = (i, va, vb)
            end
        end
    end

    return true
end

"""
Signed volume of a closed triangle mesh (divergence theorem,
`Σ dot(v1, v2 × v3) / 6`). Positive iff the winding orients normals outward.
Catches globally inside-out meshes, which `has_consistent_normals` cannot
(a perfectly consistent but inverted mesh classifies its complement as
interior — exactly the 2026-06-11 cavity corruption #2). Only meaningful for
closed surfaces.
"""
function _signed_volume(index::TriangleIndex{T}) where {T <: Real}
    vol = zero(T)
    @inbounds for i in 1:num_triangles(index)
        v1, v2, v3 = _get_triangle_vertices(index, i)
        vol += dot(v1, cross(v2, v3))
    end
    return vol / 6
end

function _create_root_octree(index::TriangleIndex{T}) where {T}
    n_triangles = num_triangles(index)
    bbox_min, bbox_max = _compute_bbox(index)

    bbox_sz = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) * T(0.5)
    expansion_factor = T(_BBOX_EXPANSION)
    expanded_half_size = (bbox_sz * T(0.5)) * expansion_factor
    bbox_min = bbox_center - expanded_half_size
    bbox_max = bbox_center + expanded_half_size

    root_size = maximum(bbox_max - bbox_min)

    estimated_boxes = max(1000, n_triangles * 2)
    tree = SpatialOctree{Int, T}(bbox_min, root_size; initial_capacity = estimated_boxes)

    root_elements = tree.element_lists[1]
    for tri_idx in 1:n_triangles
        push!(root_elements, tri_idx)
    end

    return tree
end

"""
Build geometry-adaptive octree for triangle mesh.
"""
function TriangleOctree(
        mesh::SimpleMesh{M, C};
        tolerance_relative = 1.0e-6,
        min_ratio = 1.0e-6,
        classify_leaves::Bool = true,
        verify_orientation::Bool = true,
    ) where {M <: Manifold, C <: CRS}
    T = CoordRefSystems.mactype(C)
    index = TriangleIndex(T, mesh)

    if verify_orientation && !has_consistent_normals(index)
        throw(
            ArgumentError(
                "Triangle mesh has orientation errors (flipped faces). " *
                    "Some triangles have their faces oriented incorrectly. " *
                    "This will cause incorrect isinside() calculations. " *
                    "The mesh needs to be repaired before use. " *
                    "To skip this check, pass `verify_orientation=false`.",
            )
        )
    end

    # A consistently-wound mesh can still be globally inside-out (all normals
    # pointing into the solid): isinside would then classify the complement of
    # the intended domain as interior, and no d_NN-type downstream metric can
    # detect it. The signed volume is an O(n) exact witness for closed meshes:
    # an inverted mesh measures minus the domain volume. Gated on
    # classify_leaves (open surfaces, fine for distance-only queries, have a
    # meaningless ≈0 signed volume — the scale-relative threshold lets them
    # through).
    if verify_orientation && classify_leaves
        bbox_min, bbox_max = _compute_bbox(index)
        vol_floor = -T(_DEGENERATE_EPS) * norm(bbox_max - bbox_min)^3
        if _signed_volume(index) < vol_floor
            throw(
                ArgumentError(
                    "Triangle mesh is inside-out (negative signed volume): " *
                        "the winding orients normals into the solid, so isinside() " *
                        "would classify the complement of the domain as interior. " *
                        "Flip the triangle winding. To skip this check, pass " *
                        "`verify_orientation=false`.",
                )
            )
        end
    end

    criterion = VertexResolutionCriterion(
        index;
        tolerance_relative = tolerance_relative,
        min_ratio = min_ratio
    )

    tree = _create_root_octree(index)
    _subdivide_triangle_octree!(tree, index, 1, criterion)
    balance_octree!(tree, criterion)
    classification = classify_leaves ? _classify_leaves(tree, index) : nothing

    return TriangleOctree{T}(tree, index, classification)
end

function _subdivide_triangle_octree!(
        tree::SpatialOctree{Int, T},
        index::TriangleIndex{T},
        box_idx::Int,
        criterion::VertexResolutionCriterion,
    ) where {T <: Real}
    if !should_subdivide(criterion, tree, box_idx, index)
        return
    end

    parent_triangles = tree.element_lists[box_idx]
    isempty(parent_triangles) && return

    subdivide!(tree, box_idx)
    children = tree.children[box_idx]

    for tri_idx in parent_triangles
        v1, v2, v3 = _get_triangle_vertices(index, tri_idx)

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
        _subdivide_triangle_octree!(tree, index, child_idx, criterion)
    end
    return
end

@inline function _point_box_distance_sq(
        point::SVector{3, T},
        bbox_min::SVector{3, T},
        bbox_max::SVector{3, T},
    ) where {T <: Real}
    dx = point[1] < bbox_min[1] ? (bbox_min[1] - point[1]) :
        (point[1] > bbox_max[1] ? (point[1] - bbox_max[1]) : zero(T))
    dy = point[2] < bbox_min[2] ? (bbox_min[2] - point[2]) :
        (point[2] > bbox_max[2] ? (point[2] - bbox_max[2]) : zero(T))
    dz = point[3] < bbox_min[3] ? (bbox_min[3] - point[3]) :
        (point[3] > bbox_max[3] ? (point[3] - bbox_max[3]) : zero(T))
    return dx * dx + dy * dy + dz * dz
end

@inline function _update_closest_triangle!(
        point::SVector{3, T},
        index::TriangleIndex{T},
        tri_idx::Int,
        state::NearestTriangleState{T},
    ) where {T <: Real}
    v1, v2, v3 = _get_triangle_vertices(index, tri_idx)
    cp, feature = closest_point_on_triangle_feature(point, v1, v2, v3)
    dvec = point - cp
    d2 = dot(dvec, dvec)

    if d2 < state.best_dist_sq
        state.best_dist_sq = d2
        state.closest_idx = tri_idx
        state.closest_pt = cp
        state.closest_feature = feature
    end

    return nothing
end

function _nearest_triangle_octree!(
        point::SVector{3, T},
        tree::SpatialOctree{Int, T},
        index::TriangleIndex{T},
        box_idx::Int,
        state::NearestTriangleState{T},
    ) where {T <: Real}
    bbox_min, bbox_max = box_bounds(tree, box_idx)
    _point_box_distance_sq(point, bbox_min, bbox_max) > state.best_dist_sq && return

    if is_leaf(tree, box_idx)
        @inbounds for tri_idx in tree.element_lists[box_idx]
            _update_closest_triangle!(point, index, tri_idx, state)
        end
        return
    end

    children = tree.children[box_idx]
    dists = MVector{8, T}(ntuple(_ -> typemax(T), Val(8)))
    idxs = MVector{8, Int}(ntuple(_ -> 0, Val(8)))
    n_valid = 0
    @inbounds for child_idx in children
        child_idx == 0 && continue
        cmin, cmax = box_bounds(tree, child_idx)
        d2 = _point_box_distance_sq(point, cmin, cmax)
        if d2 <= state.best_dist_sq
            n_valid += 1
            pos = n_valid
            while pos > 1 && d2 < dists[pos - 1]
                dists[pos] = dists[pos - 1]
                idxs[pos] = idxs[pos - 1]
                pos -= 1
            end
            dists[pos] = d2
            idxs[pos] = child_idx
        end
    end

    return @inbounds for i in 1:n_valid
        _nearest_triangle_octree!(point, tree, index, idxs[i], state)
    end
end

"""
Signed distance to the mesh: distance to the closest point, signed by the
angle-weighted pseudonormal of the closest feature (exact for watertight,
consistently outward-oriented meshes — Bærentzen & Aanæs 2005). Returns 0
only for points exactly on the surface (or at a degenerate fold where the
feature pseudonormal vanishes).
"""
function _compute_signed_distance_octree(
        point::SVector{3, T},
        index::TriangleIndex{T},
        tree::SpatialOctree{Int, T},
    ) where {T <: Real}
    state = NearestTriangleState{T}(point)
    _nearest_triangle_octree!(point, tree, index, 1, state)

    if state.closest_idx == 0
        return typemax(T)
    end

    # Re-gather the winning triangle's vertices once (no per-triangle cache
    # on the traversal state) to select the feature pseudonormal.
    v1, v2, v3 = _get_triangle_vertices(index, state.closest_idx)
    n = _feature_pseudonormal(
        index, state.closest_idx, state.closest_feature, v1, v2, v3
    )
    s = dot(point - state.closest_pt, n)

    nearest_dist = sqrt(state.best_dist_sq)
    s < zero(T) && return -nearest_dist
    s > zero(T) && return nearest_dist
    return zero(T)
end

function _compute_signed_distance_octree(
        point::SVector{3, <:Real}, octree::TriangleOctree{T}
    ) where {T}
    # Seam policy: convert a foreign-precision query once at the entry point;
    # the traversal below runs strictly in the octree's machine type T.
    return _compute_signed_distance_octree(
        SVector{3, T}(point), octree.index, octree.tree
    )
end

function _classify_leaves(
        tree::SpatialOctree{Int, T}, index::TriangleIndex{T}
    ) where {T <: Real}
    function mesh_query(point::SVector{3, T}, tol::T) where {T <: Real}
        sd = _compute_signed_distance_octree(point, index, tree)
        return _leaf_class_from_signed_distance(sd, tol)
    end

    classification = classify_leaves!(tree, mesh_query)

    for leaf_idx in all_leaves(tree)
        if !isempty(tree.element_lists[leaf_idx])
            classification[leaf_idx] = LEAF_BOUNDARY
        end
    end

    return classification
end

Base.length(octree::TriangleOctree) = num_triangles(octree.index)
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))
num_triangles(octree::TriangleOctree) = num_triangles(octree.index)

"""
    _classify_point_octree(point, octree; tol=0) -> Int8

Shared classification of a point against a TriangleOctree. Returns
`LEAF_INTERIOR`, `LEAF_EXTERIOR`, or `LEAF_BOUNDARY`. Uses the cached leaf
classification for fast INTERIOR/EXTERIOR dispatch; only BOUNDARY probes
fall through to the full signed-distance computation.

`tol` expands the mesh bounding box for the exterior fast-path (set to 0 for
exact bbox checks, or a positive tolerance for conservative classification).
"""
@inline function _classify_point_octree(
        point::SVector{3, <:Real}, octree::TriangleOctree{T}; tol::Real = zero(T)
    ) where {T}
    # Seam policy: convert a foreign-precision query once at the entry point;
    # everything below runs strictly in the octree's machine type T.
    p = SVector{3, T}(point)
    t = T(tol)
    if any(p .< octree.index.bbox_min .- t) || any(p .> octree.index.bbox_max .+ t)
        return LEAF_EXTERIOR
    end
    tri_cls = octree.leaf_classification
    if !isnothing(tri_cls)
        leaf_idx = find_leaf(octree.tree, p)
        cls = tri_cls[leaf_idx]
        cls != LEAF_BOUNDARY && return cls
    end
    sd = _compute_signed_distance_octree(p, octree.index, octree.tree)
    tol_val = isnothing(tri_cls) ? zero(T) : t
    return _leaf_class_from_signed_distance(sd, tol_val)
end

"""
Fast interior/exterior test using octree spatial index.
"""
function isinside(point::SVector{3, T}, octree::TriangleOctree) where {T <: Real}
    return _classify_point_octree(point, octree) == LEAF_INTERIOR
end

function isinside(points::Vector{SVector{3, T}}, octree::TriangleOctree) where {T <: Real}
    return tmap(p -> isinside(p, octree), points)
end

function isinside(point::Point{𝔼{3}}, octree::TriangleOctree{T}) where {T}
    sv = _extract_vertex(T, point)
    return isinside(sv, octree)
end

function isinside(
        points::AbstractVector{<:Point{𝔼{3}}},
        octree::TriangleOctree,
    )
    return tmap(p -> isinside(p, octree), points)
end
