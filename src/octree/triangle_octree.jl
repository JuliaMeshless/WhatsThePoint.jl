"""
    MeshPseudonormals{T}

Angle-weighted pseudonormals (Bærentzen & Aanæs 2005) for exact sign
determination in signed-distance queries. For a watertight, consistently
outward-oriented triangle mesh, `sign(dot(p - cp, n_feature))` is provably
correct for every query point `p`, where `cp` is the closest point on the
mesh and `n_feature` is the normal of the closest *feature*: the face normal
on a face interior, the sum of the two incident face normals on an edge, and
the incidence-angle-weighted sum of face normals at a vertex. (Only the sign
of the dot product is used, so the edge/vertex sums are stored unnormalized.)

This replaces the previous distance-weighted "local sign vote" over the
triangles of one leaf, which was heuristic — it could come back ambiguous
(mapped to EXTERIOR by `isinside`) and could mix votes from the two sides of
a thin sheet — and slower (a second `find_leaf` plus a closest-point
computation per vote triangle on every query).

Edges and vertices are keyed by exact coordinates, so triangle-soup input
(e.g. binary STL with duplicated vertices) needs no topology cleanup.
"""
struct MeshPseudonormals{T <: Real}
    face::Vector{SVector{3, T}}
    edge::Dict{Tuple{SVector{3, T}, SVector{3, T}}, SVector{3, T}}
    vertex::Dict{SVector{3, T}, SVector{3, T}}
end

"""
Octree spatial index for triangle mesh queries. Accelerates isinside(), signed distance, etc.
"""
struct TriangleOctree{M <: Manifold, C <: CRS, T <: Real}
    tree::SpatialOctree{Int, T}
    mesh::SimpleMesh{M, C}
    leaf_classification::Union{Nothing, Vector{Int8}}
    mesh_bbox_min::SVector{3, T}  # Actual mesh bounding box (not expanded octree bounds)
    mesh_bbox_max::SVector{3, T}
    pseudonormals::MeshPseudonormals{T}
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
        mesh::SimpleMesh{M, C};
        tolerance_relative = 1.0e-6,
        min_ratio = 1.0e-6
    ) where {M, C}
    T = CoordRefSystems.mactype(C)
    n_triangles = Meshes.nelements(mesh)
    n_triangles > 0 || throw(ArgumentError("Mesh must contain at least one triangle"))

    bbox_min, bbox_max = _compute_bbox(T, mesh)
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
        mesh::SimpleMesh
    ) where {T}
    box_size(tree, box_idx) <= c.absolute_min && return false
    isempty(tree.element_lists[box_idx]) && return false

    bbox_min, bbox_max = box_bounds(tree, box_idx)
    found = SVector{3, T}[]

    for tri_idx in tree.element_lists[box_idx], v in _get_triangle_vertices(T, mesh, tri_idx)
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
    # Winning triangle's feature + vertices, kept so the signed-distance query
    # can select the feature pseudonormal without re-extracting the triangle.
    closest_feature::Int8
    v1::SVector{3, T}
    v2::SVector{3, T}
    v3::SVector{3, T}
end

NearestTriangleState{T}(point::SVector{3, T}) where {T <: Real} =
    NearestTriangleState{T}(typemax(T), 0, point, FEATURE_FACE, point, point, point)

@inline function _normalize_normal(::Type{T}, n_vec) where {T}
    n = SVector{3, T}(ustrip(n_vec[1]), ustrip(n_vec[2]), ustrip(n_vec[3]))
    n_mag = norm(n)
    if n_mag < eps(T) * 100
        error("Degenerate triangle: zero normal")
    end
    return n / n_mag
end

@inline function _extract_vertex(::Type{T}, vert) where {T}
    coords = Meshes.to(vert)
    return SVector{3, T}(ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3]))
end

@inline function _get_triangle_vertices(::Type{T}, mesh::SimpleMesh, tri_idx::Int) where {T}
    elem = mesh[tri_idx]
    verts = Meshes.vertices(elem)
    v1 = _extract_vertex(T, verts[1])
    v2 = _extract_vertex(T, verts[2])
    v3 = _extract_vertex(T, verts[3])
    return v1, v2, v3
end

@inline function _get_triangle_normal(::Type{T}, mesh::SimpleMesh, tri_idx::Int) where {T}
    elem = mesh[tri_idx]
    return _normalize_normal(T, Meshes.normal(elem))
end

function _compute_bbox(::Type{T}, mesh::SimpleMesh) where {T}
    n = Meshes.nelements(mesh)

    v1, v2, v3 = _get_triangle_vertices(T, mesh, 1)
    min_x = min(v1[1], v2[1], v3[1])
    min_y = min(v1[2], v2[2], v3[2])
    min_z = min(v1[3], v2[3], v3[3])
    max_x = max(v1[1], v2[1], v3[1])
    max_y = max(v1[2], v2[2], v3[2])
    max_z = max(v1[3], v2[3], v3[3])

    for i in 2:n
        v1, v2, v3 = _get_triangle_vertices(T, mesh, i)
        min_x = min(min_x, v1[1], v2[1], v3[1])
        min_y = min(min_y, v1[2], v2[2], v3[2])
        min_z = min(min_z, v1[3], v2[3], v3[3])
        max_x = max(max_x, v1[1], v2[1], v3[1])
        max_y = max(max_y, v1[2], v2[2], v3[2])
        max_z = max(max_z, v1[3], v2[3], v3[3])
    end

    eps_val = max(eps(T) * 100, T(_DEGENERATE_EPS))
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

    return SVector{3, T}(min_x, min_y, min_z), SVector{3, T}(max_x, max_y, max_z)
end

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

function MeshPseudonormals(::Type{T}, mesh::SimpleMesh) where {T <: Real}
    n = Meshes.nelements(mesh)
    face = Vector{SVector{3, T}}(undef, n)
    edge = Dict{Tuple{SVector{3, T}, SVector{3, T}}, SVector{3, T}}()
    vertex = Dict{SVector{3, T}, SVector{3, T}}()
    sizehint!(edge, 3 * n ÷ 2 + 1)
    sizehint!(vertex, n ÷ 2 + 2)
    z = zero(SVector{3, T})

    for i in 1:n
        v1, v2, v3 = _get_triangle_vertices(T, mesh, i)
        nhat = _get_triangle_normal(T, mesh, i)
        face[i] = nhat
        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            k = _edge_key(va, vb)
            edge[k] = get(edge, k, z) + nhat
        end
        for (vc, va, vb) in ((v1, v2, v3), (v2, v3, v1), (v3, v1, v2))
            vertex[vc] = get(vertex, vc, z) + _corner_angle(vc, va, vb) * nhat
        end
    end

    return MeshPseudonormals(face, edge, vertex)
end

"""
Pseudonormal of the feature (face / edge / vertex) the closest point lies on.
Falls back to the face normal if the feature key is missing — cannot happen
for features built from this mesh's own triangles, but keeps the query total.
"""
@inline function _feature_pseudonormal(
        pn::MeshPseudonormals{T},
        tri_idx::Int,
        feature::Int8,
        v1::SVector{3, T},
        v2::SVector{3, T},
        v3::SVector{3, T},
    ) where {T}
    f = pn.face[tri_idx]
    feature == FEATURE_FACE && return f
    feature == FEATURE_VERTEX_1 && return get(pn.vertex, v1, f)
    feature == FEATURE_VERTEX_2 && return get(pn.vertex, v2, f)
    feature == FEATURE_VERTEX_3 && return get(pn.vertex, v3, f)
    feature == FEATURE_EDGE_12 && return get(pn.edge, _edge_key(v1, v2), f)
    feature == FEATURE_EDGE_13 && return get(pn.edge, _edge_key(v1, v3), f)
    return get(pn.edge, _edge_key(v2, v3), f)
end

"""
Check if triangle faces are consistently oriented (manifold orientation test).
"""
function has_consistent_normals(::Type{T}, mesh::SimpleMesh) where {T}
    n = Meshes.nelements(mesh)
    n <= 1 && return true

    edge_map = Dict{
        Tuple{SVector{3, T}, SVector{3, T}},
        Tuple{Int, SVector{3, T}, SVector{3, T}},
    }()
    sizehint!(edge_map, 3 * n)

    for i in 1:n
        v1, v2, v3 = _get_triangle_vertices(T, mesh, i)

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

function has_consistent_normals(mesh::SimpleMesh{M, C}) where {M, C}
    return has_consistent_normals(CoordRefSystems.mactype(C), mesh)
end

"""
Signed volume of a closed triangle mesh (divergence theorem,
`Σ dot(v1, v2 × v3) / 6`). Positive iff the winding orients normals outward.
Catches globally inside-out meshes, which `has_consistent_normals` cannot
(a perfectly consistent but inverted mesh classifies its complement as
interior — exactly the 2026-06-11 cavity corruption #2). Only meaningful for
closed surfaces.
"""
function _signed_volume(::Type{T}, mesh::SimpleMesh) where {T <: Real}
    vol = zero(T)
    for i in 1:Meshes.nelements(mesh)
        v1, v2, v3 = _get_triangle_vertices(T, mesh, i)
        vol += dot(v1, cross(v2, v3))
    end
    return vol / 6
end

function _create_root_octree(::Type{T}, mesh::SimpleMesh, n_triangles::Int) where {T}
    bbox_min, bbox_max = _compute_bbox(T, mesh)

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
    n_triangles = Meshes.nelements(mesh)

    if verify_orientation && !has_consistent_normals(T, mesh)
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
        bbox_min, bbox_max = _compute_bbox(T, mesh)
        vol_floor = -T(_DEGENERATE_EPS) * norm(bbox_max - bbox_min)^3
        if _signed_volume(T, mesh) < vol_floor
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
        mesh;
        tolerance_relative = tolerance_relative,
        min_ratio = min_ratio
    )

    tree = _create_root_octree(T, mesh, n_triangles)
    _subdivide_triangle_octree!(tree, mesh, 1, criterion)
    balance_octree!(tree, criterion)
    pseudonormals = MeshPseudonormals(T, mesh)
    classification = classify_leaves ? _classify_leaves(tree, mesh, pseudonormals) : nothing

    # Store actual mesh bounding box (not the expanded cubic octree bounds)
    mesh_bbox_min, mesh_bbox_max = _compute_bbox(T, mesh)

    return TriangleOctree(
        tree, mesh, classification, mesh_bbox_min, mesh_bbox_max, pseudonormals
    )
end

function _subdivide_triangle_octree!(
        tree::SpatialOctree{Int, T},
        mesh::SimpleMesh,
        box_idx::Int,
        criterion::VertexResolutionCriterion,
    ) where {T <: Real}
    if !should_subdivide(criterion, tree, box_idx, mesh)
        return
    end

    parent_triangles = tree.element_lists[box_idx]
    isempty(parent_triangles) && return

    subdivide!(tree, box_idx)
    children = tree.children[box_idx]

    for tri_idx in parent_triangles
        v1, v2, v3 = _get_triangle_vertices(T, mesh, tri_idx)

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
        mesh::SimpleMesh,
        tri_idx::Int,
        state::NearestTriangleState{T},
    ) where {T <: Real}
    v1, v2, v3 = _get_triangle_vertices(T, mesh, tri_idx)
    cp, feature = closest_point_on_triangle_feature(point, v1, v2, v3)
    dvec = point - cp
    d2 = dot(dvec, dvec)

    if d2 < state.best_dist_sq
        state.best_dist_sq = d2
        state.closest_idx = tri_idx
        state.closest_pt = cp
        state.closest_feature = feature
        state.v1 = v1
        state.v2 = v2
        state.v3 = v3
    end

    return nothing
end

function _nearest_triangle_octree!(
        point::SVector{3, T},
        tree::SpatialOctree{Int, T},
        mesh::SimpleMesh,
        box_idx::Int,
        state::NearestTriangleState{T},
    ) where {T <: Real}
    bbox_min, bbox_max = box_bounds(tree, box_idx)
    _point_box_distance_sq(point, bbox_min, bbox_max) > state.best_dist_sq && return

    if is_leaf(tree, box_idx)
        @inbounds for tri_idx in tree.element_lists[box_idx]
            _update_closest_triangle!(point, mesh, tri_idx, state)
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
        _nearest_triangle_octree!(point, tree, mesh, idxs[i], state)
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
        mesh::SimpleMesh,
        tree::SpatialOctree{Int, T},
        pn::MeshPseudonormals{T},
    ) where {T <: Real}
    state = NearestTriangleState{T}(point)
    _nearest_triangle_octree!(point, tree, mesh, 1, state)

    if state.closest_idx == 0
        return typemax(T)
    end

    n = _feature_pseudonormal(
        pn, state.closest_idx, state.closest_feature, state.v1, state.v2, state.v3
    )
    s = dot(point - state.closest_pt, n)

    nearest_dist = sqrt(state.best_dist_sq)
    s < zero(T) && return -nearest_dist
    s > zero(T) && return nearest_dist
    return zero(T)
end

function _compute_signed_distance_octree(
        point::SVector{3, T}, octree::TriangleOctree
    ) where {T <: Real}
    return _compute_signed_distance_octree(
        point, octree.mesh, octree.tree, octree.pseudonormals
    )
end

function _classify_leaves(
        tree::SpatialOctree{Int, T}, mesh::SimpleMesh, pn::MeshPseudonormals{T}
    ) where {T <: Real}
    function mesh_query(point::SVector{3, T}, tol::T) where {T <: Real}
        sd = _compute_signed_distance_octree(point, mesh, tree, pn)
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

Base.length(octree::TriangleOctree) = Meshes.nelements(octree.mesh)

"""
    num_leaves(octree::TriangleOctree) -> Int

Return the number of leaf nodes in the octree.
"""
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))

"""
    num_triangles(octree::TriangleOctree) -> Int

Return the number of triangles in the underlying surface mesh.
"""
num_triangles(octree::TriangleOctree) = Meshes.nelements(octree.mesh)

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
        point::SVector{3, T}, octree::TriangleOctree; tol::T = zero(T)
    ) where {T <: Real}
    if any(point .< octree.mesh_bbox_min .- tol) || any(point .> octree.mesh_bbox_max .+ tol)
        return LEAF_EXTERIOR
    end
    tri_cls = octree.leaf_classification
    if !isnothing(tri_cls)
        leaf_idx = find_leaf(octree.tree, point)
        cls = tri_cls[leaf_idx]
        cls != LEAF_BOUNDARY && return cls
    end
    sd = _compute_signed_distance_octree(point, octree)
    tol_val = isnothing(tri_cls) ? zero(T) : tol
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

function isinside(point::Point{𝔼{3}}, octree::TriangleOctree{𝔼{3}, C, T}) where {C, T}
    sv = _extract_vertex(T, point)
    return isinside(sv, octree)
end

function isinside(
        points::AbstractVector{<:Point{𝔼{3}}},
        octree::TriangleOctree{𝔼{3}},
    )
    return tmap(p -> isinside(p, octree), points)
end
