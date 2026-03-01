"""
    DensityAwareOctree <: AbstractNodeGenerationAlgorithm

Spacing-aware volume point generation using two-octree architecture:
- Triangle octree: geometry and robust inside/outside classification
- Node octree: spacing-adaptive volume partitioning for point allocation
"""
struct DensityAwareOctree{M<:Manifold,C<:CRS,T<:Real} <: AbstractNodeGenerationAlgorithm
    triangle_octree::TriangleOctree{M,C,T}
    boundary_oversampling::Float64
    placement::Symbol
end

function DensityAwareOctree(
    triangle_octree::TriangleOctree{M,C,T},
    boundary_oversampling::Real=2.0,
    placement::Symbol=:random,
) where {M<:Manifold,C<:CRS,T<:Real}
    boundary_oversampling > 0 || throw(ArgumentError("boundary_oversampling must be positive"))
    placement in (:random, :jittered, :lattice) ||
        throw(ArgumentError("placement must be one of :random, :jittered, :lattice"))
    return DensityAwareOctree{M,C,T}(triangle_octree, Float64(boundary_oversampling), placement)
end

function DensityAwareOctree(
    mesh::SimpleMesh{M,C};
    boundary_oversampling::Real=2.0,
    placement::Symbol=:random,
    tolerance_relative::Real=1e-6,
    min_ratio::Real=1e-6,
    verify_orientation::Bool=true,
) where {M<:Manifold,C<:CRS}
    tri_octree = TriangleOctree(
        mesh;
        tolerance_relative=tolerance_relative,
        min_ratio=min_ratio,
        classify_leaves=true,
        verify_orientation=verify_orientation,
    )
    return DensityAwareOctree(tri_octree, boundary_oversampling, placement)
end

function DensityAwareOctree(filepath::String; kwargs...)
    geo = GeoIO.load(filepath)
    return DensityAwareOctree(geo.geometry; kwargs...)
end

"""
    SpacingCriterion{T,S} <: SubdivisionCriterion

Subdivide node octree based on local spacing:
`box_size > alpha * h(center)` with absolute minimum guard.
"""
struct SpacingCriterion{T<:Real,S} <: SubdivisionCriterion
    spacing::S
    alpha::T
    absolute_min::T
end

function SpacingCriterion(
    spacing,
    domain_diagonal::Real;
    alpha::Real=2.0,
    min_ratio::Real=1e-6,
)
    T = Float64
    absolute_min = T(domain_diagonal) * T(min_ratio)
    return SpacingCriterion{T,typeof(spacing)}(spacing, T(alpha), absolute_min)
end

@inline function _spacing_value(::Type{T}, spacing, p::SVector{3,T}) where {T<:Real}
    q = Point(p[1], p[2], p[3])
    return T(ustrip(spacing(q)))
end

function should_subdivide(c::SpacingCriterion{T}, tree, box_idx) where {T<:Real}
    h_box = box_size(tree, box_idx)
    h_box <= c.absolute_min && return false

    center = box_center(tree, box_idx)
    h_local = _spacing_value(T, c.spacing, center)
    h_local <= eps(T) && return false

    return h_box > c.alpha * h_local
end

function can_subdivide(c::SpacingCriterion, tree, box_idx)
    return box_size(tree, box_idx) > c.absolute_min
end

function _build_node_octree(
    triangle_octree::TriangleOctree{M,C,T},
    spacing::AbstractSpacing,
) where {M<:Manifold,C<:CRS,T<:Real}
    bbox_min, bbox_max = bounding_box(triangle_octree.tree)
    root_size = triangle_octree.tree.root_size

    node_tree = SpatialOctree{Int,T}(bbox_min, root_size; initial_capacity=1000)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(spacing, diagonal; alpha=2.0)

    _subdivide_node_octree!(node_tree, 1, criterion, triangle_octree)
    balance_octree!(node_tree, criterion)

    return node_tree
end

function _subdivide_node_octree!(
    node_tree::SpatialOctree{Int,T},
    box_idx::Int,
    criterion::SpacingCriterion,
    triangle_octree::TriangleOctree,
) where {T<:Real}
    should_subdivide(criterion, node_tree, box_idx) || return
    _box_may_contain_interior(node_tree, box_idx, triangle_octree) || return

    subdivide!(node_tree, box_idx)
    children = node_tree.children[box_idx]
    for child_idx in children
        child_idx == 0 && continue
        _subdivide_node_octree!(node_tree, child_idx, criterion, triangle_octree)
    end
    return
end

@inline function _probe_leaf_sign(
    point::SVector{3,T},
    triangle_octree::TriangleOctree,
    tol::T,
) where {T<:Real}
    sd = _compute_signed_distance_octree(
        point,
        triangle_octree.mesh,
        triangle_octree.tree,
    )
    return _leaf_class_from_signed_distance(sd, tol)
end

function _box_may_contain_interior(
    node_tree::SpatialOctree{Int,T},
    box_idx::Int,
    triangle_octree::TriangleOctree,
) where {T<:Real}
    bbox_min, bbox_max = box_bounds(node_tree, box_idx)
    h = box_size(node_tree, box_idx)
    inset = max(T(_CLASSIFICATION_INSET), h * T(_CLASSIFY_TOLERANCE_REL))
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    center = box_center(node_tree, box_idx)
    corners = _inset_corners(bbox_min, bbox_max, inset)

    for probe in (center, corners...)
        cls = _probe_leaf_sign(probe, triangle_octree, tol)
        cls != LEAF_EXTERIOR && return true
    end

    return false
end

function _classify_node_leaves(
    node_tree::SpatialOctree{Int,T},
    triangle_octree::TriangleOctree,
) where {T<:Real}
    n_boxes = node_tree.num_boxes[]
    classification = fill(LEAF_UNKNOWN, n_boxes)

    for leaf_idx in all_leaves(node_tree)
        classification[leaf_idx] = _classify_node_leaf_conservative(
            node_tree,
            leaf_idx,
            triangle_octree,
        )
    end

    return classification
end

function _classify_node_leaf_conservative(
    node_tree::SpatialOctree{Int,T},
    leaf_idx::Int,
    triangle_octree::TriangleOctree,
) where {T<:Real}
    bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
    h = box_size(node_tree, leaf_idx)

    inset = max(T(_CLASSIFICATION_INSET), h * T(_CLASSIFY_TOLERANCE_REL))
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    center = box_center(node_tree, leaf_idx)
    corners = _inset_corners(bbox_min, bbox_max, inset)
    classes = map(p -> _probe_leaf_sign(p, triangle_octree, tol), (center, corners...))

    any(==(LEAF_BOUNDARY), classes) && return LEAF_BOUNDARY
    all(==(LEAF_INTERIOR), classes) && return LEAF_INTERIOR
    all(==(LEAF_EXTERIOR), classes) && return LEAF_EXTERIOR
    return LEAF_BOUNDARY
end

function _collect_classified_node_leaves(
    node_tree::SpatialOctree{Int,T},
    classification::Vector{Int8},
) where {T<:Real}
    interior_leaves = Int[]
    boundary_leaves = Int[]
    interior_volumes = T[]
    boundary_volumes = T[]

    for leaf_idx in all_leaves(node_tree)
        cls = classification[leaf_idx]
        if cls == LEAF_INTERIOR
            push!(interior_leaves, leaf_idx)
            push!(interior_volumes, box_size(node_tree, leaf_idx)^3)
        elseif cls == LEAF_BOUNDARY
            push!(boundary_leaves, leaf_idx)
            push!(boundary_volumes, box_size(node_tree, leaf_idx)^3)
        end
    end

    return interior_leaves, interior_volumes, boundary_leaves, boundary_volumes
end

function _compute_node_leaf_weights(
    leaves::Vector{Int},
    node_tree::SpatialOctree{Int,T},
    spacing::AbstractSpacing,
    volumes::Vector{T},
) where {T<:Real}
    weights = Vector{T}(undef, length(leaves))

    for (i, leaf_idx) in enumerate(leaves)
        center = box_center(node_tree, leaf_idx)
        h_local = _spacing_value(T, spacing, center)
        h_local = max(h_local, sqrt(eps(T)))
        density = inv(h_local^3)
        weights[i] = density * volumes[i]
    end

    return weights
end

function _allocate_counts_by_weight(weights::Vector{T}, total_count::Int) where {T<:Real}
    return _allocate_counts_by_volume(weights, total_count)
end

function _generate_points_in_box(
    bbox_min::SVector{3,T},
    bbox_max::SVector{3,T},
    n::Int,
    placement::Symbol,
) where {T<:Real}
    n <= 0 && return SVector{3,T}[]

    if placement == :random
        return [_rand_point_in_box(bbox_min, bbox_max) for _ in 1:n]
    end

    m = max(1, ceil(Int, cbrt(n)))
    h = (bbox_max - bbox_min) / T(m)
    pts = SVector{3,T}[]
    sizehint!(pts, n)

    for k in 0:(m-1), j in 0:(m-1), i in 0:(m-1)
        length(pts) >= n && break
        cell_min = bbox_min + SVector{3,T}(T(i), T(j), T(k)) .* h

        if placement == :lattice
            push!(pts, cell_min + h / T(2))
        elseif placement == :jittered
            push!(pts, cell_min + rand(SVector{3,T}) .* h)
        else
            throw(ArgumentError("Unknown placement: $placement"))
        end
    end

    return pts
end

# Delegate spacing-free call only for explicit spacing algorithms.
function _discretize_volume(
    cloud::PointCloud{𝔼{3},C},
    spacing::AbstractSpacing,
    alg::DensityAwareOctree;
    max_points=1_000,
) where {C}
    T = Float64

    node_tree = _build_node_octree(alg.triangle_octree, spacing)
    leaf_classification = _classify_node_leaves(node_tree, alg.triangle_octree)

    interior_leaves, interior_volumes, boundary_leaves, boundary_volumes =
        _collect_classified_node_leaves(node_tree, leaf_classification)

    interior_weights = _compute_node_leaf_weights(
        interior_leaves,
        node_tree,
        spacing,
        interior_volumes,
    )
    boundary_weights = _compute_node_leaf_weights(
        boundary_leaves,
        node_tree,
        spacing,
        boundary_volumes,
    )

    total_weight = sum(interior_weights; init=zero(T)) + sum(boundary_weights; init=zero(T))
    total_weight <= zero(T) && return PointVolume(Point{𝔼{3},C}[])

    n_interior = round(Int, max_points * (sum(interior_weights; init=zero(T)) / total_weight))
    n_boundary = max_points - n_interior
    n_boundary_samples = round(Int, n_boundary * alg.boundary_oversampling)

    interior_counts = _allocate_counts_by_weight(interior_weights, n_interior)
    boundary_counts = _allocate_counts_by_weight(boundary_weights, n_boundary_samples)

    raw_points = SVector{3,T}[]
    sizehint!(raw_points, max_points)

    for (leaf_idx, n_leaf) in zip(interior_leaves, interior_counts)
        n_leaf == 0 && continue
        bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
        append!(raw_points, _generate_points_in_box(bbox_min, bbox_max, n_leaf, alg.placement))
    end

    boundary_candidates = SVector{3,T}[]
    sizehint!(boundary_candidates, n_boundary_samples)
    for (leaf_idx, n_leaf) in zip(boundary_leaves, boundary_counts)
        n_leaf == 0 && continue
        bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
        append!(boundary_candidates, _generate_points_in_box(bbox_min, bbox_max, n_leaf, alg.placement))
    end

    for p in boundary_candidates
        if isinside(p, alg.triangle_octree)
            push!(raw_points, p)
            length(raw_points) >= max_points && break
        end
    end

    if length(raw_points) < max_points && !isempty(interior_leaves)
        missing = max_points - length(raw_points)
        w = isempty(interior_weights) ? fill(inv(T(length(interior_leaves))), length(interior_leaves)) : interior_weights
        cumw = cumsum(w)
        totalw = cumw[end]

        for _ in 1:missing
            r = rand(T) * totalw
            idx = clamp(searchsortedfirst(cumw, r), 1, length(interior_leaves))
            leaf_idx = interior_leaves[idx]
            bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
            push!(raw_points, _rand_point_in_box(bbox_min, bbox_max))
        end
    end

    result_points = Point{𝔼{3},C}[]
    sizehint!(result_points, length(raw_points))
    for p in raw_points
        push!(result_points, Point(p[1], p[2], p[3]))
    end

    return PointVolume(result_points)
end
