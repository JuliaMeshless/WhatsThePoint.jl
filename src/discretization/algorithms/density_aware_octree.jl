"""
    DensityAwareOctree <: AbstractNodeGenerationAlgorithm

Spacing-aware volume point generation using two-octree architecture:
- Triangle octree: geometry and robust inside/outside classification
- Node octree: spacing-adaptive volume partitioning for point allocation
"""
struct DensityAwareOctree{M <: Manifold, C <: CRS, T <: Real} <: AbstractNodeGenerationAlgorithm
    triangle_octree::TriangleOctree{M, C, T}
    boundary_oversampling::Float64
    placement::Symbol
end

function DensityAwareOctree(
        triangle_octree::TriangleOctree{M, C, T},
        boundary_oversampling::Real = 2.0,
        placement::Symbol = :random,
    ) where {M <: Manifold, C <: CRS, T <: Real}
    boundary_oversampling > 0 || throw(ArgumentError("boundary_oversampling must be positive"))
    placement in (:random, :jittered, :lattice) ||
        throw(ArgumentError("placement must be one of :random, :jittered, :lattice"))
    return DensityAwareOctree{M, C, T}(triangle_octree, Float64(boundary_oversampling), placement)
end

function DensityAwareOctree(
        mesh::SimpleMesh{M, C};
        boundary_oversampling::Real = 2.0,
        placement::Symbol = :random,
        tolerance_relative::Real = 1.0e-6,
        min_ratio::Real = 1.0e-6,
        verify_orientation::Bool = true,
    ) where {M <: Manifold, C <: CRS}
    tri_octree = TriangleOctree(
        mesh;
        tolerance_relative = tolerance_relative,
        min_ratio = min_ratio,
        classify_leaves = true,
        verify_orientation = verify_orientation,
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
struct SpacingCriterion{T <: Real, S} <: SubdivisionCriterion
    spacing::S
    alpha::T
    absolute_min::T
end

function SpacingCriterion(
        spacing,
        domain_diagonal::Real;
        alpha::Real = 2.0,
        min_ratio::Real = 1.0e-6,
    )
    T = Float64
    absolute_min = T(domain_diagonal) * T(min_ratio)
    return SpacingCriterion{T, typeof(spacing)}(spacing, T(alpha), absolute_min)
end

@inline function _spacing_value(::Type{T}, spacing, p::SVector{3, T}) where {T <: Real}
    q = Point(p[1], p[2], p[3])
    return T(ustrip(spacing(q)))
end

function should_subdivide(c::SpacingCriterion{T}, tree, box_idx) where {T <: Real}
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
        triangle_octree::TriangleOctree{M, C, T},
        spacing::AbstractSpacing,
    ) where {M <: Manifold, C <: CRS, T <: Real}
    bbox_min, bbox_max = bounding_box(triangle_octree.tree)
    root_size = triangle_octree.tree.root_size

    node_tree = SpatialOctree{Int, T}(bbox_min, root_size; initial_capacity = 1000)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(spacing, diagonal; alpha = 2.0)

    _subdivide_node_octree!(node_tree, 1, criterion, triangle_octree)
    balance_octree!(node_tree, criterion)

    return node_tree
end

function _subdivide_node_octree!(
        node_tree::SpatialOctree{Int, T},
        box_idx::Int,
        criterion::SpacingCriterion,
        triangle_octree::TriangleOctree,
    ) where {T <: Real}
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

@inline function _mesh_geometry_query(
        point::SVector{3, T},
        tol::T,
        triangle_octree::TriangleOctree,
    ) where {T <: Real}
    sd = _compute_signed_distance_octree(point, triangle_octree.mesh, triangle_octree.tree)
    return _leaf_class_from_signed_distance(sd, tol)
end

function _box_may_contain_interior(
        node_tree::SpatialOctree{Int, T},
        box_idx::Int,
        triangle_octree::TriangleOctree,
    ) where {T <: Real}
    bbox_min, bbox_max = box_bounds(node_tree, box_idx)
    h = box_size(node_tree, box_idx)
    inset = max(T(_CLASSIFICATION_INSET), h * T(_CLASSIFY_TOLERANCE_REL))
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    center = box_center(node_tree, box_idx)
    corners = _inset_corners(bbox_min, bbox_max, inset)

    for probe in (center, corners...)
        cls = _mesh_geometry_query(probe, tol, triangle_octree)
        cls != LEAF_EXTERIOR && return true
    end

    return false
end

function _classify_node_leaves(
        node_tree::SpatialOctree{Int, T},
        triangle_octree::TriangleOctree,
    ) where {T <: Real}
    mesh_query(point::SVector{3, T}, tol::T) = _mesh_geometry_query(point, tol, triangle_octree)
    return classify_leaves!(node_tree, mesh_query)
end

struct LeafGroup{T <: Real}
    indices::Vector{Int}
    weights::Vector{T}
end

function _collect_weighted_leaves(
        node_tree::SpatialOctree{Int, T},
        classification::Vector{Int8},
        spacing::AbstractSpacing,
    ) where {T <: Real}
    interior_indices = Int[]
    interior_weights = T[]
    near_surface_indices = Int[]
    near_surface_weights = T[]

    for leaf_idx in all_leaves(node_tree)
        cls = classification[leaf_idx]
        if cls == LEAF_INTERIOR || cls == LEAF_BOUNDARY
            center = box_center(node_tree, leaf_idx)
            h_local = max(_spacing_value(T, spacing, center), sqrt(eps(T)))
            volume = box_size(node_tree, leaf_idx)^3
            weight = volume / h_local^3

            if cls == LEAF_INTERIOR
                push!(interior_indices, leaf_idx)
                push!(interior_weights, weight)
            else
                push!(near_surface_indices, leaf_idx)
                push!(near_surface_weights, weight)
            end
        end
    end

    return LeafGroup{T}(interior_indices, interior_weights),
        LeafGroup{T}(near_surface_indices, near_surface_weights)
end

function _generate_interior_points(
        leaf_group::LeafGroup{T},
        node_tree::SpatialOctree{Int, T},
        n_points::Int,
        placement::Symbol,
    ) where {T <: Real}
    counts = _allocate_counts_by_volume(leaf_group.weights, n_points)
    points = SVector{3, T}[]
    sizehint!(points, n_points)

    for (leaf_idx, n_leaf) in zip(leaf_group.indices, counts)
        n_leaf == 0 && continue
        bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
        append!(points, _generate_points_in_box(bbox_min, bbox_max, n_leaf, placement))
    end

    return points
end

function _generate_near_surface_points(
        leaf_group::LeafGroup{T},
        node_tree::SpatialOctree{Int, T},
        triangle_octree::TriangleOctree,
        n_target::Int,
        oversampling::Float64,
        placement::Symbol,
    ) where {T <: Real}
    n_candidates = round(Int, n_target * oversampling)
    counts = _allocate_counts_by_volume(leaf_group.weights, n_candidates)

    candidates = SVector{3, T}[]
    sizehint!(candidates, n_candidates)
    for (leaf_idx, n_leaf) in zip(leaf_group.indices, counts)
        n_leaf == 0 && continue
        bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
        append!(candidates, _generate_points_in_box(bbox_min, bbox_max, n_leaf, placement))
    end

    accepted = SVector{3, T}[]
    sizehint!(accepted, n_target)
    for p in candidates
        isinside(p, triangle_octree) && push!(accepted, p)
        length(accepted) >= n_target && break
    end

    return accepted
end

function _fill_deficit(
        interior_leaves::LeafGroup{T},
        node_tree::SpatialOctree{Int, T},
        n_needed::Int,
    ) where {T <: Real}
    isempty(interior_leaves.indices) && return SVector{3, T}[]

    w = isempty(interior_leaves.weights) ?
        fill(inv(T(length(interior_leaves.indices))), length(interior_leaves.indices)) :
        interior_leaves.weights

    cumw = cumsum(w)
    totalw = cumw[end]

    points = SVector{3, T}[]
    sizehint!(points, n_needed)

    for _ in 1:n_needed
        r = rand(T) * totalw
        idx = clamp(searchsortedfirst(cumw, r), 1, length(interior_leaves.indices))
        leaf_idx = interior_leaves.indices[idx]
        bbox_min, bbox_max = box_bounds(node_tree, leaf_idx)
        push!(points, _rand_point_in_box(bbox_min, bbox_max))
    end

    return points
end

function _generate_points_in_box(
        bbox_min::SVector{3, T},
        bbox_max::SVector{3, T},
        n::Int,
        placement::Symbol,
    ) where {T <: Real}
    n <= 0 && return SVector{3, T}[]

    if placement == :random
        return [_rand_point_in_box(bbox_min, bbox_max) for _ in 1:n]
    end

    m = max(1, ceil(Int, cbrt(n)))
    h = (bbox_max - bbox_min) / T(m)
    pts = SVector{3, T}[]
    sizehint!(pts, n)

    for k in 0:(m - 1), j in 0:(m - 1), i in 0:(m - 1)
        length(pts) >= n && break
        cell_min = bbox_min + SVector{3, T}(T(i), T(j), T(k)) .* h

        if placement == :lattice
            push!(pts, cell_min + h / T(2))
        elseif placement == :jittered
            push!(pts, cell_min + rand(SVector{3, T}) .* h)
        else
            throw(ArgumentError("Unknown placement: $placement"))
        end
    end

    return pts
end

function _discretize_volume(
        cloud::PointCloud{𝔼{3}, C},
        spacing::AbstractSpacing,
        alg::DensityAwareOctree;
        max_points = 1_000,
    ) where {C}
    T = Float64

    # Build and classify node octree
    node_tree = _build_node_octree(alg.triangle_octree, spacing)
    classification = _classify_node_leaves(node_tree, alg.triangle_octree)
    interior, near_surface = _collect_weighted_leaves(node_tree, classification, spacing)

    # Allocate points proportionally to weighted volumes
    total_weight = sum(interior.weights; init = zero(T)) + sum(near_surface.weights; init = zero(T))
    total_weight <= zero(T) && return PointVolume(Point{𝔼{3}, C}[])

    interior_ratio = sum(interior.weights; init = zero(T)) / total_weight
    n_interior = round(Int, max_points * interior_ratio)
    n_near_surface = max_points - n_interior

    # Generate points
    raw_points = SVector{3, T}[]
    sizehint!(raw_points, max_points)

    append!(raw_points, _generate_interior_points(interior, node_tree, n_interior, alg.placement))
    append!(
        raw_points, _generate_near_surface_points(
            near_surface,
            node_tree,
            alg.triangle_octree,
            n_near_surface,
            alg.boundary_oversampling,
            alg.placement,
        )
    )

    # Fill deficit if near-surface rejection left us short
    if length(raw_points) < max_points
        deficit = max_points - length(raw_points)
        append!(raw_points, _fill_deficit(interior, node_tree, deficit))
    end

    # Convert to Point objects
    return PointVolume([Point(p[1], p[2], p[3]) for p in raw_points])
end
