"""
    AdaptiveOctree <: AbstractNodeGenerationAlgorithm

Octree-based volume point generation with spacing-aware refinement.

Uses two octrees:
- **Triangle octree**: Geometry representation and inside/outside classification
- **Node octree**: Spacing-adaptive spatial partitioning for point allocation

# Constructors
```julia
# From mesh (recommended)
AdaptiveOctree(mesh::SimpleMesh)
AdaptiveOctree(mesh; min_ratio=auto, placement=:random, boundary_oversampling=2.0)

# From file
AdaptiveOctree("model.stl"; placement=:jittered)

# From pre-built octree
AdaptiveOctree(octree::TriangleOctree; placement=:random)
```

# Parameters
- `placement::Symbol` - Point placement: `:random` (default), `:jittered`, or `:lattice`
- `boundary_oversampling::Float64` - Oversampling factor for boundary regions (default: 2.0)
- `alpha::Float64` - Subdivision aggressiveness: boxes are at most `alpha*h_local` (default: 2.0, use 1.0-1.5 for fine boundary layers)

# Examples
```julia
# Uniform distribution
alg = AdaptiveOctree(mesh)
cloud = discretize(boundary, ConstantSpacing(1m); alg, max_points=100_000)

# Adaptive boundary layer
spacing = BoundaryLayerSpacing(points(boundary); at_wall=0.1m, bulk=2.0m, layer_thickness=5.0m)
alg = AdaptiveOctree(mesh; placement=:jittered)
cloud = discretize(boundary, spacing; alg, max_points=100_000)
```
"""
struct AdaptiveOctree{M<:Manifold,C<:CRS,T<:Real} <: AbstractNodeGenerationAlgorithm
    triangle_octree::TriangleOctree{M,C,T}
    boundary_oversampling::Float64
    placement::Symbol
    alpha::Float64  # Subdivision aggressiveness: boxes are at most alpha*h_local
end

# Constructors
function AdaptiveOctree(
    triangle_octree::TriangleOctree{M,C,T};
    boundary_oversampling::Real=2.0,
    placement::Symbol=:random,
    alpha::Real=2.0,
) where {M,C,T}
    boundary_oversampling > 0 || throw(ArgumentError("boundary_oversampling must be positive"))
    placement in (:random, :jittered, :lattice) ||
        throw(ArgumentError("placement must be :random, :jittered, or :lattice"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    return AdaptiveOctree{M,C,T}(triangle_octree, Float64(boundary_oversampling), placement, Float64(alpha))
end

function AdaptiveOctree(
    mesh::SimpleMesh{M,C};
    min_ratio=nothing,
    tolerance_relative::Real=1.0e-6,
    boundary_oversampling::Real=2.0,
    placement::Symbol=:random,
    alpha::Real=2.0,
    verify_orientation::Bool=true,
) where {M,C}
    T = Float64
    min_ratio_val = isnothing(min_ratio) ? _auto_min_ratio(T, mesh) : T(min_ratio)
    triangle_octree = TriangleOctree(
        mesh;
        tolerance_relative,
        min_ratio=min_ratio_val,
        classify_leaves=true,
        verify_orientation,
    )
    return AdaptiveOctree(triangle_octree; boundary_oversampling, placement, alpha)
end

AdaptiveOctree(filepath::String; kwargs...) = AdaptiveOctree(GeoIO.load(filepath).geometry; kwargs...)

# ============================================================================
# Helper functions
# ============================================================================

"""
    _auto_min_ratio(::Type{T}, mesh) where {T}

Compute a default `min_ratio` for octree construction based on triangle count.
Heuristic: `1 / (2 * cbrt(n_triangles))`
"""
function _auto_min_ratio(::Type{T}, mesh::SimpleMesh) where {T}
    n = Meshes.nelements(mesh)
    return inv(T(2) * cbrt(T(n)))
end

"""
    _rand_point_in_box(bbox_min, bbox_max)

Generate a random point uniformly distributed in a bounding box.
"""
@inline function _rand_point_in_box(bbox_min::SVector{3, T}, bbox_max::SVector{3, T}) where {T}
    return bbox_min + rand(SVector{3, T}) .* (bbox_max - bbox_min)
end

"""
    _allocate_counts_by_volume(volumes, total_count; ensure_one=false, dither=true)

Allocate integer counts across leaves proportionally to volumes.

When `dither=true` (default), uses probabilistic rounding to reduce clustering
when total_count < length(volumes). When `dither=false`, uses deterministic
largest-remainder rounding.
"""
function _allocate_counts_by_volume(
    volumes::Vector{T},
    total_count::Int;
    ensure_one::Bool = false,
    dither::Bool = true,
) where {T <: Real}
    n = length(volumes)
    n == 0 && return Int[]
    total_count <= 0 && return zeros(Int, n)

    total_volume = sum(volumes; init = zero(T))
    weights = total_volume > zero(T) ? volumes ./ total_volume : fill(inv(T(n)), n)

    counts = zeros(Int, n)
    remaining = total_count

    if ensure_one && total_count >= n
        counts .= 1
        remaining -= n
    end

    remaining <= 0 && return counts

    expected = remaining .* weights
    base = floor.(Int, expected)
    counts .+= base

    leftover = remaining - sum(base)
    if leftover > 0
        frac = expected .- base

        if dither
            # Probabilistic rounding: each fractional part becomes a probability
            # This spreads points more uniformly when total_count << n
            total_frac = sum(frac)
            cumulative = cumsum(frac)

            for _ in 1:leftover
                r = rand(T) * total_frac
                idx = searchsortedfirst(cumulative, r)
                idx = clamp(idx, 1, n)
                counts[idx] += 1
            end
        else
            # Deterministic largest-remainder
            idxs = sortperm(frac; rev = true)
            for i in 1:leftover
                counts[idxs[i]] += 1
            end
        end
    end

    return counts
end

"""
    _generate_points_in_box(bbox_min, bbox_max, n, placement)

Generate `n` points in a bounding box using the specified placement strategy.

Placement modes:
- `:random` - Pure random sampling
- `:jittered` - Stratified random (regular grid with random jitter)
- `:lattice` - Regular grid centers
"""
function _generate_points_in_box(
    bbox_min::SVector{3, T},
    bbox_max::SVector{3, T},
    n::Int,
    placement::Symbol,
) where {T <: Real}
    n <= 0 && return SVector{3, T}[]
    placement == :random && return [_rand_point_in_box(bbox_min, bbox_max) for _ in 1:n]

    # For :jittered and :lattice, use regular grid structure
    m = max(1, ceil(Int, cbrt(n)))
    h = (bbox_max - bbox_min) / T(m)
    pts = SVector{3, T}[]
    sizehint!(pts, n)

    for k in 0:(m - 1), j in 0:(m - 1), i in 0:(m - 1)
        length(pts) >= n && break
        cell_min = bbox_min + SVector{3, T}(T(i), T(j), T(k)) .* h
        pt = placement == :lattice ? cell_min + h / T(2) : cell_min + rand(SVector{3, T}) .* h
        push!(pts, pt)
    end

    return pts
end

# ============================================================================
# Node octree construction (spacing-aware subdivision)
# ============================================================================

struct SpacingCriterion{T<:Real,S} <: SubdivisionCriterion
    spacing::S
    alpha::T
    absolute_min::T
end

function SpacingCriterion(spacing, diagonal; alpha=2.0, min_ratio=1.0e-6)
    T = Float64
    return SpacingCriterion{T,typeof(spacing)}(spacing, T(alpha), T(diagonal) * T(min_ratio))
end

@inline function _spacing_value(::Type{T}, spacing, p::SVector{3,T}) where {T}
    return T(ustrip(spacing(Point(p[1], p[2], p[3]))))
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

@inline function _mesh_geometry_query(pt::SVector{3,T}, tol, octree) where {T}
    sd = _compute_signed_distance_octree(pt, octree.mesh, octree.tree)
    return _leaf_class_from_signed_distance(sd, tol)
end

function _box_may_contain_interior(node_tree, box_idx, triangle_octree)
    T = Float64
    bbox_min, bbox_max = box_bounds(node_tree, box_idx)
    h = box_size(node_tree, box_idx)
    inset = max(T(_CLASSIFICATION_INSET), h * T(_CLASSIFY_TOLERANCE_REL))
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    for pt in (box_center(node_tree, box_idx), _inset_corners(bbox_min, bbox_max, inset)...)
        _mesh_geometry_query(pt, tol, triangle_octree) != LEAF_EXTERIOR && return true
    end
    return false
end

function _subdivide_node_octree!(node_tree, box_idx, criterion, triangle_octree)
    should_subdivide(criterion, node_tree, box_idx) || return
    _box_may_contain_interior(node_tree, box_idx, triangle_octree) || return

    subdivide!(node_tree, box_idx)
    for child_idx in node_tree.children[box_idx]
        child_idx == 0 && continue
        _subdivide_node_octree!(node_tree, child_idx, criterion, triangle_octree)
    end
end

function _build_node_octree(triangle_octree, spacing, alpha)
    T = Float64
    bbox_min, bbox_max = bounding_box(triangle_octree.tree)
    node_tree = SpatialOctree{Int,T}(bbox_min, triangle_octree.tree.root_size; initial_capacity=1000)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(spacing, diagonal; alpha)

    _subdivide_node_octree!(node_tree, 1, criterion, triangle_octree)
    balance_octree!(node_tree, criterion)

    return node_tree
end

function _classify_node_leaves(node_tree, triangle_octree)
    query(pt::SVector{3,Float64}, tol::Float64) = _mesh_geometry_query(pt, tol, triangle_octree)
    return classify_leaves!(node_tree, query)
end

# ============================================================================
# Point generation
# ============================================================================

struct LeafGroup{T<:Real}
    indices::Vector{Int}
    weights::Vector{T}
end

function _collect_weighted_leaves(node_tree, classification, spacing)
    T = Float64
    interior, near_surface = LeafGroup(Int[], T[]), LeafGroup(Int[], T[])

    for leaf_idx in all_leaves(node_tree)
        cls = classification[leaf_idx]
        if cls == LEAF_INTERIOR || cls == LEAF_BOUNDARY
            h_local = max(_spacing_value(T, spacing, box_center(node_tree, leaf_idx)), sqrt(eps(T)))
            weight = box_size(node_tree, leaf_idx)^3 / h_local^3

            group = cls == LEAF_INTERIOR ? interior : near_surface
            push!(group.indices, leaf_idx)
            push!(group.weights, weight)
        end
    end

    return interior, near_surface
end

function _generate_from_leaves(leaves::LeafGroup, tree, n_points, placement)
    T = Float64
    counts = _allocate_counts_by_volume(leaves.weights, n_points)

    # DEBUG: Check allocation distribution
    nonzero_counts = count(>(0), counts)
    max_count = isempty(counts) ? 0 : maximum(counts)
    println("DEBUG: _generate_from_leaves - $(length(leaves.indices)) leaves, $n_points points -> $nonzero_counts leaves with points (max: $max_count per leaf)")

    points = SVector{3,T}[]
    sizehint!(points, n_points)

    for (leaf_idx, n) in zip(leaves.indices, counts)
        n == 0 && continue
        bbox_min, bbox_max = box_bounds(tree, leaf_idx)
        append!(points, _generate_points_in_box(bbox_min, bbox_max, n, placement))
    end

    return points
end

# ============================================================================
# Main discretization interface
# ============================================================================

function _discretize_volume(
    cloud::PointCloud{𝔼{3},C},
    spacing::AbstractSpacing,
    alg::AdaptiveOctree;
    max_points=1_000,
) where {C}
    isnothing(alg.triangle_octree.leaf_classification) &&
        error("TriangleOctree must be built with classify_leaves=true")

    T = Float64

    # Build and classify node octree
    node_tree = _build_node_octree(alg.triangle_octree, spacing, alg.alpha)
    classification = _classify_node_leaves(node_tree, alg.triangle_octree)
    interior, near_surface = _collect_weighted_leaves(node_tree, classification, spacing)

    # Allocate points proportionally to weighted volumes
    total_weight = sum(interior.weights; init=zero(T)) + sum(near_surface.weights; init=zero(T))
    total_weight <= zero(T) && return PointVolume(Point{𝔼{3},C}[])

    interior_ratio = sum(interior.weights; init=zero(T)) / total_weight
    n_interior = round(Int, max_points * interior_ratio)
    n_near_surface = max_points - n_interior

    # DEBUG: Print weight statistics
    println("=== DEBUG: Weight Distribution ===")
    println("Interior leaves: ", length(interior.indices))
    println("Near-surface leaves: ", length(near_surface.indices))
    println("Interior weight range: ", isempty(interior.weights) ? "empty" : "$(minimum(interior.weights)) to $(maximum(interior.weights))")
    println("Near-surface weight range: ", isempty(near_surface.weights) ? "empty" : "$(minimum(near_surface.weights)) to $(maximum(near_surface.weights))")
    println("Points allocated - interior: $n_interior, near_surface: $n_near_surface")

    # Generate points
    raw_points = SVector{3, T}[]
    sizehint!(raw_points, max_points)

    interior_points = _generate_from_leaves(interior, node_tree, n_interior, alg.placement)
    println("DEBUG: Generated $(length(interior_points)) interior points")
    append!(raw_points, interior_points)

    n_candidates = round(Int, n_near_surface * alg.boundary_oversampling)
    candidates = _generate_from_leaves(near_surface, node_tree, n_candidates, alg.placement)
    println("DEBUG: Generated $(length(candidates)) near-surface candidates (target: $n_candidates)")

    # Filter candidates (fast operation, no progress bar needed)
    n_accepted = 0
    for p in candidates
        if isinside(p, alg.triangle_octree)
            push!(raw_points, p)
            n_accepted += 1
        end
        length(raw_points) >= n_interior + n_near_surface && break
    end
    println("DEBUG: Accepted $n_accepted near-surface points (target: $n_near_surface)")

    # Fill deficit if near-surface rejection left us short
    if length(raw_points) < max_points
        deficit = max_points - length(raw_points)
        cumw = isempty(interior.weights) ? T[] : cumsum(interior.weights)
        totalw = isempty(cumw) ? zero(T) : cumw[end]

        for _ in 1:deficit
            if totalw > zero(T)
                idx = clamp(searchsortedfirst(cumw, rand(T) * totalw), 1, length(interior.indices))
                bbox_min, bbox_max = box_bounds(node_tree, interior.indices[idx])
                push!(raw_points, _rand_point_in_box(bbox_min, bbox_max))
            end
        end
    end

    # Convert to Point objects
    return PointVolume([Point(pt[1], pt[2], pt[3]) for pt in raw_points])
end
