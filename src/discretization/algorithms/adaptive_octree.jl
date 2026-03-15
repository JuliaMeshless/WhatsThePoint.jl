"""
    AdaptiveOctree <: AbstractNodeGenerationAlgorithm

Octree-based volume discretization with spacing-aware refinement.

Uses dual octrees: triangle octree (geometry) + node octree (spacing-adaptive points).

# Key Parameters
- `min_ratio`: Triangle octree resolution (default: auto from mesh complexity)
- `node_min_ratio`: Node octree resolution (default: auto from `spacing` if provided)
- `alpha`: Subdivision aggressiveness, `h_box ≤ alpha * h_spacing` (default: 2.0, use 1.0 for fine boundary layers)
- `placement`: `:random`, `:jittered`, or `:lattice` (default: `:random`)
- `boundary_oversampling`: Oversampling near boundaries (default: 2.0)

# Examples
```julia
# Automatic (recommended)
alg = AdaptiveOctree(mesh; spacing, alpha=1.0)
cloud = discretize(boundary, spacing; alg, max_points=100_000)

# Manual geometry resolution
alg = AdaptiveOctree(mesh; min_ratio=1e-3, spacing, alpha=1.0)
```
"""
struct AdaptiveOctree{M <: Manifold, C <: CRS, T <: Real} <: AbstractNodeGenerationAlgorithm
    triangle_octree::TriangleOctree{M, C, T}
    boundary_oversampling::Float64
    placement::Symbol
    alpha::Float64  # Subdivision aggressiveness: boxes are at most alpha*h_local
    node_min_ratio::T  # Node octree minimum box size ratio (can be much finer than triangle octree)
end

# Constructors
function AdaptiveOctree(
        triangle_octree::TriangleOctree{M, C, T};
        node_min_ratio::Union{Nothing, Real} = nothing,
        boundary_oversampling::Real = 2.0,
        placement::Symbol = :random,
        alpha::Real = 2.0,
    ) where {M, C, T}
    boundary_oversampling > 0 || throw(ArgumentError("boundary_oversampling must be positive"))
    placement in (:random, :jittered, :lattice) ||
        throw(ArgumentError("placement must be :random, :jittered, or :lattice"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))

    # Default: use triangle octree's min_ratio
    node_ratio = isnothing(node_min_ratio) ? triangle_octree.tree.min_ratio : T(node_min_ratio)

    return AdaptiveOctree{M, C, T}(
        triangle_octree,
        Float64(boundary_oversampling),
        placement,
        Float64(alpha),
        node_ratio
    )
end

function AdaptiveOctree(
        mesh::SimpleMesh{M, C};
        spacing::Union{Nothing, AbstractSpacing} = nothing,
        min_ratio::Union{Nothing, Real} = nothing,
        node_min_ratio::Union{Nothing, Real} = nothing,
        tolerance_relative::Real = 1.0e-6,
        boundary_oversampling::Real = 2.0,
        placement::Symbol = :random,
        alpha::Real = 2.0,
        verify_orientation::Bool = true,
    ) where {M, C}
    T = Float64

    # Triangle octree: geometry-based or user override
    tri_min_ratio = isnothing(min_ratio) ? _auto_min_ratio(T, mesh) : T(min_ratio)
    triangle_octree = TriangleOctree(
        mesh;
        tolerance_relative,
        min_ratio = tri_min_ratio,
        classify_leaves = true,
        verify_orientation,
    )

    # Node octree: spacing-aware automatic or user override
    node_ratio = if !isnothing(node_min_ratio)
        # User provided explicit node_min_ratio
        T(node_min_ratio)
    elseif !isnothing(spacing)
        # Compute from spacing
        h_min = _extract_min_spacing(spacing)
        if !isnothing(h_min)
            # Compute domain size (strip units to get Float64)
            bbox = Meshes.boundingbox(mesh)
            extents = bbox.max - bbox.min
            # Get maximum extent value, stripping units
            max_extent = maximum([T(ustrip(extents[i])) for i in 1:length(extents)])

            # Allow node octree to subdivide alpha times finer than minimum spacing
            # This ensures we can properly resolve the spacing function
            T(h_min / T(alpha)) / max_extent
        else
            # Unknown spacing type, fall back to triangle octree ratio
            tri_min_ratio
        end
    else
        # No spacing hint, use triangle octree ratio
        tri_min_ratio
    end

    return AdaptiveOctree{M, C, T}(
        triangle_octree,
        Float64(boundary_oversampling),
        placement,
        Float64(alpha),
        node_ratio
    )
end

AdaptiveOctree(filepath::String; kwargs...) = AdaptiveOctree(GeoIO.load(filepath).geometry; kwargs...)

# ============================================================================
# Helper functions
# ============================================================================

"""
    _auto_min_ratio(::Type{T}, mesh) where {T}

Default triangle octree resolution: `1 / (4 * cbrt(n_triangles))`.

Factor of 4 (vs. 2) ensures accurate geometry in high-curvature regions.
Override with explicit `min_ratio` parameter if needed.
"""
function _auto_min_ratio(::Type{T}, mesh::SimpleMesh) where {T}
    n = Meshes.nelements(mesh)
    # Factor of 4 (rather than 2) for 2× finer subdivision
    # This ensures accurate geometry representation in high-curvature regions
    return inv(T(4) * cbrt(T(n)))
end

"""
    _extract_min_spacing(spacing)

Extract minimum spacing value from spacing object (field access, no sampling).
"""
function _extract_min_spacing(spacing::ConstantSpacing)
    return Float64(ustrip(spacing.Δx))
end

function _extract_min_spacing(spacing::BoundaryLayerSpacing)
    return Float64(ustrip(spacing.at_wall))
end

# Fallback for other spacing types: return nothing to indicate unknown
_extract_min_spacing(::AbstractSpacing) = nothing

"""
    _rand_point_in_box(bbox_min, bbox_max)

Generate a random point uniformly distributed in a bounding box.
"""
@inline function _rand_point_in_box(bbox_min::SVector{3, T}, bbox_max::SVector{3, T}) where {T}
    return bbox_min + rand(SVector{3, T}) .* (bbox_max - bbox_min)
end

"""
    _allocate_counts_by_volume(volumes, total_count; ensure_one=false, dither=true)

Proportionally allocate counts to volumes.
- `dither=true`: Probabilistic rounding (reduces clustering)
- `dither=false`: Deterministic largest-remainder
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

Generate `n` points: `:random`, `:jittered` (stratified), or `:lattice` (grid).
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

@inline function _mesh_geometry_query(pt::SVector{3, T}, tol, octree) where {T}
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
    return
end

function _build_node_octree(triangle_octree, spacing, alpha, node_min_ratio)
    T = Float64
    bbox_min, bbox_max = bounding_box(triangle_octree.tree)
    node_tree = SpatialOctree{Int, T}(bbox_min, triangle_octree.tree.root_size; initial_capacity = 1000)

    diagonal = norm(bbox_max - bbox_min)
    criterion = SpacingCriterion(spacing, diagonal; alpha, min_ratio = node_min_ratio)

    _subdivide_node_octree!(node_tree, 1, criterion, triangle_octree)
    balance_octree!(node_tree, criterion)

    return node_tree
end

function _classify_node_leaves(node_tree, triangle_octree)
    # Reuse existing classification infrastructure from spatial_octree.jl
    query(pt, tol) = _mesh_geometry_query(pt, tol, triangle_octree)
    return classify_leaves!(node_tree, query)
end

# ============================================================================
# Point generation
# ============================================================================

struct LeafGroup{T <: Real}
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

    points = SVector{3, T}[]
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
        cloud::PointCloud{𝔼{3}, C},
        spacing::AbstractSpacing,
        alg::AdaptiveOctree;
        max_points = 1_000,
    ) where {C}
    isnothing(alg.triangle_octree.leaf_classification) &&
        error("TriangleOctree must be built with classify_leaves=true")

    T = Float64

    # Build and classify node octree
    print("  Building node octree...")
    t1 = @elapsed node_tree = _build_node_octree(alg.triangle_octree, spacing, alg.alpha, alg.node_min_ratio)
    n_leaves = length(collect(all_leaves(node_tree)))
    println(" done ($(n_leaves) leaves, $(round(t1, digits = 2))s)")

    print("  Classifying node octree leaves...")
    t2 = @elapsed classification = _classify_node_leaves(node_tree, alg.triangle_octree)
    println(" done ($(round(t2, digits = 2))s)")

    print("  Collecting weighted leaves...")
    t3 = @elapsed interior, near_surface = _collect_weighted_leaves(node_tree, classification, spacing)
    println(" done (interior: $(length(interior.indices)), near_surface: $(length(near_surface.indices)), $(round(t3, digits = 2))s)")

    # Allocate points proportionally to weighted volumes
    total_weight = sum(interior.weights; init = zero(T)) + sum(near_surface.weights; init = zero(T))
    total_weight <= zero(T) && return PointVolume(Point{𝔼{3}, C}[])

    interior_ratio = sum(interior.weights; init = zero(T)) / total_weight
    n_interior = round(Int, max_points * interior_ratio)
    n_near_surface = max_points - n_interior
    println("  Allocating $n_interior interior + $n_near_surface near-surface points")

    # Generate points
    raw_points = SVector{3, T}[]
    sizehint!(raw_points, max_points)

    print("  Generating interior points...")
    t4 = @elapsed interior_points = _generate_from_leaves(interior, node_tree, n_interior, alg.placement)
    println(" done ($(length(interior_points)) points, $(round(t4, digits = 2))s)")
    append!(raw_points, interior_points)

    n_candidates = round(Int, n_near_surface * alg.boundary_oversampling)
    print("  Generating near-surface candidates...")
    t5 = @elapsed candidates = _generate_from_leaves(near_surface, node_tree, n_candidates, alg.placement)
    println(" done ($(length(candidates)) candidates, $(round(t5, digits = 2))s)")

    # Filter candidates
    print("  Filtering candidates (isinside check)...")
    n_accepted = 0
    t6 = @elapsed begin
        for p in candidates
            if isinside(p, alg.triangle_octree)
                push!(raw_points, p)
                n_accepted += 1
            end
            length(raw_points) >= n_interior + n_near_surface && break
        end
    end
    println(" done ($n_accepted accepted, $(round(t6, digits = 2))s)")

    # Fill deficit if near-surface rejection left us short
    if length(raw_points) < max_points
        deficit = max_points - length(raw_points)
        println("  Filling deficit of $deficit points...")
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
