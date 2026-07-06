# ======================================================================
# Public API
# ======================================================================

"""
    repel(cloud::PointCloud, spacing; kwargs...) -> PointCloud

Optimize the point distribution via node repulsion. Only volume
points move; boundary points form a fixed wall. Points pushed outside the
domain are discarded (`isinside` filter). Returns a new cloud with `NoTopology`.

Each iteration rebuilds the k-NN graph from the *current* positions, computes
the repel force on every point, and moves it by the adaptive step
`α_i = clamp(1/|F_i|, α_min, α_max)` scaled by the local spacing and capped at
one spacing unit. Convergence is the force norm `max_i(|F_i|·s_i)`, which
vanishes at equilibrium.

# Keywords
- `force_model = ClippedSpacingForce(β)`: force law, any
  [`RepelForceModel`](@ref); the default is repulsive below `r = s` and zero
  beyond, so a cloud that already satisfies the Poisson-disk criterion is
  preserved rather than re-packed. `β = 0.2` feeds the default and is ignored
  when `force_model` is passed explicitly.
- `α`, `α_min`: step-size bounds. Defaults: `α = 0.05·min(spacing)`, `α_min = α/100`.
- `k = 21`: neighborhood size.
- `max_iters = 1000`, `tol = 1e-6`: iteration and convergence limits.
- `cv_target = 0.0`: quality-based stop — end the relaxation once the movable
  points' d_NN/s coefficient of variation drops to this value (read off the
  sweep's nearest-neighbor data, no extra cost). The natural setting is the
  raw quality of the direct generation pipeline (`≈ 0.07` on the cavity):
  relaxing past the quality a re-seed would give is wasted budget. The stop
  returns the configuration the measurement describes (the pre-sweep
  snapshot), so a cloud already at target comes back unchanged. Off when `0`.
- `stall_after = 50`: stop when that same CV has not improved by ≥0.1 % for
  this many consecutive iterations. The force residual of a saturated
  repulsion-only packing plateaus at a nonzero value instead of reaching
  `tol`, so `cv_target`/`stall_after` are the practical stops for the default
  force; CV keeps creeping down for hundreds of iterations, making
  `stall_after` the backstop (on by default so default runs terminate instead
  of burning `max_iters`) and `cv_target` the primary. Pass `0` to disable
  and rely on `tol`/`max_iters` alone.
- `rebuild_every = 1`: iterations between k-NN graph rebuilds (larger = cheaper,
  staler).
- `kick_after = 0`: if the closest pair freezes at the same `r/s` for this many
  iterations (a balanced standoff), kick one point by `0.1·s` in a random
  direction to break the symmetry. Off when `0`; `10`–`20` is reasonable.
- `cull_ratio = 0.0`: after relaxation, drop near-duplicates closer than
  `cull_ratio·spacing` to a kept point. A safety net — a healthy relaxation
  leaves nothing to cull, so a `@warn` is emitted whenever it fires.
- `convergence`: pass a `Float64[]` to collect the per-iteration force norm.
- `trace`: pass a `NamedTuple[]` to record the closest pair each iteration
  (global boundary-then-volume indices, measured on that iteration's snapshot).
"""
function repel(
        cloud::PointCloud{𝔼{N}, C},
        spacing;
        β = 0.2,
        force_model::RepelForceModel = ClippedSpacingForce(β),
        α = minimum(spacing.(to(cloud))) * 0.05,
        α_min = α / 100,
        k = 21,
        max_iters = 1000,
        tol = 1.0e-6,
        rebuild_every::Int = 1,
        cull_ratio::Real = 0.0,
        kick_after::Int = 0,
        stall_after::Int = 50,
        cv_target::Real = 0.0,
        convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
        trace::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    ) where {N, C <: CRS}
    rebuild_every >= 1 || throw(ArgumentError("rebuild_every must be ≥ 1"))
    bnd_p = points(boundary(cloud))
    n_bnd = length(bnd_p)
    p = copy(volume(cloud).points)
    p_old = copy(p)
    # Search snapshot: fixed boundary head (the wall) + movable volume tail.
    snap = vcat(bnd_p, p)

    conv = _relax!(
        p, p_old, snap, spacing, force_model, (id, xi, x_proposed) -> x_proposed;
        n_fixed = n_bnd, n_protected = n_bnd,
        α_lo = ustrip(α_min), α_max = ustrip(α),
        k, max_iters, tol, rebuild_every, kick_after, stall_after, cv_target, trace,
    )
    isnothing(convergence) || append!(convergence, conv)

    survivors = filter(x -> isinside(x, cloud), p)
    if cull_ratio > 0 && !isempty(survivors)
        survivors = survivors[_cull(survivors, spacing, cull_ratio)]
    end
    return PointCloud(boundary(cloud), PointVolume(survivors), NoTopology())
end

"""
    repel(cloud::PointCloud, spacing, octree::TriangleOctree; kwargs...) -> PointCloud

Node repulsion with boundary projection: *all* points move, boundary points are
re-projected onto the mesh surface every iteration, and volume points that
escape the domain bounce back — or stick to the surface (see `deposit_ratio`).
Step size, convergence, and the shared keywords are as in the method without
`octree`.

The returned boundary is a single surface named `:boundary` (use
`split_surface!` to re-establish surface distinctions); topology is `NoTopology`.

# Additional keywords
- `deposit_ratio = 0.0`: when `> 0`, an escaped volume point is *deposited* —
  projected onto the nearest triangle and converted into a boundary point,
  accepted only if no boundary point already lies within
  `deposit_ratio·spacing` of the landing site. Surface sampling then emerges
  from volume containment instead of the mesh tessellation; the acceptance test
  keeps the deposited density self-limiting (conversion is one-way). Deposited
  points carry the landing triangle's normal and `spacing²` as area.
  `0.5`–`0.7` is reasonable.
- `cull_ratio = 0.0`: as in the volume-only method; here it also targets
  boundary pairs that the deterministic projection parks on a shared
  edge/vertex.
"""
function repel(
        cloud::PointCloud{𝔼{3}, C},
        spacing,
        octree::TriangleOctree;
        β = 0.2,
        force_model::RepelForceModel = ClippedSpacingForce(β),
        α = minimum(spacing.(to(cloud))) * 0.05,
        α_min = α / 100,
        k = 21,
        max_iters = 1000,
        tol = 1.0e-6,
        rebuild_every::Int = 1,
        cull_ratio::Real = 0.0,
        kick_after::Int = 0,
        stall_after::Int = 50,
        cv_target::Real = 0.0,
        deposit_ratio::Real = 0.0,
        convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
        trace::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    ) where {C <: CRS}
    rebuild_every >= 1 || throw(ArgumentError("rebuild_every must be ≥ 1"))
    deposit_ratio >= 0 || throw(ArgumentError("deposit_ratio must be ≥ 0"))
    all_p = points(cloud)
    npoints = length(all_p)
    n_boundary = length(boundary(cloud))
    p, p_old, snap = copy(all_p), copy(all_p), copy(all_p)

    len_unit = Unitful.unit(Meshes.to(first(all_p))[1])
    offset_dist = 1.0e-6 * norm(octree.mesh_bbox_max - octree.mesh_bbox_min)
    tri_indices = zeros(Int, npoints)
    # Mutable membership: deposition converts volume points into boundary
    # points. Vector{Bool} (not BitVector) — per-slot writes from tmap! tasks
    # are safe.
    is_bnd = [id <= n_boundary for id in 1:npoints]
    escaped = fill(false, npoints)

    constrain = (id, xi, x_proposed) -> _constrain_octree(
        id, xi, x_proposed, is_bnd, escaped, tri_indices, octree, offset_dist, len_unit
    )
    deposit! = deposit_ratio <= 0 ? nothing : (p_cur, tree, iter) -> begin
            n = _deposit_escaped!(
                p_cur, tree, min(k, length(p_cur)), escaped, is_bnd, tri_indices, octree,
                spacing, deposit_ratio, offset_dist, len_unit,
            )
            n > 0 && @debug "Deposited $n escaped point(s) onto the boundary at iteration $iter"
            return nothing
        end

    conv = _relax!(
        p, p_old, snap, spacing, force_model, constrain;
        n_fixed = 0, n_protected = n_boundary,
        α_lo = ustrip(α_min), α_max = ustrip(α),
        k, max_iters, tol, rebuild_every, kick_after, stall_after, cv_target,
        trace, deposit!,
    )
    isnothing(convergence) || append!(convergence, conv)

    keep = cull_ratio > 0 ? _cull(p, spacing, cull_ratio) : trues(npoints)
    return _reconstruct_cloud(cloud, p, tri_indices, is_bnd, n_boundary, octree, spacing, keep)
end

# ======================================================================
# Core relaxation loop
# ======================================================================

"""
    _relax!(p, p_old, snap, spacing, force_model, constrain; kwargs...) -> Vector{Float64}

Shared relaxation loop behind both `repel` methods. `p` holds the movable
points (updated in place); `snap` is the search snapshot whose first `n_fixed`
entries are static and whose tail mirrors `p`, refreshed every `rebuild_every`
iterations. `constrain(id, xi, x_proposed)` maps a proposed position to the
final one (identity, or the octree wall rule). `deposit!(p, method, i)`, when
given, runs serially after each sweep. Kick targets prefer indices past
`n_protected`. `cv_target > 0` and `stall_after > 0` add the
quality-based stops: end when the movable points' d_NN/s CV (read off the
sweep's nearest-neighbor data, no extra search) reaches the target, or has
not improved for that many iterations. Returns the force-norm convergence
history `max_i(|F_i|·s_i)`.
"""
function _relax!(
        p, p_old, snap, spacing, force_model, constrain;
        n_fixed, n_protected, α_lo, α_max, k, max_iters, tol, rebuild_every,
        kick_after, trace, stall_after = 0, cv_target = 0.0, (deposit!) = nothing,
    )
    n_move = length(p)
    kk = min(k, length(snap))
    spacings = ustrip.(spacing.(snap))
    len_unit = Unitful.unit(Meshes.to(first(snap))[1])
    # Force-norm convergence data: at equilibrium forces vanish, so this detects
    # convergence earlier than displacement (which can be small mid-oscillation).
    forces = zeros(n_move)
    # Per-point nearest neighbor, recorded during the sweep — gives the closest
    # pair for trace/kick without an extra search.
    nn_dist = fill(Inf, n_move)
    nn_id = zeros(Int, n_move)
    # Raw-coordinate kd-tree queried via in-place knn! into task-local buffers:
    # the per-query allocations of the generic searchdists path were 66% of the
    # loop's allocations (the tree rebuild itself is only ~3% of its time).
    # Ref so the per-iteration rebuild does not rebind a captured variable
    # (OhMyThreads' tmap! rejects boxed closure captures).
    coords = _raw_point.(snap)
    tree_ref = Ref(KDTree(coords))
    buffers = TaskLocalValue{Tuple{Vector{Int}, Vector{Float64}}}(
        () -> (Vector{Int}(undef, kk), Vector{Float64}(undef, kk))
    )
    kick_state = (; pair = (0, 0), rs = Inf, count = 0)
    best_cv = Inf
    last_cv_improvement = 0

    conv = Float64[]
    i = 1
    while i <= max_iters
        p_old .= p
        if (i - 1) % rebuild_every == 0
            @views snap[(n_fixed + 1):end] .= p
            @views coords[(n_fixed + 1):end] .= _raw_point.(p)
            tree_ref[] = KDTree(coords)
        end
        # Jacobi-style sweep: every force is evaluated against the same frozen
        # snapshot.
        tmap!(p, 1:n_move) do id
            xi = p_old[id]
            ids, dists = buffers[]
            knn!(ids, dists, tree_ref[], _raw_point(xi), kk, true)
            # first hit is the query point itself (distance 0 in a live graph)
            nn_id[id] = ids[2]
            nn_dist[id] = dists[2]
            s = spacing(xi)

            repel_force = sum(2:kk) do j
                @inbounds xj = snap[ids[j]]
                @inbounds r = dists[j] * len_unit
                compute_force(force_model, r / s) * _safe_direction(xi, xj, r)
            end

            F_norm = norm(repel_force)
            forces[id] = F_norm * ustrip(s)
            # Adaptive per-point step, capped at one local spacing.
            α_i = clamp(inv(F_norm + 1.0e-30), α_lo, α_max)
            disp = Vec(s * α_i * repel_force)
            d_norm = norm(disp)
            if d_norm > s
                disp = disp * (s / d_norm)
            end
            return constrain(id, xi, xi + disp)
        end
        push!(conv, maximum(forces; init = 0.0))
        if n_move > 0 && (!isnothing(trace) || kick_after > 0)
            pair = _closest_pair(nn_dist, nn_id, spacings, n_fixed)
            isnothing(trace) || push!(trace, (; iteration = i, pair...))
            if kick_after > 0
                kick_state, kicked = _maybe_kick!(
                    p, pair, kick_state, kick_after, spacings, n_fixed,
                    n_protected, len_unit,
                )
                kicked && @debug "Kicked frozen pair at iteration $i"
            end
        end
        if (stall_after > 0 || cv_target > 0) && n_move > 0
            cv = _dnn_cv(nn_dist, spacings, n_fixed)
            if cv_target > 0 && cv <= cv_target
                # The monitor reads the sweep's nn data, which measures the
                # *pre-sweep* snapshot — return that configuration, so a cloud
                # already at target comes back untouched. Must run before
                # deposit!: the sweep and kick mutate only positions (undone
                # here), while deposition also flips is_bnd/tri_indices state
                # that a position revert cannot undo.
                p .= p_old
                @info "Node repel stopped in $i iterations: spacing CV target reached" cv cv_target
                break
            end
            if stall_after > 0
                if cv < best_cv * (1 - 1.0e-3)
                    best_cv = cv
                    last_cv_improvement = i
                elseif i - last_cv_improvement >= stall_after
                    @info "Node repel stopped in $i iterations: spacing CV stalled for $stall_after iterations" cv convergence = conv[end]
                    break
                end
            end
        end
        isnothing(deposit!) || deposit!(p, tree_ref[], i)
        if conv[end] < tol
            @info "Node repel finished in $i iterations" convergence = conv[end]
            break
        end
        i += 1
    end
    if i > max_iters
        @warn "Node repel reached maximum iterations" max_iters convergence = conv[end]
    end
    return conv
end

# ======================================================================
# Per-iteration helpers
# ======================================================================

"""
    _raw_point(pt) -> SVector

Unitless Float64 coordinates of a point (dimension-generic). Float32 CRS
coordinates are promoted so the search tree has a uniform eltype.
"""
@inline _raw_point(pt) = Float64.(ustrip.(Meshes.to(pt)))

"""
    _safe_direction(xi, xj, r) -> Vec

Unit direction from xj to xi; a random unit vector when `r == 0`, avoiding the
0/0 NaN that traps coincident points.
"""
function _safe_direction(xi, xj, r)
    if r > zero(r)
        return (xi - xj) / r
    end
    d = randn(typeof(_raw_point(xi)))
    return d / norm(d)
end

"""
    _dnn_cv(nn_dist, spacings, n_fixed) -> Float64

Coefficient of variation of `d_NN/s` over the movable points, read off the
sweep's per-point nearest-neighbor data. The quality monitor behind
`stall_after`: it tracks the gate's binding spacing-CV metric for free.
"""
function _dnn_cv(nn_dist, spacings, n_fixed)
    n = length(nn_dist)
    s1 = 0.0
    s2 = 0.0
    @inbounds for i in 1:n
        u = nn_dist[i] / spacings[i + n_fixed]
        s1 += u
        s2 += u * u
    end
    μ = s1 / n
    return sqrt(max(s2 / n - μ * μ, 0.0)) / μ
end

"""
    _closest_pair(nn_dist, nn_id, spacings, n_fixed) -> NamedTuple

Closest pair read off the sweep's per-point nearest-neighbor data (no extra
search). Returns `(; r, s, r_over_s, idx_a, idx_b)` in snapshot-global indices.
A frozen `r_over_s` across iterations indicates a balanced standoff; an
oscillating one, an overshoot limit-cycle.
"""
function _closest_pair(nn_dist, nn_id, spacings, n_fixed)
    i = argmin(nn_dist)
    j = nn_id[i]
    ig = i + n_fixed
    r = nn_dist[i]
    s = (spacings[ig] + spacings[j]) / 2
    return (; r, s, r_over_s = r / s, idx_a = min(ig, j), idx_b = max(ig, j))
end

"""
    _maybe_kick!(p, pair, state, kick_after, spacings, n_fixed, n_protected, len_unit)
        -> (state, kicked)

Kick one point of the closest pair by `0.1·s` in a random direction once the
pair has stayed frozen (same indices and `r/s`) for `kick_after` consecutive
iterations. Prefers an index past `n_protected` (a volume point) and always
picks a movable one (past `n_fixed`). `state` is the `(pair, rs, count)` tuple
the caller threads through.
"""
function _maybe_kick!(
        p::AbstractVector, pair::NamedTuple, state::NamedTuple, kick_after::Int,
        spacings::AbstractVector{<:AbstractFloat}, n_fixed::Int, n_protected::Int,
        len_unit,
    )
    frozen = (pair.idx_a, pair.idx_b) == state.pair &&
        abs(pair.r_over_s - state.rs) < 1.0e-8
    count = frozen ? state.count + 1 : 1
    if count < kick_after
        return (; pair = (pair.idx_a, pair.idx_b), rs = pair.r_over_s, count), false
    end

    a, b = pair.idx_a, pair.idx_b
    target = a > n_protected ? a : (b > n_protected ? b : (a > n_fixed ? a : b))
    s = spacings[target]
    d = randn(typeof(_raw_point(p[target - n_fixed])))
    p[target - n_fixed] += Vec(0.1 * s * (d / norm(d)) * len_unit)
    return (; pair = (a, b), rs = pair.r_over_s, count = 0), true
end

# ======================================================================
# Octree wall handling
# ======================================================================

"""
    _constrain_octree(id, xi, x_proposed, is_bnd, escaped, tri_indices,
                      octree, offset_dist, len_unit) -> Point

Wall rule for one point: boundary points are re-projected onto the mesh
(falling back to projecting their previous position), volume points keep the
proposed position while it stays inside, and escapees revert — flagged in
`escaped` as deposition candidates.
"""
function _constrain_octree(
        id, xi, x_proposed, is_bnd, escaped, tri_indices,
        octree, offset_dist, len_unit,
    )
    sv_proposed = _extract_vertex(Float64, x_proposed)
    if is_bnd[id]
        sv, tri_idx = _project_to_boundary(sv_proposed, octree, offset_dist)
        if tri_idx == 0
            sv, tri_idx = _project_to_boundary(
                _extract_vertex(Float64, xi), octree, offset_dist
            )
        end
        tri_indices[id] = tri_idx
        return Point(sv[1] * len_unit, sv[2] * len_unit, sv[3] * len_unit)
    end
    isinside(sv_proposed, octree) && return x_proposed
    escaped[id] = true
    return xi
end

"""
    _deposit_escaped!(p, tree, kq, escaped, is_bnd, tri_indices, octree, spacing,
                      deposit_ratio, offset_dist, len_unit) -> n_deposited

One deposition pass: each escaped volume point is projected onto its nearest
triangle and converted to a boundary point, unless another boundary point
already sits within `deposit_ratio·spacing` of the landing site. `tree` is the
sweep's snapshot kd-tree and `kq` the neighbor count to inspect. Serial on
purpose — earlier deposits must be visible to later candidates, because
conversion is one-way and a parallel pass would over-deposit when a whole
layer escapes in one iteration.
"""
function _deposit_escaped!(
        p, tree, kq, escaped, is_bnd, tri_indices, octree,
        spacing, deposit_ratio, offset_dist, len_unit,
    )
    n_dep = 0
    for id in eachindex(p)
        escaped[id] || continue
        escaped[id] = false
        is_bnd[id] && continue
        site, tri_idx = _project_to_boundary(
            _extract_vertex(Float64, p[id]), octree, offset_dist
        )
        tri_idx == 0 && continue
        site_pt = Point(site[1] * len_unit, site[2] * len_unit, site[3] * len_unit)
        thr = deposit_ratio * ustrip(spacing(site_pt))
        ids_near, _ = knn(tree, _raw_point(site_pt), kq, true)
        occupied = any(
            j -> j != id && is_bnd[j] && ustrip(norm(p[j] - site_pt)) < thr,
            ids_near,
        )
        occupied && continue
        p[id] = site_pt
        is_bnd[id] = true
        tri_indices[id] = tri_idx
        n_dep += 1
    end
    return n_dep
end

"""
    _project_to_boundary(sv, octree, offset_dist) -> (SVector, tri_idx)

Nearest point on the mesh surface, nudged `offset_dist` inward along the
triangle normal. `tri_idx == 0` means no triangle was found.
"""
function _project_to_boundary(
        sv::SVector{3, T}, octree::TriangleOctree, offset_dist::T,
    ) where {T <: Real}
    state = NearestTriangleState{T}(sv)
    _nearest_triangle_octree!(sv, octree.tree, octree.mesh, 1, state)
    state.closest_idx == 0 && return (sv, 0)

    v1, v2, v3 = _get_triangle_vertices(T, octree.mesh, state.closest_idx)
    projected = closest_point_on_triangle(sv, v1, v2, v3)

    n = _get_triangle_normal(T, octree.mesh, state.closest_idx)
    return (projected - offset_dist * n, state.closest_idx)
end

# ======================================================================
# Post-processing
# ======================================================================

"""
    _cull(pts, spacing, ratio) -> BitVector

Near-duplicate keep-mask plus the defect warning: the cull is a safety net, so
any non-zero removal is surfaced.
"""
function _cull(pts, spacing, ratio)
    keep = _near_duplicate_keep_mask(pts, ustrip.(spacing.(pts)), ratio)
    n_culled = count(!, keep)
    n_culled > 0 &&
        @warn "Cull removed $n_culled near-duplicate point(s) — repel left defects behind" cull_ratio = ratio
    return keep
end

"""
    _near_duplicate_keep_mask(pts, spacings, ratio) -> BitVector

Greedy, order-preserving keep-mask: drop any point closer than `ratio·spacing`
to a kept, lower-indexed point (so boundary points, indexed first, survive over
volume points). The ball search at the largest cull radius sees *every* point
inside the threshold, so the guarantee holds for clusters of any size.
"""
function _near_duplicate_keep_mask(pts, spacings, ratio)
    n = length(pts)
    keep = trues(n)
    (ratio <= 0 || n < 2) && return keep
    len_unit = Unitful.unit(Meshes.to(first(pts))[1])
    method = BallSearch(pts, MetricBall(ratio * maximum(spacings) * len_unit))
    @inbounds for i in 1:n
        keep[i] || continue
        thr = ratio * spacings[i]
        for j in search(pts[i], method)
            (j == i || !keep[j]) && continue
            ustrip(norm(pts[j] - pts[i])) < thr && (keep[j] = false)
        end
    end
    return keep
end

"""
    _reconstruct_cloud(cloud, p, tri_indices, is_bnd, n_boundary, octree, spacing, keep)

Rebuild a `PointCloud` after octree repel: kept points are partitioned by
`is_bnd` into a single `:boundary` surface and the volume. Projected boundary
points take the landing triangle's normal; imported ones (`id ≤ n_boundary`)
keep their original area, deposited ones get `spacing²`.
"""
function _reconstruct_cloud(
        cloud::PointCloud{𝔼{3}, C},
        p::AbstractVector{<:Point{𝔼{3}, C}},
        tri_indices::Vector{Int},
        is_bnd::AbstractVector{Bool},
        n_boundary::Int,
        octree::TriangleOctree,
        spacing,
        keep::AbstractVector{Bool} = trues(length(p)),
    ) where {C <: CRS}
    orig_normals = normal(boundary(cloud))
    orig_areas = area(boundary(cloud))

    new_bnd_pts = Point{𝔼{3}, C}[]
    new_bnd_normals = SVector{3, Float64}[]
    new_bnd_areas = eltype(orig_areas)[]
    new_vol_pts = Point{𝔼{3}, C}[]

    for id in eachindex(p)
        keep[id] || continue
        if is_bnd[id]
            push!(new_bnd_pts, p[id])
            if tri_indices[id] > 0
                push!(new_bnd_normals, _get_triangle_normal(Float64, octree.mesh, tri_indices[id]))
            else
                push!(new_bnd_normals, orig_normals[id])
            end
            push!(new_bnd_areas, id <= n_boundary ? orig_areas[id] : spacing(p[id])^2)
        else
            push!(new_vol_pts, p[id])
        end
    end

    new_surf = PointSurface(new_bnd_pts, new_bnd_normals, new_bnd_areas)
    new_bnd = PointBoundary(LittleDict{Symbol, typeof(new_surf)}(:boundary => new_surf))
    new_vol = isempty(new_vol_pts) ? PointVolume{𝔼{3}, C}() : PointVolume(new_vol_pts)

    return PointCloud(new_bnd, new_vol, NoTopology())
end
