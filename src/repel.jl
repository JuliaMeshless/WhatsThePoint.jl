"""
    _safe_direction(xi, xj, r) -> Vec

Unit direction from xj to xi. When r == 0 (coincident points), returns a random
unit vector instead of NaN — the 0/0 singularity that traps coincident points.
"""
function _safe_direction(xi, xj, r)
    if r > zero(r)
        return (xi - xj) / r
    end
    # Random unit vector — breaks the NaN trap for coincident points
    d = randn(SVector{3, Float64})
    return d / norm(d)
end

"""
    _near_duplicate_keep_mask(pts, spacings, ratio) -> BitVector

Flag points to keep, dropping near-duplicates closer than `ratio * spacing` to a
kept point. Greedy and order-preserving: the lower-indexed point of each too-close
pair is kept (so fixed boundary points, which come first, survive over volume
points). Targets the stuck boundary pair that the deterministic
`closest_point_on_triangle` projection parks at a shared edge/vertex — a single
mesh-ratio-dominating defect that no number of repel iterations separates.
"""
function _near_duplicate_keep_mask(pts, spacings, ratio)
    n = length(pts)
    keep = trues(n)
    (ratio <= 0 || n < 2) && return keep
    k = min(n, 8)
    method = KNearestSearch(pts, k)
    @inbounds for i in 1:n
        keep[i] || continue
        thr = ratio * spacings[i]
        ids, dists = searchdists(pts[i], method)
        for (j, d) in zip(ids, dists)
            (j == i || !keep[j]) && continue
            ustrip(d) < thr && (keep[j] = false)
        end
    end
    return keep
end

"""
    _trace_closest_pair!(trace, points, spacings, iteration)

Push a diagnostic snapshot of the closest pair in `points` to `trace`.

Each entry records the iteration number, the pair's separation `r`, the
per-point spacing `s`, the normalized separation `r/s`, and the indices of
the pair. Comparing entries across iterations distinguishes:

- **Standoff**: `r/s` is constant (non-zero), forces balance exactly.
- **Overshoot limit-cycle**: `r/s` oscillates (displacement alternates sign).
"""
function _trace_closest_pair!(
        trace::AbstractVector{<:NamedTuple},
        points::AbstractVector,
        spacings::AbstractVector{<:AbstractFloat},
        iteration::Int,
    )
    n = length(points)
    n < 2 && return
    best_r = Inf
    best_i, best_j = 1, 2
    for a in 1:(n - 1)
        for b in (a + 1):n
            d = ustrip(norm(points[a] - points[b]))
            if d < best_r
                best_r = d
                best_i, best_j = a, b
            end
        end
    end
    s_avg = (spacings[best_i] + spacings[best_j]) / 2
    push!(trace, (;
        iteration,
        r = best_r,
        s = s_avg,
        r_over_s = best_r / s_avg,
        idx_a = best_i,
        idx_b = best_j,
    ))
end

"""
    _kick_frozen_pair!(p, spacings, n_boundary, state, kick_after) -> (kicked, state)

Detect if the closest pair is frozen (same indices, same r/s for several
iterations). If frozen for `kick_after` consecutive iterations, apply a random
displacement to the volume point in the pair and return `(true, new_state)`.
Otherwise return `(false, updated_state)`.

`state` is a `NamedTuple` tracking `(pair, rs, count)` across calls — the caller
owns it and passes it in each iteration.
"""
function _kick_frozen_pair!(
        p::AbstractVector, spacings::AbstractVector{<:AbstractFloat},
        n_boundary::Int, state::NamedTuple, kick_after::Int,
    )
    n = length(p)
    n < 2 && return false, state

    # Find closest pair
    best_r = Inf; best_i, best_j = 1, 2
    for a in 1:(n - 1)
        for b in (a + 1):n
            d = ustrip(norm(p[a] - p[b]))
            if d < best_r
                best_r = d; best_i, best_j = a, b
            end
        end
    end
    s_avg = (spacings[best_i] + spacings[best_j]) / 2
    rs = best_r / s_avg

    # Check if frozen (same pair, same r/s within tolerance)
    frozen = (best_i == state.pair[1] && best_j == state.pair[2] &&
              abs(rs - state.rs) < 1e-8)
    count = frozen ? state.count + 1 : 1

    if count >= kick_after
        # Pick the volume point to kick (prefer the one past n_boundary)
        target = best_i > n_boundary ? best_i : (best_j > n_boundary ? best_j : best_i)
        s = spacings[target]
        d = randn(SVector{3, Float64})
        kick_dir = d / (norm(d) + 1e-30)
        p[target] = p[target] + Vec(0.1 * s * kick_dir * Unitful.m)
        return true, (; pair = (best_i, best_j), rs, count = 0)
    end

    return false, (; pair = (best_i, best_j), rs, count)
end

"""
    repel(cloud::PointCloud, spacing;
          force_model=SpacingEquilibriumForce(β), β=0.2,
          α=auto, α_min=auto, k=21, max_iters=1000, tol=1e-6, rebuild_every=1,
          cull_ratio=0.0, kick_after=0, convergence=nothing, trace=nothing)

Optimize point distribution via node repulsion (Miotti 2023).
Only volume points move; escaped points are discarded via `isinside` filtering.

Uses an **adaptive per-point step size** to eliminate oscillation and accelerate
convergence. Each point gets `α_i = clamp(s_i / (|F_i| + ε), α_min, α_max)` where
`s_i` is the local spacing and `|F_i|` is the repel-force norm. Points with strong
forces (near-duplicates, voids) take the maximum safe step; points near equilibrium
take tiny steps. `α` sets `α_max`; `α_min` defaults to `α_max / 100`.

Displacement is capped at one spacing unit per iteration (`|Δp| ≤ s_i`). This
prevents runaway when strong short-range forces (near-duplicates, standoff pairs)
produce large force magnitudes — the point moves at most one local spacing per
step, keeping the relaxation stable.

Convergence is measured by the **force-norm** criterion `max_i(|F_i| * s_i)` rather
than displacement. At equilibrium forces vanish, so this detects true convergence
earlier and avoids false stops from oscillating points whose displacement is small
but restoring forces are still large.

When `cull_ratio > 0`, near-duplicate points closer than `cull_ratio * spacing` to a
kept point are removed after relaxation (the lower-indexed point of each pair is kept).
This is `0.0` (off) by default; a value like `0.5` removes pathological close pairs
that the force law alone settles into a standoff with. Note this changes the point
count.

When `kick_after > 0`, the closest pair is tracked across iterations. If the same
pair stays at the same `r/s` for `kick_after` consecutive iterations (a balanced
standoff), a small random displacement (0.1·s) is applied to the volume point in
the pair to break the symmetry. This is `0` (off) by default; `10`–`20` is a
reasonable value. The kick is safe — the displacement cap still applies.

This is a true N-body relaxation: the k-nearest-neighbor graph is rebuilt from
the *current* positions every `rebuild_every` iterations (default `1`), so each
point responds to where its neighbors have actually moved rather than to the
initial configuration. Boundary points stay fixed and act as a static wall the
volume points pack against. Larger `rebuild_every` amortizes the kd-tree rebuild
at the cost of a slightly staler neighbor graph between rebuilds.

The returned cloud has `NoTopology` since points have moved.

Pass a `Vector{Float64}` via the `convergence` keyword to collect the convergence history.

The force law is controlled by `force_model`, any subtype of [`RepelForceModel`](@ref).
The default [`SpacingEquilibriumForce`](@ref) vanishes at `r = s` (the target
spacing), so the cloud settles at the prescribed density instead of drifting
outward under purely repulsive forcing. Use [`InverseDistanceForce`](@ref) for the
original purely-repulsive Miotti (2023) law (equilibrium then relies on `α`
damping alone). `β` is kept as a convenience kwarg that feeds the default force
model; it is ignored when `force_model` is passed explicitly.

Pass a `Vector{NamedTuple}` via `trace` to record per-iteration diagnostics of
the closest pair (iteration, r, s, r/s, indices). Comparing entries distinguishes
a balanced standoff (`r/s` frozen) from an overshoot limit-cycle (`r/s` oscillating).
"""
function repel(
        cloud::PointCloud{𝔼{N}, C},
        spacing;
        β = 0.2,
        force_model::RepelForceModel = SpacingEquilibriumForce(β),
        α = minimum(spacing.(to(cloud))) * 0.05,
        α_min = α / 100,
        k = 21,
        max_iters = 1000,
        tol = 1.0e-6,
        rebuild_every::Int = 1,
        cull_ratio::Real = 0.0,
        kick_after::Int = 0,
        convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
        trace::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    ) where {N, C <: CRS}
    rebuild_every >= 1 || throw(ArgumentError("rebuild_every must be ≥ 1"))
    α_max = ustrip(α)
    α_lo = ustrip(α_min)
    bnd_p = points(boundary(cloud))
    n_bnd = length(bnd_p)
    p = copy(volume(cloud).points)
    p_old = deepcopy(p)
    npoints = length(p)

    # Combined boundary+volume buffer the kd-tree is built from. The volume tail
    # is refreshed in place on each rebuild so repulsion sees current neighbor
    # positions; the boundary head is fixed.
    all_cur = vcat(bnd_p, p)
    # Held in a Ref so the per-iteration rebuild does not rebind a captured
    # variable (OhMyThreads' tmap! rejects boxed closure captures).
    method_ref = Ref(KNearestSearch(all_cur, k))

    vol_spacings = ustrip.(spacing.(p))
    # Force-norm convergence: max_i(|F_i| * s_i). At equilibrium forces vanish,
    # so this detects true convergence earlier than displacement-based metrics
    # which can be small while forces are still large (oscillation).
    forces = zeros(npoints)

    # Frozen-pair kick state
    kick_state = (; pair = (0, 0), rs = Inf, count = 0)

    conv = Float64[]
    i = 1
    while i <= max_iters
        p_old .= p
        if (i - 1) % rebuild_every == 0
            @views all_cur[(n_bnd + 1):end] .= p
            method_ref[] = KNearestSearch(all_cur, k)
        end
        tmap!(p, 1:npoints) do id
            xi = p_old[id]
            ids, dists = searchdists(xi, method_ref[])
            ids = @view ids[2:end]
            neighborhood = @view all_cur[ids]
            rij = norm.(@view dists[2:end])
            s = spacing(xi)

            repel_force = sum(zip(neighborhood, rij)) do z
                xj, r = z
                @inbounds compute_force(force_model, r / s) * _safe_direction(xi, xj, r)
            end

            F_norm = norm(repel_force)
            forces[id] = F_norm * ustrip(s)
            α_i = clamp(inv(F_norm + 1.0e-30), α_lo, α_max)
            disp = Vec(s * α_i * repel_force)
            d_norm = norm(disp)
            if d_norm > s
                disp = disp * (s / d_norm)
            end
            return xi + disp
        end
        push!(conv, maximum(forces))
        if !isnothing(trace)
            _trace_closest_pair!(trace, p, vol_spacings, i)
        end
        # Stochastic kick: break frozen-pair standoffs
        if kick_after > 0
            kicked, kick_state = _kick_frozen_pair!(
                p, vol_spacings, n_bnd, kick_state, kick_after
            )
            if kicked
                @debug "Kicked frozen pair at iteration $i"
            end
        end
        if conv[end] < tol
            @info "Node repel finished in $i iterations" convergence = conv[end]
            break
        end
        i = i + 1
    end
    if i > max_iters
        @warn "Node repel reached maximum iterations" max_iters convergence = conv[end]
    end
    if !isnothing(convergence)
        append!(convergence, conv)
    end
    survivors = filter(x -> isinside(x, cloud), p)
    if cull_ratio > 0 && !isempty(survivors)
        keep = _near_duplicate_keep_mask(survivors, ustrip.(spacing.(survivors)), cull_ratio)
        survivors = survivors[keep]
    end
    new_volume = PointVolume(survivors)
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

"""
    repel(cloud::PointCloud, spacing, octree::TriangleOctree;
          force_model=SpacingEquilibriumForce(β), β=0.2,
          α=auto, α_min=auto, k=21, max_iters=1000, tol=1e-6, rebuild_every=1,
          cull_ratio=0.0, kick_after=0, convergence=nothing, trace=nothing)

Optimize point distribution via node repulsion (Miotti 2023) with boundary projection.

All points (boundary and volume) participate in repulsion. Points that escape the domain
are projected back to the nearest mesh triangle. Boundary points are always projected
back to the surface after each iteration.

Uses an **adaptive per-point step size** to eliminate oscillation and accelerate
convergence. Each point gets `α_i = clamp(s_i / (|F_i| + ε), α_min, α_max)` where
`s_i` is the local spacing and `|F_i|` is the repel-force norm. `α` sets `α_max`;
`α_min` defaults to `α_max / 100`. Displacement is capped at one spacing unit per
iteration (`|Δp| ≤ s_i`) to prevent runaway from strong short-range forces.

Convergence is measured by the **force-norm** criterion `max_i(|F_i| * s_i)` — at
equilibrium forces vanish, so this detects true convergence earlier than
displacement-based metrics.

When `cull_ratio > 0`, near-duplicate points closer than `cull_ratio * spacing` to a
kept point are removed after relaxation (boundary points, indexed first, are kept over
volume points). This targets the stuck boundary pair the deterministic
`closest_point_on_triangle` projection parks at a shared edge/vertex — a single defect
that dominates `mesh_ratio` and that no number of iterations separates. Off (`0.0`) by
default; `0.5` is a reasonable value. Note this changes the point count.

When `kick_after > 0`, the closest pair is tracked across iterations. If the same
pair stays at the same `r/s` for `kick_after` consecutive iterations (a balanced
standoff), a small random displacement (0.1·s) is applied to the volume point in
the pair to break the symmetry. Off (`0`) by default; `10`–`20` is reasonable.

Like the volume-only method, this is a true N-body relaxation: the
k-nearest-neighbor graph is rebuilt from the *current* positions every
`rebuild_every` iterations (default `1`) so points respond to their neighbors'
actual motion rather than the initial layout.

The returned cloud has `NoTopology` since points have moved. The boundary is returned as a
single unified surface named `:boundary`. Use `split_surface!(cloud.boundary, angle)` to
re-establish surface distinctions if needed.

Pass a `Vector{Float64}` via the `convergence` keyword to collect the convergence history.

`force_model` accepts any subtype of [`RepelForceModel`](@ref); see the method without
`octree` for details on the available force laws.
"""
function repel(
        cloud::PointCloud{𝔼{3}, C},
        spacing,
        octree::TriangleOctree;
        β = 0.2,
        force_model::RepelForceModel = SpacingEquilibriumForce(β),
        α = minimum(spacing.(to(cloud))) * 0.05,
        α_min = α / 100,
        k = 21,
        max_iters = 1000,
        tol = 1.0e-6,
        rebuild_every::Int = 1,
        cull_ratio::Real = 0.0,
        kick_after::Int = 0,
        convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
        trace::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    ) where {C <: CRS}
    rebuild_every >= 1 || throw(ArgumentError("rebuild_every must be ≥ 1"))
    α_max = ustrip(α)
    α_lo = ustrip(α_min)

    all_p = points(cloud)
    npoints_total = length(all_p)
    n_boundary = length(boundary(cloud))

    p = copy(all_p)
    p_old = copy(all_p)
    # Snapshot the kd-tree is built from; refreshed in place on each rebuild so
    # repulsion uses current neighbor positions (a Jacobi-style update — every
    # force in a sweep is evaluated against the same frozen snapshot).
    snap = copy(all_p)
    # Ref so the per-iteration rebuild does not rebind a captured variable
    # (OhMyThreads' tmap! rejects boxed closure captures).
    method_ref = Ref(KNearestSearch(snap, k))

    all_spacings = ustrip.(spacing.(p))
    # Force-norm convergence: max_i(|F_i| * s_i)
    forces = zeros(npoints_total)

    # Frozen-pair kick state
    kick_state = (; pair = (0, 0), rs = Inf, count = 0)

    sample_coords = Meshes.to(first(all_p))
    len_unit = Unitful.unit(sample_coords[1])

    offset_dist = 1.0e-6 * norm(octree.mesh_bbox_max - octree.mesh_bbox_min)

    tri_indices = zeros(Int, npoints_total)

    conv = Float64[]
    i = 1
    while i <= max_iters
        p_old .= p
        if (i - 1) % rebuild_every == 0
            snap .= p
            method_ref[] = KNearestSearch(snap, k)
        end
        tmap!(p, 1:npoints_total) do id
            xi = p_old[id]
            ids, dists = searchdists(xi, method_ref[])
            ids = @view ids[2:end]
            neighborhood = @view snap[ids]
            rij = norm.(@view dists[2:end])
            s = spacing(xi)

            repel_force = sum(zip(neighborhood, rij)) do z
                xj, r = z
                @inbounds compute_force(force_model, r / s) * _safe_direction(xi, xj, r)
            end

            F_norm = norm(repel_force)
            forces[id] = F_norm * ustrip(s)
            α_i = clamp(inv(F_norm + 1.0e-30), α_lo, α_max)
            disp = Vec(s * α_i * repel_force)
            d_norm = norm(disp)
            if d_norm > s
                disp = disp * (s / d_norm)
            end
            x_proposed = xi + disp

            sv_proposed = _extract_vertex(Float64, x_proposed)
            sv_original = _extract_vertex(Float64, xi)
            is_bnd = id <= n_boundary

            sv_result, tri_idx = _constrain_to_domain(
                sv_proposed, sv_original, octree, is_bnd, offset_dist
            )
            tri_indices[id] = tri_idx

            return Point(
                sv_result[1] * len_unit, sv_result[2] * len_unit, sv_result[3] * len_unit
            )
        end

        push!(conv, maximum(forces))
        if !isnothing(trace)
            _trace_closest_pair!(trace, p, all_spacings, i)
        end
        # Stochastic kick: break frozen-pair standoffs
        if kick_after > 0
            kicked, kick_state = _kick_frozen_pair!(
                p, all_spacings, n_boundary, kick_state, kick_after
            )
            if kicked
                @debug "Kicked frozen pair at iteration $i"
            end
        end
        if conv[end] < tol
            @info "Node repel finished in $i iterations" convergence = conv[end]
            break
        end
        i += 1
    end
    if i > max_iters
        @warn "Node repel reached maximum iterations" max_iters convergence = conv[end]
    end
    if !isnothing(convergence)
        append!(convergence, conv)
    end

    keep = cull_ratio > 0 ?
        _near_duplicate_keep_mask(p, ustrip.(spacing.(p)), cull_ratio) :
        trues(npoints_total)
    return _reconstruct_cloud(cloud, p, tri_indices, n_boundary, octree, keep)
end

function _reconstruct_cloud(
        cloud::PointCloud{𝔼{3}, C},
        p::AbstractVector{<:Point{𝔼{3}, C}},
        tri_indices::Vector{Int},
        n_boundary::Int,
        octree::TriangleOctree,
        keep::AbstractVector{Bool} = trues(length(p)),
    ) where {C <: CRS}
    npoints_total = length(p)

    orig_normals = normal(boundary(cloud))
    orig_areas = area(boundary(cloud))

    new_bnd_pts = Point{𝔼{3}, C}[]
    new_bnd_normals = SVector{3, Float64}[]
    new_bnd_areas = eltype(orig_areas)[]
    new_vol_pts = Point{𝔼{3}, C}[]

    for id in 1:npoints_total
        keep[id] || continue
        if id <= n_boundary
            push!(new_bnd_pts, p[id])
            if tri_indices[id] > 0
                push!(new_bnd_normals, _get_triangle_normal(Float64, octree.mesh, tri_indices[id]))
            else
                push!(new_bnd_normals, orig_normals[id])
            end
            push!(new_bnd_areas, orig_areas[id])
        else
            push!(new_vol_pts, p[id])
        end
    end

    new_surf = PointSurface(new_bnd_pts, new_bnd_normals, new_bnd_areas)
    new_bnd = PointBoundary(LittleDict{Symbol, typeof(new_surf)}(:boundary => new_surf))
    new_vol = isempty(new_vol_pts) ? PointVolume{𝔼{3}, C}() : PointVolume(new_vol_pts)

    return PointCloud(new_bnd, new_vol, NoTopology())
end

function _project_to_boundary(
        sv::SVector{3, T}, octree::TriangleOctree, offset_dist::T,
    ) where {T <: Real}
    state = NearestTriangleState{T}(sv)
    _nearest_triangle_octree!(sv, octree.tree, octree.mesh, 1, state)
    state.closest_idx == 0 && return (sv, 0)

    v1, v2, v3 = _get_triangle_vertices(T, octree.mesh, state.closest_idx)
    projected = closest_point_on_triangle(sv, v1, v2, v3)

    n = _get_triangle_normal(T, octree.mesh, state.closest_idx)
    projected = projected - offset_dist * n

    return (projected, state.closest_idx)
end

function _constrain_to_domain(
        sv_proposed::SVector{3, T}, sv_original::SVector{3, T},
        octree::TriangleOctree, is_boundary::Bool, offset_dist::T,
    ) where {T <: Real}
    if is_boundary
        sv_result, tri_idx = _project_to_boundary(sv_proposed, octree, offset_dist)
        if tri_idx > 0
            return (sv_result, tri_idx)
        end
        return _project_to_boundary(sv_original, octree, offset_dist)
    end
    if isinside(sv_proposed, octree)
        return (sv_proposed, 0)
    end
    return (sv_original, 0)
end
