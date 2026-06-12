# shape_opt_tradeoff.jl — incremental repel vs re-seed per shape-opt design step.
#
# Pre-step for NODEGEN_FINDINGS.md plan item 2 (gap-tracking sampler trigger a):
# starting from the direct-pipeline cloud on the base cavity (r_inner=0.547),
# deform the geometry programmatically and compare, per deformation rung:
#
#   A.  re-seed:    new Poisson-disk boundary + :bridson volume from scratch.
#   B.  warm-start: same new boundary + OLD volume points, production repel
#                   (kick_after=10, cv_target=0.07, stall_after=50, cull_ratio=0.5).
#       Old volume points now outside the new domain are the stress case:
#       _constrain_octree reverts an escaped point to its previous position, so
#       a point that STARTS outside can only re-enter if a force proposal lands
#       inside — watch the stranded count.
#   B2. warm-start with old volume points outside the new domain dropped first
#       (one isinside pass — what a practical loop would do). Run only when the
#       rung actually strands points.
#
# Deformation ladder: r_inner 0.547 → {0.55, 0.56, 0.60} (≈0.5/2/10 % steps),
# plus a 5 % ellipsoidal x-stretch of the outer shell (domain GROWS: no escapes,
# but a fresh underdense layer at the outer wall that repulsion-only repel must
# fill by expansion).
#
# Every deformed geometry is validated BEFORE measuring (total area vs analytic,
# isinside probes) — d_NN metrics cannot see a broken domain (lesson of
# 2026-06-11). For the same reason every FINAL cloud gets a fill-distance
# estimate (max/p99 over ~20k interior probes of distance to the nearest cloud
# point): gate metrics are d_NN-based and structurally blind to coverage voids.
#
# Run single-threaded:  julia --project=. -t 1 shape_opt_tradeoff.jl

using Pkg
Pkg.activate(@__DIR__)

using WhatsThePoint
using Meshes
using NearestNeighbors: KDTree, knn, inrange
using Unitful: m, ustrip
using StaticArrays: SVector
using LinearAlgebra: norm, svd
using Statistics: mean, std, median, quantile
using Printf, Random

const Δ = 0.08
const R_OUTER = 1.0
const R_INNER_BASE = 0.547
const N_ANGULAR = 24
const SEED = 20260611
const SPACING = ConstantSpacing(Δ * m)

# ============================================================================
# Geometry: annular cavity with parameterized inner radius and outer stretch
# (generator re-parameterized from validate_cavity.jl; cavity.stl untouched)
# ============================================================================

function _make_sphere_mesh(R, nθ, nφ; stretch=(1.0, 1.0, 1.0))
    sx, sy, sz = stretch
    # NOTE index layout: connectivity addresses point (i, j) at linear index
    # i*nφ + j + 1 (row-major over i); comprehensions flatten column-major, so
    # j must be the FIRST iterator (the 2026-06-11 corruption lesson). Guarded
    # by the analytic-area check in validate_geometry.
    pts = [Meshes.Point(sx * R * sin(π * i / nθ) * cos(2π * j / nφ),
                        sy * R * sin(π * i / nθ) * sin(2π * j / nφ),
                        sz * R * cos(π * i / nθ))
           for j in 0:(nφ - 1), i in 0:nθ][:]

    conn = Connectivity{Triangle}[]
    for i in 0:(nθ - 1), j in 0:(nφ - 1)
        a = i * nφ + j + 1
        b = i * nφ + (j + 1) % nφ + 1
        c = (i + 1) * nφ + j + 1
        d = (i + 1) * nφ + (j + 1) % nφ + 1
        # Winding chosen so Meshes.normal points OUTWARD (away from the
        # center). The pre-2026-06-11-PM order (a,b,c)/(b,d,c) wound inward:
        # the whole cavity classified inside-out (solid annulus EXTERIOR,
        # hole + bbox corners INTERIOR) and every volume fill landed in the
        # complement — invisible to all d_NN gate metrics. Guarded by the
        # isinside probes in validate_geometry.
        i > 0 && push!(conn, connect((a, c, b)))
        i < nθ - 1 && push!(conn, connect((b, c, d)))
    end

    return SimpleMesh(pts, conn)
end

function _make_cavity_mesh(r_inner, outer_stretch)
    nθ, nφ = N_ANGULAR, 2 * N_ANGULAR
    outer = _make_sphere_mesh(R_OUTER, nθ, nφ; stretch=outer_stretch)
    inner = _make_sphere_mesh(r_inner, nθ, nφ)

    outer_v = collect(Meshes.vertices(outer))
    inner_v = collect(Meshes.vertices(inner))
    n_outer = length(outer_v)
    all_v = vcat(outer_v, inner_v)

    all_conn = Connectivity{Triangle}[]
    for c in Meshes.topology(outer)
        push!(all_conn, c)
    end
    for c in Meshes.topology(inner)
        pts = Meshes.indices(c)
        # Reverse winding to flip normals (inward toward origin)
        push!(all_conn, connect((pts[1] + n_outer, pts[3] + n_outer, pts[2] + n_outer)))
    end

    return SimpleMesh(all_v, all_conn)
end

# Knud Thomsen approximation (exact for spheres, rel err < 1.1 % in general)
function _ellipsoid_area(a, b, c)
    p = 1.6075
    return 4π * (((a * b)^p + (a * c)^p + (b * c)^p) / 3)^(1 / p)
end

_analytic_max_points(r_inner, stretch) =
    round(Int, (4π / 3) * (prod(stretch) * R_OUTER^3 - r_inner^3) / Δ^3)

const PROBE_DIRS = let
    dirs = [SVector{3,Float64}(1, 0, 0), SVector{3,Float64}(0, 1, 0),
            SVector{3,Float64}(0, 0, 1), SVector{3,Float64}(-1, 0, 0),
            SVector{3,Float64}(0, -1, 0), SVector{3,Float64}(0, 0, -1)]
    for x in (-1, 1), y in (-1, 1), z in (-1, 1)
        push!(dirs, SVector{3,Float64}(x, y, z) / sqrt(3.0))
    end
    dirs
end

"""Area vs analytic + isinside probes. Errors out on any failure — a broken
domain must not reach the measurement (d_NN metrics cannot see it)."""
function validate_geometry(label, mesh, octree, r_inner, stretch)
    area_mesh = sum(ustrip(Meshes.area(e)) for e in Meshes.elements(mesh))
    a, b, c = R_OUTER .* stretch
    area_true = _ellipsoid_area(a, b, c) + 4π * r_inner^2
    rel = abs(area_mesh - area_true) / area_true
    @printf("  geometry[%s]: %d facets, area %.4f vs analytic %.4f (rel err %.2e)\n",
        label, Meshes.nelements(mesh), area_mesh, area_true, rel)
    rel < 0.02 || error("[$label] mesh area deviates $rel from analytic — geometry bug")

    n_in, n_out = 0, 0
    for d in PROBE_DIRS
        R_out = 1 / sqrt((d[1] / a)^2 + (d[2] / b)^2 + (d[3] / c)^2)
        isinside(((r_inner + R_out) / 2) * d, octree) ||
            error("[$label] mid-annulus probe along $d misclassified outside")
        n_in += 1
        for q in ((0.5 * r_inner) * d, (1.1 * R_out) * d)
            isinside(q, octree) &&
                error("[$label] exterior probe $q misclassified inside")
            n_out += 1
        end
    end
    @printf("  geometry[%s]: %d inside + %d outside isinside probes OK\n", label, n_in, n_out)
    return nothing
end

# ============================================================================
# Metrics: gate (d_NN), near-wall slice, stencil conditioning, fill distance
# ============================================================================

_raw(p) = SVector{3,Float64}(Float64.(ustrip.(Meshes.to(p)))...)

function _dnn(coords)
    tree = KDTree(coords)
    _, dists = knn(tree, coords, 2, true)
    return [d[2] for d in dists]
end

function gate_metrics(cloud)
    all_p = points(cloud)
    nb = length(points(WhatsThePoint.boundary(cloud)))
    coords = _raw.(all_p)
    dnn = _dnn(coords) ./ Δ
    tree = KDTree(coords)
    coordn = mean(length(inrange(tree, c, 1.4 * Δ)) - 1 for c in coords)
    return (; sep=minimum(dnn), cv=std(dnn) / mean(dnn), mean=mean(dnn),
        coord=coordn, n=length(all_p), nb, coords, dnn)
end

"""d_NN/h stats restricted to points near the moved wall (pred on coords)."""
function near_wall_metrics(g, pred)
    sel = findall(pred, g.coords)
    isempty(sel) && return (; n=0, mean=NaN, cv=NaN, min=NaN)
    d = g.dnn[sel]
    return (; n=length(sel), mean=mean(d), cv=std(d) / mean(d), min=minimum(d))
end

function vander3(P)
    n = length(P)
    V = Matrix{Float64}(undef, n, 20)
    for (r, p) in enumerate(P)
        x, y, z = p[1], p[2], p[3]
        V[r, :] .= (1, x, y, z, x^2, y^2, z^2, x * y, x * z, y * z,
            x^3, y^3, z^3, x^2 * y, x^2 * z, y^2 * x, y^2 * z, z^2 * x, z^2 * y, x * y * z)
    end
    return V
end

function singular_stencils(cloud; k=50)
    pts = _raw.(points(cloud))
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)
    nsing, nval, minσ = 0, 0, Inf
    for i in eachindex(pts)
        ids = results[i][1]
        nbrs = [pts[j] for j in ids if j != i]
        length(nbrs) < 20 && continue
        c = sum(nbrs) / length(nbrs)
        h = maximum(norm(p - c) for p in nbrs)
        s = svd(vander3([(p - c) / h for p in nbrs])).S
        σ = s[end] / s[1]
        nval += 1
        σ < minσ && (minσ = σ)
        σ < 1e-8 && (nsing += 1)
    end
    return (; nsing, minσ, nval)
end

"""Coverage check the d_NN gate cannot do: distance from ~nprobes interior
points to the nearest cloud point, in units of Δ. Own RNG — leaves the global
experiment stream untouched."""
function fill_distance(coords, octree; nprobes=20_000)
    rng = MersenneTwister(424242)
    lo, hi = octree.mesh_bbox_min, octree.mesh_bbox_max
    tree = KDTree(coords)
    dists = Vector{Float64}(undef, nprobes)
    got = 0
    while got < nprobes
        q = lo .+ rand(rng, SVector{3,Float64}) .* (hi .- lo)
        isinside(q, octree) || continue
        got += 1
        _, ds = knn(tree, q, 1)
        dists[got] = ds[1]
    end
    return (; max=maximum(dists) / Δ, p99=quantile(dists, 0.99) / Δ)
end

"""Per-point displacement of surviving volume points, in units of Δ — the
point-correspondence quantity that decides warm-starting RBF-FD weights."""
function displacement_stats(old_coords, new_coords)
    d = [norm(a - b) for (a, b) in zip(old_coords, new_coords)] ./ Δ
    frac(t) = count(<(t), d) / length(d)
    return (; mean=mean(d), median=median(d), p95=quantile(d, 0.95),
        max=maximum(d), f010=frac(0.10), f025=frac(0.25), f050=frac(0.50))
end

# ============================================================================
# Arms
# ============================================================================

function arm_reseed(mesh, bnd, max_pts)
    t = @elapsed begin
        alg = Octree(mesh; spacing=SPACING, alpha=1.0, placement=:bridson)
        cloud = discretize(deepcopy(bnd), SPACING; alg, max_points=max_pts)
    end
    return cloud, t
end

function arm_warm(bnd, vol_pts, octree_new)
    cloud0 = PointCloud(deepcopy(bnd), PointVolume(copy(vol_pts)))
    conv = Float64[]
    t = @elapsed cloud = repel(cloud0, SPACING, octree_new;
        max_iters=300, tol=1.0e-4, kick_after=10, cv_target=0.07,
        stall_after=50, cull_ratio=0.5, convergence=conv)
    return cloud, t, length(conv)
end

function report_arm(label, cloud, octree, pred; wall=NaN, iters=-1,
        n_before=-1, n_out0=-1, disp=nothing)
    g = gate_metrics(cloud)
    s = singular_stencils(cloud)
    f = fill_distance(g.coords, octree)
    nw = near_wall_metrics(g, pred)
    culled = n_before >= 0 ? n_before - g.n : 0
    vol_pts = WhatsThePoint.volume(cloud).points
    n_out = count(!, isinside(vol_pts, octree))
    verdict = s.nsing > 0 ? "FAIL" :
              (g.sep > 0.1 && g.cv < 0.15) ? "PASS" : "MARGINAL"

    @printf("  %-22s %7.2fs %s  N=%d (bnd %d)  culled=%d  outside=%d\n",
        label, wall, iters >= 0 ? @sprintf("%4d it", iters) : "       ", g.n, g.nb,
        culled, n_out)
    @printf("    sep/Δ=%.3f  CV=%.3f  mean=%.3f  coord=%.1f  singular=%d (minσ %.1e)\n",
        g.sep, g.cv, g.mean, g.coord, s.nsing, s.minσ)
    @printf("    fill max=%.2fΔ p99=%.2fΔ   near-wall (n=%d): mean=%.3f CV=%.3f min=%.3f\n",
        f.max, f.p99, nw.n, nw.mean, nw.cv, nw.min)
    if disp !== nothing
        @printf("    displacement/Δ: med=%.3f mean=%.3f p95=%.3f max=%.2f   <0.1Δ %.0f%%  <0.25Δ %.0f%%  <0.5Δ %.0f%%\n",
            disp.median, disp.mean, disp.p95, disp.max,
            100disp.f010, 100disp.f025, 100disp.f050)
    end
    println("    verdict: $verdict")
    return (; label, wall, iters, g.n, g.nb, culled, n_out0, n_out,
        g.sep, g.cv, g.mean, g.coord, s.nsing, fmax=f.max, fp99=f.p99,
        nw_cv=nw.cv, nw_min=nw.min, disp, verdict)
end

# ============================================================================
# Rung driver
# ============================================================================

function run_rung(idx, rung, old_vol_pts, rows)
    println("\n", "="^84)
    @printf("RUNG %d: %s  (r_inner %.3f → %.3f, outer stretch %s)\n",
        idx, rung.label, R_INNER_BASE, rung.r_inner, rung.stretch)
    println("="^84)

    t_mesh = @elapsed mesh = _make_cavity_mesh(rung.r_inner, rung.stretch)
    t_tri = @elapsed octree = TriangleOctree(mesh; classify_leaves=true)
    validate_geometry(rung.label, mesh, octree, rung.r_inner, rung.stretch)

    Random.seed!(SEED + 1000 * idx)
    t_bnd = @elapsed bnd = PointBoundary(mesh, SPACING)
    max_pts = _analytic_max_points(rung.r_inner, rung.stretch)
    @printf("  shared: mesh %.2fs, TriangleOctree %.2fs, boundary %d pts %.2fs  (cap %d vol pts)\n",
        t_mesh, t_tri, length(points(bnd)), t_bnd, max_pts)

    old_coords = _raw.(old_vol_pts)
    t_filter = @elapsed inside0 = isinside(old_vol_pts, octree)
    n_out0 = count(!, inside0)
    @printf("  warm-start stress: %d of %d old volume points outside the new domain (isinside pass %.3fs)\n",
        n_out0, length(old_vol_pts), t_filter)

    # --- A: re-seed from scratch (TriangleOctree not needed: charged to B/B2)
    Random.seed!(SEED + 1000 * idx + 1)
    cloudA, tA = arm_reseed(mesh, bnd, max_pts)
    rowA = report_arm("A re-seed", cloudA, octree, rung.pred; wall=t_bnd + tA)
    push!(rows, (; rung=rung.label, rowA...))

    # --- B: warm start with ALL old volume points (spec'd stress case)
    Random.seed!(SEED + 1000 * idx + 2)
    cloudB, tB, itB = arm_warm(bnd, old_vol_pts, octree)
    nb_new = length(points(bnd))
    n_beforeB = nb_new + length(old_vol_pts)
    volB = WhatsThePoint.volume(cloudB).points
    dispB = length(volB) == length(old_vol_pts) ?
            displacement_stats(old_coords, _raw.(volB)) : nothing
    dispB === nothing && length(volB) != length(old_vol_pts) &&
        println("  [B] point count changed — index correspondence broken, displacement stats skipped")
    rowB = report_arm("B warm (unfiltered)", cloudB, octree, rung.pred;
        wall=t_bnd + t_tri + tB, iters=itB, n_before=n_beforeB, n_out0, disp=dispB)
    push!(rows, (; rung=rung.label, rowB...))

    # --- B2: warm start, stranded-outside points dropped first
    if n_out0 > 0
        kept = old_vol_pts[inside0]
        kept_coords = old_coords[inside0]
        Random.seed!(SEED + 1000 * idx + 3)
        cloudB2, tB2, itB2 = arm_warm(bnd, kept, octree)
        volB2 = WhatsThePoint.volume(cloudB2).points
        dispB2 = length(volB2) == length(kept) ?
                 displacement_stats(kept_coords, _raw.(volB2)) : nothing
        rowB2 = report_arm("B2 warm (filtered)", cloudB2, octree, rung.pred;
            wall=t_bnd + t_tri + t_filter + tB2, iters=itB2,
            n_before=nb_new + length(kept), n_out0, disp=dispB2)
        push!(rows, (; rung=rung.label, rowB2...))
        @printf("  [B2] surviving old-volume fraction: %.3f (%d of %d)\n",
            length(kept) / length(old_vol_pts), length(kept), length(old_vol_pts))
    else
        println("  [B2] skipped — no old volume points outside the new domain")
    end
    return nothing
end

# ============================================================================

function warmup()
    println("-- warmup (compilation; coarse Δ-equivalent problem, timings discarded) --")
    mesh = _make_cavity_mesh(0.55, (1.0, 1.0, 1.0))
    octree = TriangleOctree(mesh; classify_leaves=true)
    Random.seed!(1)
    spacing_w = ConstantSpacing(0.25 * m)
    bnd = PointBoundary(mesh, spacing_w)
    alg = Octree(mesh; spacing=spacing_w, alpha=1.0, placement=:bridson)
    cloud = discretize(deepcopy(bnd), spacing_w; max_points=500, alg)
    repel(cloud, spacing_w, octree;
        max_iters=3, kick_after=2, cv_target=0.07, stall_after=50, cull_ratio=0.5)
    g = gate_metrics(cloud)
    singular_stencils(cloud)
    fill_distance(g.coords, octree; nprobes=200)
    isinside(WhatsThePoint.volume(cloud).points, octree)
    displacement_stats(g.coords, g.coords)
    return nothing
end

function main()
    println("Threads: ", Threads.nthreads())
    println("Δ=$Δ  base cavity: R_outer=$R_OUTER r_inner=$R_INNER_BASE  seed=$SEED")
    warmup()

    println("\n", "="^84)
    println("BASE STATE (both arms start from this cloud)")
    println("="^84)
    t_mesh0 = @elapsed mesh0 = _make_cavity_mesh(R_INNER_BASE, (1.0, 1.0, 1.0))
    t_tri0 = @elapsed octree0 = TriangleOctree(mesh0; classify_leaves=true)
    validate_geometry("base", mesh0, octree0, R_INNER_BASE, (1.0, 1.0, 1.0))
    Random.seed!(SEED)
    t_bnd0 = @elapsed bnd0 = PointBoundary(mesh0, SPACING)
    cloud0, t0 = arm_reseed(mesh0, bnd0, _analytic_max_points(R_INNER_BASE, (1.0, 1.0, 1.0)))
    @printf("  built in %.2fs (mesh %.2fs, octree %.2fs, boundary %.2fs, volume %.2fs)\n",
        t_mesh0 + t_tri0 + t_bnd0 + t0, t_mesh0, t_tri0, t_bnd0, t0)
    pred0 = c -> norm(c) < R_INNER_BASE + 2Δ
    report_arm("base direct pipeline", cloud0, octree0, pred0; wall=t_bnd0 + t0)
    old_vol_pts = WhatsThePoint.volume(cloud0).points

    rungs = [
        (label="r_inner 0.550 (+0.5%)", r_inner=0.550, stretch=(1.0, 1.0, 1.0),
            pred=c -> norm(c) < 0.550 + 2Δ),
        (label="r_inner 0.560 (+2.4%)", r_inner=0.560, stretch=(1.0, 1.0, 1.0),
            pred=c -> norm(c) < 0.560 + 2Δ),
        (label="r_inner 0.600 (+9.7%)", r_inner=0.600, stretch=(1.0, 1.0, 1.0),
            pred=c -> norm(c) < 0.600 + 2Δ),
        (label="outer x-stretch 1.05", r_inner=R_INNER_BASE, stretch=(1.05, 1.0, 1.0),
            pred=c -> sqrt((c[1] / 1.05)^2 + c[2]^2 + c[3]^2) > R_OUTER - 2Δ),
    ]

    rows = NamedTuple[]
    for (i, rung) in enumerate(rungs)
        run_rung(i, rung, old_vol_pts, rows)
    end

    println("\n", "="^84)
    println("SUMMARY  (wall = boundary sample + arm-specific work; B includes TriangleOctree)")
    println("="^84)
    @printf("%-22s %-22s %7s %5s %6s %6s %6s %5s %5s %6s %6s %7s %9s\n",
        "rung", "arm", "wall", "iters", "N", "sep/Δ", "CV", "coord", "sing",
        "fmax/Δ", "out", "culled", "verdict")
    for r in rows
        @printf("%-22s %-22s %6.2fs %5s %6d %6.3f %6.3f %5.1f %5d %6.2f %6d %7d %9s\n",
            r.rung, r.label, r.wall, r.iters >= 0 ? string(r.iters) : "-", r.n,
            r.sep, r.cv, r.coord, r.nsing, r.fmax, r.n_out, r.culled, r.verdict)
    end
    println("\ncorrespondence (B arms): fraction of surviving volume points displaced <0.25Δ")
    for r in rows
        r.disp === nothing && continue
        @printf("  %-22s %-22s  med=%.3fΔ  <0.1Δ %.0f%%  <0.25Δ %.0f%%  <0.5Δ %.0f%%\n",
            r.rung, r.label, r.disp.median, 100r.disp.f010, 100r.disp.f025, 100r.disp.f050)
    end
    return nothing
end

main()
