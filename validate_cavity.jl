# validate_cavity.jl — Cavity-geometry validation for WhatsThePoint node generation.
#
# Tests the Octree→repel pipeline on an annular cavity domain (outer sphere minus
# inner sphere) that mirrors the Macchiato.jl cavity_sphere_recovery geometry.
# Measures: separation, mesh_ratio, spacing CV, per-stencil conditioning.
#
# The geometry is built programmatically, exported once to test/data/cavity.stl,
# and re-imported from the STL so the gate exercises the production path
# (GeoIO import, face-center sampling, Float32 STL precision).
#
# Usage:  jlrun validate_cavity.jl
#         jlrun validate_cavity.jl 0.06                  (custom spacing)
#         jlrun validate_cavity.jl --placement=jittered  (octree seeding: random|jittered|lattice|bridson)
#         jlrun validate_cavity.jl --resample-boundary   (Poisson-disk surface sampling instead of face centers)
#         jlrun validate_cavity.jl --save-vtk            (dump final cloud to cavity_cloud.vtu for ParaView)
using Pkg
Pkg.activate(@__DIR__)

using WhatsThePoint
using GeoIO, Meshes
using Unitful: m, ustrip
using StaticArrays: SVector
using LinearAlgebra: norm, dot, svd
using Statistics: mean, std, median, quantile
using Printf

# ============================================================================
# Geometry: build annular cavity (outer sphere - inner sphere) as SimpleMesh
# ============================================================================

function _make_sphere_mesh(R, nθ, nφ)
    # NOTE index layout: the connectivity below addresses point (i, j) at linear
    # index i*nφ + j + 1 (row-major over i). Julia comprehensions flatten
    # column-major, so j must be the FIRST iterator. Having i first scrambled
    # the connectivity (608 chord triangles through the interior, 6× the true
    # surface area) and silently corrupted every STL-gate run until 2026-06-11
    # — the d_NN-based gate metrics are blind to it. Guarded by the area check
    # at export below.
    pts = [
        Meshes.Point(
                R * sin(π * i / nθ) * cos(2π * j / nφ),
                R * sin(π * i / nθ) * sin(2π * j / nφ),
                R * cos(π * i / nθ)
            )
            for j in 0:(nφ - 1), i in 0:nθ
    ][:]

    conn = Connectivity{Triangle}[]
    for i in 0:(nθ - 1), j in 0:(nφ - 1)
        a = i * nφ + j + 1
        b = i * nφ + (j + 1) % nφ + 1
        c = (i + 1) * nφ + j + 1
        d = (i + 1) * nφ + (j + 1) % nφ + 1
        # Pole rows collapse one quad edge (a==b at the north pole, c==d at the
        # south pole): emit only the non-degenerate triangle there. The
        # zero-area slivers otherwise fail the manifold-orientation edge test
        # and put near-duplicate face centers at the poles.
        # Winding order makes Meshes.normal point OUTWARD (away from the
        # sphere center). Discovered 2026-06-11 (shape-opt trade-off session):
        # the previous order (a,b,c)/(b,d,c) wound INWARD, so the assembled
        # cavity was inside-out — the signed-distance classification put the
        # solid annulus OUTSIDE and the complement (inner hole + bbox corners)
        # INSIDE, and every volume fill landed in the complement. The
        # d_NN-based gate metrics cannot see that (second corruption in one
        # day with the same blind spot); the area check cannot see orientation
        # either. Guarded by the isinside probes after octree construction.
        i > 0 && push!(conn, connect((a, c, b)))
        i < nθ - 1 && push!(conn, connect((b, c, d)))
    end

    return SimpleMesh(pts, conn)
end

function _make_annular_cavity_mesh(R_outer, r_inner, nθ, nφ)
    outer = _make_sphere_mesh(R_outer, nθ, nφ)
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

# ============================================================================
# Config
# ============================================================================

const R_OUTER = 1.0
const R_INNER = 0.547
const N_ANGULAR = 24
const STL_PATH = joinpath(@__DIR__, "test", "data", "cavity.stl")

function _parse_args(args)
    Δ = 0.08
    placement = :random
    save_vtk = false
    resample = false
    for a in args
        if startswith(a, "--placement=")
            placement = Symbol(last(split(a, "=")))
        elseif a == "--save-vtk"
            save_vtk = true
        elseif a == "--resample-boundary"
            resample = true
        else
            Δ = parse(Float64, a)
        end
    end
    return Δ, placement, save_vtk, resample
end

const Δ, PLACEMENT, SAVE_VTK, RESAMPLE = _parse_args(ARGS)

@printf(
    "Annular cavity: R_outer=%.3f  r_inner=%.3f  Δ=%.4f  placement=%s  boundary=%s\n",
    R_OUTER, R_INNER, Δ, PLACEMENT, RESAMPLE ? "poisson-disk" : "face-centers"
)

# ----------------------------------------------------------------------------
# STL roundtrip: export the programmatic mesh once, then always import from
# the STL so the gate runs the production path. Binary STL stores Float32 by
# spec; promote to Float64 (same workaround as validate_repel.jl — the proper
# octree.jl fix is tracked in NODEGEN_FINDINGS.md housekeeping).
# ----------------------------------------------------------------------------

if !isfile(STL_PATH)
    @printf(
        "Building mesh (%d×%d per shell) and exporting to %s\n",
        N_ANGULAR, 2 * N_ANGULAR, STL_PATH
    )
    mesh_src = _make_annular_cavity_mesh(R_OUTER, R_INNER, N_ANGULAR, 2 * N_ANGULAR)

    # Geometry sanity: total area must match the analytic value (tessellation
    # deficit only). Catches connectivity scrambling, which the d_NN-based gate
    # metrics cannot see (discovered 2026-06-11: a column-major/row-major index
    # mismatch produced chord triangles with 6× the true area and a broken
    # isinside, while every gate metric still looked plausible).
    area_mesh = sum(ustrip(Meshes.area(e)) for e in Meshes.elements(mesh_src))
    area_true = 4π * (R_OUTER^2 + R_INNER^2)
    rel_err = abs(area_mesh - area_true) / area_true
    @printf("mesh area: %.4f vs analytic %.4f (rel err %.2e)\n", area_mesh, area_true, rel_err)
    rel_err < 0.02 || error("mesh area deviates $(rel_err) from analytic — connectivity bug?")

    GeoIO.save(STL_PATH, GeoIO.georef(nothing, mesh_src))

    # one-time verification: face centers must survive the roundtrip (Float32 noise only)
    back = GeoIO.load(STL_PATH).geometry
    c_src = [Float64.(ustrip.(Meshes.to(Meshes.centroid(e)))) for e in Meshes.elements(mesh_src)]
    c_stl = [Float64.(ustrip.(Meshes.to(Meshes.centroid(e)))) for e in Meshes.elements(back)]
    maxdev = maximum(minimum(norm(a .- b) for b in c_stl) for a in c_src)
    @printf(
        "STL roundtrip: %d → %d triangles, max face-center deviation %.2e\n",
        length(c_src), length(c_stl), maxdev
    )
    maxdev < 1.0e-5 || error("STL roundtrip deviation too large: $maxdev")
    area_back = sum(ustrip(Meshes.area(e)) for e in Meshes.elements(back))
    abs(area_back - area_true) / area_true < 0.02 ||
        error("loaded STL area $(area_back) deviates from analytic $(area_true)")
end

mesh_raw = GeoIO.load(STL_PATH).geometry
mesh = Meshes.SimpleMesh(
    [Meshes.Point((1.0 .* Meshes.to(v))...) for v in Meshes.vertices(mesh_raw)],
    Meshes.topology(mesh_raw),
)
@printf(
    "Loaded %s: %d vertices, %d triangles\n",
    STL_PATH, length(Meshes.vertices(mesh)), Meshes.nelements(mesh)
)

diag = norm(Meshes.boundingbox(mesh).max - Meshes.boundingbox(mesh).min)
spacing = ConstantSpacing(Δ * m)
boundary = RESAMPLE ? PointBoundary(mesh, spacing) : PointBoundary(mesh)
@printf(
    "Boundary: %d points (%s)\n", length(points(boundary)),
    RESAMPLE ? "Poisson-disk sampled" : "face centers"
)

# ============================================================================
# Octree discretization
# ============================================================================

println("\n############ OCTREE DISCRETIZATION ############")
octree = TriangleOctree(mesh; classify_leaves = true)

# Orientation guard (2026-06-11): area checks cannot see winding orientation;
# these probes can. A cavity.stl written by the old inward-winding generator
# fails here — delete the file and re-run to regenerate it correctly.
isinside(SVector(0.5 * (R_INNER + R_OUTER), 0.01, 0.02), octree) || error(
    "mid-annulus probe classified OUTSIDE — cavity.stl is inside-out (old " *
        "generator winding); delete test/data/cavity.stl and re-run to regenerate"
)
!isinside(SVector(0.5 * R_INNER, 0.01, 0.02), octree) || error(
    "inner-hole probe classified INSIDE — cavity.stl is inside-out (old " *
        "generator winding); delete test/data/cavity.stl and re-run to regenerate"
)
alg = Octree(mesh; spacing, alpha = 1.0, placement = PLACEMENT)

max_pts = round(Int, (4 / 3) * π * (R_OUTER^3 - R_INNER^3) / Δ^3)
@printf("Expected volume nodes ≈ %d\n", max_pts)

cloud0 = discretize(boundary, spacing; alg, max_points = min(max_pts, 50_000))
m0 = metrics(cloud0)

# ============================================================================
# Repel
# ============================================================================

println("\n############ REPEL (live tree, SpacingEquilibriumForce) ############")
println("-- no kick, no cull --")
conv_r = Float64[]
t_r = @elapsed cloud_r = repel(
    cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1, convergence = conv_r
)
mr = metrics(cloud_r)
@printf("   %d iterations, %.1f s, final residual %.3e\n", length(conv_r), t_r, last(conv_r))

println("-- kick_after=10 --")
conv_rk = Float64[]
t_rk = @elapsed cloud_rk = repel(
    cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1, kick_after = 10, convergence = conv_rk
)
mrk = metrics(cloud_rk)
@printf("   %d iterations, %.1f s, final residual %.3e\n", length(conv_rk), t_rk, last(conv_rk))

println("-- kick_after=10 + cull_ratio=0.5 --")
conv_rkc = Float64[]
t_rkc = @elapsed cloud_rkc = repel(
    cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1, kick_after = 10, cull_ratio = 0.5,
    convergence = conv_rkc
)
mrkc = metrics(cloud_rkc)
@printf("   %d iterations, %.1f s, final residual %.3e\n", length(conv_rkc), t_rkc, last(conv_rkc))

# ============================================================================
# Per-stencil conditioning (degree-3 3D Vandermonde σ_min/σ_max)
# ============================================================================

function vander3(P)
    n = length(P)
    V = Matrix{Float64}(undef, n, 20)
    for (r, p) in enumerate(P)
        x, y, z = p[1], p[2], p[3]
        V[r, :] .= (
            1, x, y, z, x^2, y^2, z^2, x * y, x * z, y * z,
            x^3, y^3, z^3, x^2 * y, x^2 * z, y^2 * x, y^2 * z, z^2 * x, z^2 * y, x * y * z,
        )
    end
    return V
end

function stencil_conditioning(cloud; k = 50)
    pts_raw = points(cloud)
    pts = [
        let v = Meshes.to(p)
                SVector(Float64(ustrip(v[1])), Float64(ustrip(v[2])), Float64(ustrip(v[3])))
        end
            for p in pts_raw
    ]
    N = length(pts)
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)

    σ = fill(NaN, N)
    mindist = fill(Inf, N)
    for i in 1:N
        ids = results[i][1]
        nbrs = [pts[j] for j in ids if j != i]
        length(nbrs) < 20 && continue

        c = sum(nbrs) / length(nbrs)
        h = maximum(norm(p - c) for p in nbrs)
        V = vander3([(p - c) / h for p in nbrs])
        s = svd(V).S
        σ[i] = s[end] / s[1]

        mindist[i] = minimum(norm(pts[i] - pts[j]) for j in ids if j != i)
    end
    return σ, mindist
end

# ============================================================================
# Spacing fidelity — the primary node-placement quality metric.
# Uses spacing_fidelity_metrics from WhatsThePoint.
# ============================================================================

function report_fidelity(label, cloud, spacing)
    m = spacing_fidelity_metrics(cloud, spacing)
    @printf(
        "  %-18s  mean=%.3f  CV=%.3f  p05=%.3f p50=%.3f p95=%.3f  coord=%.1f\n",
        label, m.mean_dnn_h, m.cv, m.p05, m.p50, m.p95, m.coordination
    )
    return m
end

println("\n############ SPACING FIDELITY (d_NN / h) ############")
fid0 = report_fidelity("raw octree", cloud0, spacing)
fidr = report_fidelity("repel", cloud_r, spacing)
fidrk = report_fidelity("repel+kick", cloud_rk, spacing)
fidrkc = report_fidelity("repel+kick+cull", cloud_rkc, spacing)

println("\n############ STENCIL CONDITIONING (k=50, poly_deg=3) ############")
σ_rkc, mindist_rkc = stencil_conditioning(cloud_rkc; k = 50)

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^84)
println("CAVITY-GEOMETRY VALIDATION SUMMARY")
println("="^84)

@printf(
    "\n%-22s  %10s  %10s  %10s  %14s\n",
    "metric", "raw octree", "repel", "repel+kick", "repel+kick+cull"
)
@printf("%-22s  %10s  %10s  %10s  %14s\n", "-"^22, "-"^10, "-"^10, "-"^10, "-"^14)
@printf(
    "%-22s  %10.4f  %10.4f  %10.4f  %14.4f\n", "separation / Δ",
    ustrip(m0.separation) / Δ, ustrip(mr.separation) / Δ,
    ustrip(mrk.separation) / Δ, ustrip(mrkc.separation) / Δ
)
@printf(
    "%-22s  %10.3f  %10.3f  %10.3f  %14.3f\n", "spacing CV",
    fid0.cv, fidr.cv, fidrk.cv, fidrkc.cv
)
@printf(
    "%-22s  %10.3f  %10.3f  %10.3f  %14.3f\n", "p05 (d_NN/h)",
    fid0.p05, fidr.p05, fidrk.p05, fidrkc.p05
)
@printf(
    "%-22s  %10.3f  %10.3f  %10.3f  %14.3f\n", "p95 (d_NN/h)",
    fid0.p95, fidr.p95, fidrk.p95, fidrkc.p95
)
@printf(
    "%-22s  %10.1f  %10.1f  %10.1f  %14.1f\n", "coordination (≤1.4h)",
    fid0.coordination, fidr.coordination, fidrk.coordination, fidrkc.coordination
)

npts_raw = length(points(cloud0))
npts_rep = length(points(cloud_r))
npts_kick = length(points(cloud_rk))
npts_culled = length(points(cloud_rkc))
@printf(
    "\npoints: %d (raw) → %d (repel) → %d (kick) → %d (kick+cull, %d removed)\n",
    npts_raw, npts_rep, npts_kick, npts_culled, npts_kick - npts_culled
)
@printf(
    "repel iterations / wall-clock: %d / %.1fs (plain), %d / %.1fs (kick), %d / %.1fs (kick+cull)\n",
    length(conv_r), t_r, length(conv_rk), t_rk, length(conv_rkc), t_rkc
)

valid_σ = filter(!isnan, σ_rkc)
SING_THRESH = 1.0e-8
nsing = count(<(SING_THRESH), valid_σ)
@printf("\nstencil conditioning (k=50, poly_deg=3, on repel+kick+cull cloud):\n")
@printf("  min σ_min/σ_max:   %.3e\n", minimum(valid_σ))
@printf("  median σ_min/σ_max: %.3e\n", median(valid_σ))
@printf("  singular (<%.0e):   %d / %d\n", SING_THRESH, nsing, length(valid_σ))

# Gate on the final (repel+kick+cull) cloud
sep_ratio = ustrip(mrkc.separation) / Δ
fid_final = spacing_fidelity_metrics(cloud_rkc, spacing)
@printf("\nVERDICT: ")
if nsing == 0 && sep_ratio > 0.1 && fid_final.cv < 0.15
    println("PASS — cloud is well-conditioned for RBF-FD (poly_deg=3).")
    @printf(
        "  separation/Δ=%.3f (>0.1 ✓)  CV=%.3f (<0.15 ✓)  singular=%d\n",
        sep_ratio, fid_final.cv, nsing
    )
elseif nsing == 0
    println("MARGINAL — no singular stencils, but quality metrics need improvement.")
    @printf(
        "  separation/Δ=%.3f %s  CV=%.3f %s  singular=%d\n",
        sep_ratio, sep_ratio > 0.1 ? "(✓)" : "(✗)",
        fid_final.cv, fid_final.cv < 0.15 ? "(✓)" : "(✗)", nsing
    )
else
    println("FAIL — $nsing singular stencils. Repel needs more iterations or different parameters.")
end

if SAVE_VTK
    save(joinpath(@__DIR__, "cavity_cloud"), cloud_rkc; format = :vtk)
    println("\nSaved final cloud to cavity_cloud.vtu (open in ParaView alongside test/data/cavity.stl)")
end
