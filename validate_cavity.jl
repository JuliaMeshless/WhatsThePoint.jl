# validate_cavity.jl — Cavity-geometry validation for WhatsThePoint node generation.
#
# Tests the Octree→repel pipeline on an annular cavity domain (outer sphere minus
# inner sphere) that mirrors the Macchiato.jl cavity_sphere_recovery geometry.
# Measures: separation, mesh_ratio, spacing CV, per-stencil conditioning.
#
# Usage:  jlrun validate_cavity.jl
#         jlrun validate_cavity.jl 0.06     (custom spacing)
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
    pts = [Meshes.Point(R * sin(π * i / nθ) * cos(2π * j / nφ),
                        R * sin(π * i / nθ) * sin(2π * j / nφ),
                        R * cos(π * i / nθ))
           for i in 0:nθ, j in 0:(nφ - 1)][:]

    conn = Connectivity{Triangle}[]
    for i in 0:(nθ - 1), j in 0:(nφ - 1)
        a = i * nφ + j + 1
        b = i * nφ + (j + 1) % nφ + 1
        c = (i + 1) * nφ + j + 1
        d = (i + 1) * nφ + (j + 1) % nφ + 1
        push!(conn, connect((a, b, c)))
        push!(conn, connect((b, d, c)))
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

h_input = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.08
const Δ = h_input

@printf("Annular cavity: R_outer=%.3f  r_inner=%.3f  Δ=%.4f\n", R_OUTER, R_INNER, Δ)
@printf("Building mesh (%d×%d per shell)...\n", N_ANGULAR, 2 * N_ANGULAR)

mesh = _make_annular_cavity_mesh(R_OUTER, R_INNER, N_ANGULAR, 2 * N_ANGULAR)
@printf("Mesh: %d vertices, %d triangles\n",
    length(Meshes.vertices(mesh)), Meshes.nelements(mesh))

boundary = PointBoundary(mesh)
diag = norm(Meshes.boundingbox(mesh).max - Meshes.boundingbox(mesh).min)
spacing = ConstantSpacing(Δ * m)

# ============================================================================
# Octree discretization
# ============================================================================

println("\n############ OCTREE DISCRETIZATION ############")
octree = TriangleOctree(mesh; classify_leaves = true)
alg = Octree(mesh; spacing, alpha = 1.0, placement = :random)

max_pts = round(Int, (4 / 3) * π * (R_OUTER^3 - R_INNER^3) / Δ^3)
@printf("Expected volume nodes ≈ %d\n", max_pts)

cloud0 = discretize(boundary, spacing; alg, max_points = min(max_pts, 50_000))
m0 = metrics(cloud0)

# ============================================================================
# Repel
# ============================================================================

println("\n############ REPEL (live tree, SpacingEquilibriumForce) ############")
println("-- no kick, no cull --")
cloud_r = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1)
mr = metrics(cloud_r)

println("-- kick_after=10 --")
cloud_rk = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1, kick_after = 10)
mrk = metrics(cloud_rk)

println("-- kick_after=10 + cull_ratio=0.5 --")
cloud_rkc = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1, kick_after = 10, cull_ratio = 0.5)
mrkc = metrics(cloud_rkc)

# ============================================================================
# Per-stencil conditioning (degree-3 3D Vandermonde σ_min/σ_max)
# ============================================================================

function vander3(P)
    n = length(P)
    V = Matrix{Float64}(undef, n, 20)
    for (r, p) in enumerate(P)
        x, y, z = p[1], p[2], p[3]
        V[r, :] .= (1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z,
            x^3, y^3, z^3, x^2*y, x^2*z, y^2*x, y^2*z, z^2*x, z^2*y, x*y*z)
    end
    return V
end

function stencil_conditioning(cloud; k = 50)
    pts_raw = points(cloud)
    pts = [let v = Meshes.to(p); SVector(Float64(ustrip(v[1])), Float64(ustrip(v[2])), Float64(ustrip(v[3]))) end
           for p in pts_raw]
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
    @printf("  %-18s  mean=%.3f  CV=%.3f  p05=%.3f p50=%.3f p95=%.3f  coord=%.1f\n",
        label, m.mean_dnn_h, m.cv, m.p05, m.p50, m.p95, m.coordination)
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

@printf("\n%-22s  %10s  %10s  %10s  %14s\n",
    "metric", "raw octree", "repel", "repel+kick", "repel+kick+cull")
@printf("%-22s  %10s  %10s  %10s  %14s\n", "-"^22, "-"^10, "-"^10, "-"^10, "-"^14)
@printf("%-22s  %10.4f  %10.4f  %10.4f  %14.4f\n", "separation / Δ",
    ustrip(m0.separation) / Δ, ustrip(mr.separation) / Δ,
    ustrip(mrk.separation) / Δ, ustrip(mrkc.separation) / Δ)
@printf("%-22s  %10.3f  %10.3f  %10.3f  %14.3f\n", "spacing CV",
    fid0.cv, fidr.cv, fidrk.cv, fidrkc.cv)
@printf("%-22s  %10.3f  %10.3f  %10.3f  %14.3f\n", "p05 (d_NN/h)",
    fid0.p05, fidr.p05, fidrk.p05, fidrkc.p05)
@printf("%-22s  %10.3f  %10.3f  %10.3f  %14.3f\n", "p95 (d_NN/h)",
    fid0.p95, fidr.p95, fidrk.p95, fidrkc.p95)
@printf("%-22s  %10.1f  %10.1f  %10.1f  %14.1f\n", "coordination (≤1.4h)",
    fid0.coordination, fidr.coordination, fidrk.coordination, fidrkc.coordination)

npts_raw = length(points(cloud0))
npts_rep = length(points(cloud_r))
npts_kick = length(points(cloud_rk))
npts_culled = length(points(cloud_rkc))
@printf("\npoints: %d (raw) → %d (repel) → %d (kick) → %d (kick+cull, %d removed)\n",
    npts_raw, npts_rep, npts_kick, npts_culled, npts_kick - npts_culled)

valid_σ = filter(!isnan, σ_rkc)
SING_THRESH = 1e-8
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
    @printf("  separation/Δ=%.3f (>0.1 ✓)  CV=%.3f (<0.15 ✓)  singular=%d\n",
        sep_ratio, fid_final.cv, nsing)
elseif nsing == 0
    println("MARGINAL — no singular stencils, but quality metrics need improvement.")
    @printf("  separation/Δ=%.3f %s  CV=%.3f %s  singular=%d\n",
        sep_ratio, sep_ratio > 0.1 ? "(✓)" : "(✗)",
        fid_final.cv, fid_final.cv < 0.15 ? "(✓)" : "(✗)", nsing)
else
    println("FAIL — $nsing singular stencils. Repel needs more iterations or different parameters.")
end
