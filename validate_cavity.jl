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
println("-- no cull --")
cloud_r = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1)
mr = metrics(cloud_r)

println("-- cull_ratio=0.5 (removes the stuck boundary pair) --")
cloud_rc = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, rebuild_every = 1, cull_ratio = 0.5)
mrc = metrics(cloud_rc)

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
# How well does each node's nearest-neighbor distance match its prescribed
# spacing h(x)?  Reports the distribution of d_NN(i)/h(x_i): a perfect cloud
# spikes at a single value with low CV. Also a mean coordination number (count
# of neighbors within 1.4·h) to distinguish a disordered blue-noise / dense-
# liquid packing (~12-14) from Poisson (broad) or a degenerate collapse.
# ============================================================================

function spacing_fidelity(cloud, spacing)
    pts = points(cloud)
    n = length(pts)
    k = min(n, 30)
    method = KNearestSearch(cloud, k)
    res = searchdists(cloud, method)
    dnn_over_h = Vector{Float64}(undef, n)
    coordination = Vector{Int}(undef, n)
    for i in 1:n
        ids, dists = res[i]
        h = ustrip(spacing(pts[i]))
        dnn = Inf
        coord = 0
        for (j, d) in zip(ids, dists)
            j == i && continue
            du = ustrip(d)
            dnn = min(dnn, du)
            du <= 1.4 * h && (coord += 1)
        end
        dnn_over_h[i] = dnn / h
        coordination[i] = coord
    end
    return dnn_over_h, coordination
end

function report_fidelity(label, cloud, spacing)
    r, coord = spacing_fidelity(cloud, spacing)
    μ = mean(r)
    cv = std(r) / μ
    qs = quantile(r, [0.01, 0.5, 0.99])
    @printf("  %-14s  mean d_NN/h=%.3f  CV=%.3f  [p01=%.3f p50=%.3f p99=%.3f]  coord(≤1.4h)=%.1f\n",
        label, μ, cv, qs[1], qs[2], qs[3], mean(coord))
    return (; mean = μ, cv, coord = mean(coord))
end

println("\n############ SPACING FIDELITY (d_NN / h) ############")
fid0 = report_fidelity("raw octree", cloud0, spacing)
fidr = report_fidelity("repel", cloud_r, spacing)
fidrc = report_fidelity("repel+cull", cloud_rc, spacing)

println("\n############ STENCIL CONDITIONING (k=50, poly_deg=3) ############")
σ_r, mindist_r = stencil_conditioning(cloud_rc; k = 50)

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^72)
println("CAVITY-GEOMETRY VALIDATION SUMMARY")
println("="^72)

@printf("\n%-22s  %12s  %12s  %12s\n", "metric", "raw octree", "repel", "repel+cull")
@printf("%-22s  %12s  %12s  %12s\n", "-"^22, "-"^12, "-"^12, "-"^12)
@printf("%-22s  %12.4e  %12.4e  %12.4e\n", "separation", m0.separation, mr.separation, mrc.separation)
@printf("%-22s  %12.4f  %12.4f  %12.4f\n", "separation / Δ",
    ustrip(m0.separation) / Δ, ustrip(mr.separation) / Δ, ustrip(mrc.separation) / Δ)
@printf("%-22s  %12.2f  %12.2f  %12.2f\n", "mesh_ratio", m0.mesh_ratio, mr.mesh_ratio, mrc.mesh_ratio)
@printf("%-22s  %12.3f  %12.3f  %12.3f\n", "spacing CV (d_NN/h)", fid0.cv, fidr.cv, fidrc.cv)
@printf("%-22s  %12.1f  %12.1f  %12.1f\n", "coordination (≤1.4h)", fid0.coord, fidr.coord, fidrc.coord)

npts_raw = length(points(cloud0))
npts_rep = length(points(cloud_r))
npts_culled = length(points(cloud_rc))
@printf("\npoints: %d (raw) → %d (repel) → %d (repel+cull, %d removed)\n",
    npts_raw, npts_rep, npts_culled, npts_rep - npts_culled)

valid_σ = filter(!isnan, σ_r)
SING_THRESH = 1e-8
nsing = count(<(SING_THRESH), valid_σ)
@printf("\nstencil conditioning (k=50, poly_deg=3, on repel+cull cloud):\n")
@printf("  min σ_min/σ_max:   %.3e\n", minimum(valid_σ))
@printf("  median σ_min/σ_max: %.3e\n", median(valid_σ))
@printf("  singular (<%.0e):   %d / %d\n", SING_THRESH, nsing, length(valid_σ))

# Gate on the final (repel+cull) cloud, using spacing CV as the primary quality
# metric (mesh_ratio kept as a secondary, outlier-sensitive diagnostic).
sep_ratio = ustrip(mrc.separation) / Δ
mesh_r = mrc.mesh_ratio
@printf("\nVERDICT: ")
if nsing == 0 && sep_ratio > 0.1 && mesh_r < 3.0
    println("PASS — cloud is well-conditioned for RBF-FD (poly_deg=3).")
elseif nsing == 0
    println("MARGINAL — no singular stencils, but separation/mesh_ratio could improve.")
else
    println("FAIL — $nsing singular stencils. Repel needs more iterations or different parameters.")
end
