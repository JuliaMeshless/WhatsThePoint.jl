# diagnose_repel_quality.jl — Session: repel quality refinement (open defect 2).
#
# Question: why does repel degrade an already-blue-noise cloud
# (CV 0.071 → 0.093, sep/Δ 0.75 → 0.55 over 300 iters), and which dynamics
# change makes it improve-or-preserve instead?
#
# Experiments:
#   preserve — from the raw-PASS seeded cloud (Poisson-disk boundary + :bridson
#              volume), run 300 staged iterations under candidate force models /
#              step sizes and track gate metrics.
#   recover  — perturb every point by 0.3·h and measure iterations back to
#              gate-PASS quality (the shape-opt design-step proxy).
#
# Run single-threaded:  julia --project=. -t 1 diagnose_repel_quality.jl [preserve|recover ...]

using Pkg
Pkg.activate(@__DIR__)

using WhatsThePoint
using GeoIO, Meshes
using NearestNeighbors: KDTree, knn, inrange
using Unitful: m, ustrip
using StaticArrays: SVector
using LinearAlgebra: norm
using Statistics: mean, std
using Printf, Random, Logging
import WhatsThePoint: RepelForceModel, compute_force

const Δ = 0.08
const R_OUTER = 1.0
const R_INNER = 0.547
const STL_PATH = joinpath(@__DIR__, "test", "data", "cavity.stl")
const SEED = 20260611

# ----------------------------------------------------------------------------
# Candidate force models (script-local; promoted to src/ only if they win)
# ----------------------------------------------------------------------------

# ClippedSpacingForce now lives in src/repel_forces.jl (promoted after the
# preserve experiment confirmed the condensation diagnosis).

# Attraction kept but scaled by c — dose-response check of the same hypothesis.
struct DampedAttractionForce <: RepelForceModel
    β::Float64
    c::Float64
end
@inline function compute_force(f::DampedAttractionForce, u::Real)
    u² = u * u
    F = (1 - u²) / (u² + f.β)^2
    return F < 0 ? f.c * F : F
end

# ----------------------------------------------------------------------------
# Seeded cloud (the raw-PASS configuration from validate_cavity --placement=bridson
# --resample-boundary)
# ----------------------------------------------------------------------------

function build_seeded()
    mesh_raw = GeoIO.load(STL_PATH).geometry
    mesh = SimpleMesh(
        [Meshes.Point((1.0 .* Meshes.to(v))...) for v in Meshes.vertices(mesh_raw)],
        Meshes.topology(mesh_raw),
    )
    spacing = ConstantSpacing(Δ * m)
    octree = TriangleOctree(mesh; classify_leaves=true)
    Random.seed!(SEED)
    bnd = PointBoundary(mesh, spacing)
    alg = Octree(mesh; spacing, alpha=1.0, placement=:bridson)
    max_pts = round(Int, (4 / 3) * π * (R_OUTER^3 - R_INNER^3) / Δ^3)
    cloud = discretize(bnd, spacing; alg, max_points=max_pts)
    return cloud, spacing, octree
end

# ----------------------------------------------------------------------------
# Metrics (gate-equivalent, plus boundary/volume decomposition)
# ----------------------------------------------------------------------------

_raw(p) = Float64.(ustrip.(Meshes.to(p)))

function _dnn(pts)
    coords = _raw.(pts)
    tree = KDTree(coords)
    _, dists = knn(tree, coords, 2, true)
    return [d[2] for d in dists]
end

function gate_metrics(cloud)
    all_p = points(cloud)
    nb = length(points(WhatsThePoint.boundary(cloud)))
    dnn = _dnn(all_p) ./ Δ
    coords = _raw.(all_p)
    tree = KDTree(coords)
    coord = mean(length(inrange(tree, c, 1.4 * Δ)) - 1 for c in coords)
    cv(x) = std(x) / mean(x)
    bnd_dnn = _dnn(all_p[1:nb]) ./ Δ
    vol_dnn = _dnn(all_p[(nb+1):end]) ./ Δ
    return (;
        sep=minimum(dnn), cv=cv(dnn), mean=mean(dnn), coord,
        cv_bnd=cv(bnd_dnn), cv_vol=cv(vol_dnn), n=length(all_p), nb,
    )
end

function row(cum, cloud, resid)
    g = gate_metrics(cloud)
    @printf("  %5d  %6.3f  %6.3f  %6.3f  %5.1f  %6.3f  %6.3f  %9.3e  %6d\n",
        cum, g.sep, g.cv, g.mean, g.coord, g.cv_bnd, g.cv_vol, resid, g.n)
    return g
end

function header(label)
    println("\n== $label ==")
    @printf("  %5s  %6s  %6s  %6s  %5s  %6s  %6s  %9s  %6s\n",
        "iters", "sep/Δ", "CV", "mean", "coord", "CV_bnd", "CV_vol", "residual", "N")
end

# ----------------------------------------------------------------------------
# Staged repel driver (chained calls ≡ continuous run: kick off, no cross-call state)
# ----------------------------------------------------------------------------

quietly(f) = with_logger(() -> f(), ConsoleLogger(stderr, Logging.Error))

function staged(label, cloud0, spacing, octree; chunks=[25, 25, 50, 100, 100], kwargs...)
    header(label)
    row(0, cloud0, NaN)
    cloud, cum = cloud0, 0
    conv_all = Float64[]
    for c in chunks
        conv = Float64[]
        cloud = quietly() do
            repel(cloud, spacing, octree;
                max_iters=c, tol=1.0e-4, convergence=conv, kwargs...)
        end
        append!(conv_all, conv)
        cum += c
        row(cum, cloud, last(conv))
    end
    @printf("  residual: start %.3e  min %.3e  final %.3e\n",
        first(conv_all), minimum(conv_all), last(conv_all))
    return cloud
end

# ----------------------------------------------------------------------------
# Perturbation (shape-opt design-step proxy)
# ----------------------------------------------------------------------------

function perturb(cloud, spacing, frac)
    jiggle(p) = begin
        d = randn(SVector{3,Float64})
        p + Vec(frac * ustrip(spacing(p)) * (d / norm(d)) * m)
    end
    bnd = WhatsThePoint.boundary(cloud)
    new_bnd = PointBoundary(
        map(jiggle, points(bnd)), WhatsThePoint.normal(bnd), WhatsThePoint.area(bnd)
    )
    new_vol = PointVolume(map(jiggle, WhatsThePoint.volume(cloud).points))
    return PointCloud(new_bnd, new_vol, NoTopology())
end

# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

function experiment_preserve(cloud0, spacing, octree)
    println("\n############ PRESERVE: 300 iters from the raw-PASS cloud ############")
    staged("A baseline SpacingEquilibriumForce(0.2)", cloud0, spacing, octree;
        force_model=SpacingEquilibriumForce(0.2))
    staged("B ClippedSpacingForce(0.2, u0=1.0) — repulsion-only", cloud0, spacing, octree;
        force_model=ClippedSpacingForce(0.2, 1.0))
    staged("C DampedAttractionForce(0.2, c=0.1)", cloud0, spacing, octree;
        force_model=DampedAttractionForce(0.2, 0.1))
    staged("D ClippedSpacingForce(0.2, u0=0.9)", cloud0, spacing, octree;
        force_model=ClippedSpacingForce(0.2, 0.9))
    staged("E baseline force, α ×0.25", cloud0, spacing, octree;
        force_model=SpacingEquilibriumForce(0.2), α=0.25 * 0.05 * Δ * m)
    return nothing
end

# Integrated check of the production configuration: new default force +
# kick_after (frees the standoff pair that pins sep) + cv_target (stop at
# direct-pipeline quality) + stall_after backstop.
function experiment_stallkick(cloud0, spacing, octree)
    println("\n############ STALL+KICK: default force, kick_after=10, cv_target=0.07, stall_after=50 ############")
    for (label, cloud) in [("raw-PASS start", cloud0),
                           ("perturbed 0.3h start", (Random.seed!(SEED + 1); perturb(cloud0, spacing, 0.3)))]
        header("$label — single call, max_iters=300")
        row(0, cloud, NaN)
        conv = Float64[]
        t = @elapsed result = quietly() do
            repel(cloud, spacing, octree;
                max_iters=300, tol=1.0e-4, kick_after=10, cv_target=0.07,
                stall_after=50, convergence=conv)
        end
        g = row(length(conv), result, last(conv))
        gate = (g.sep > 0.1 && g.cv < 0.15) ? "gate-PASS" : "below gate"
        @printf("         -> stopped at %d iters (%.1f s): %s\n", length(conv), t, gate)
    end
    return nothing
end

function experiment_recover(cloud0, spacing, octree)
    println("\n############ RECOVER: perturb all points by 0.3·h ############")
    Random.seed!(SEED + 1)
    cloudp = perturb(cloud0, spacing, 0.3)
    configs = [
        ("A baseline SpacingEquilibriumForce(0.2)", (; force_model=SpacingEquilibriumForce(0.2))),
        ("B ClippedSpacingForce(0.2, u0=1.0)", (; force_model=ClippedSpacingForce(0.2, 1.0))),
        ("C DampedAttractionForce(0.2, c=0.1)", (; force_model=DampedAttractionForce(0.2, 0.1))),
    ]
    for (label, kw) in configs
        header("$label (perturbed start)")
        row(0, cloudp, NaN)
        cloud, cum = cloudp, 0
        for c in [5, 5, 5, 5, 5, 25, 25, 25]
            conv = Float64[]
            cloud = quietly() do
                repel(cloud, spacing, octree;
                    max_iters=c, tol=1.0e-4, convergence=conv, kw...)
            end
            cum += c
            g = row(cum, cloud, last(conv))
            gate = (g.sep > 0.1 && g.cv < 0.15) ? "gate-PASS" : "below gate"
            seeded = (g.sep >= 0.70 && g.cv <= 0.075) ? " + seeded-quality" : ""
            println("         -> $gate$seeded")
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------

function main()
    which = isempty(ARGS) ? ["preserve"] : ARGS
    println("Threads: $(Threads.nthreads())")
    t = @elapsed begin
        cloud0, spacing, octree = build_seeded()
    end
    @printf("Seeded cloud: %d boundary + %d volume points (%.1f s)\n",
        length(points(WhatsThePoint.boundary(cloud0))), length(WhatsThePoint.volume(cloud0).points), t)
    "preserve" in which && experiment_preserve(cloud0, spacing, octree)
    "recover" in which && experiment_recover(cloud0, spacing, octree)
    "stallkick" in which && experiment_stallkick(cloud0, spacing, octree)
    return nothing
end

main()
