# validate_repel.jl — repel improve-or-preserve check on box.stl (geometry
# ladder rung 1: sharp edges / flat faces).
#
# The direct pipeline (Poisson-disk surface sampling + :bridson volume) seeds
# the cloud; repel under the ClippedSpacingForce default must then improve, or
# at worst preserve, the spacing statistics (NODEGEN_FINDINGS.md, Session 3).
# Two runs:
#   1. fixed 300-iteration budget — the preserve check;
#   2. production stopping (kick_after=10, cv_target=0.07, stall_after=50) —
#      should end in tens of iterations, not the full budget.
#
# History: this script used to compare frozen vs live neighbor trees on a
# face-center boundary. That comparison is retired — the live tree has long
# been the default, and the face-center scenario ends in projection-parked
# coincident pairs (separation 0, mesh_ratio Inf for *both* arms), which says
# nothing about tree mode. The mesh is consumed as loaded (Float32):
# discretize now promotes Octree-algorithm boundaries itself.
#
# Run single-threaded for comparable numbers:
#   julia --project=. -t 1 validate_repel.jl
using Pkg
Pkg.activate(@__DIR__)
using WhatsThePoint
using GeoIO, Meshes
using NearestNeighbors: KDTree, knn
using LinearAlgebra: norm
using Unitful: ustrip
using Printf, Random

path = joinpath(@__DIR__, "test", "data", "box.stl")
mesh = GeoIO.load(path).geometry
diag = norm(Meshes.boundingbox(mesh).max - Meshes.boundingbox(mesh).min)
spacing = ConstantSpacing(diag / 18)
const Δ = ustrip(diag / 18)
@printf("box.stl: diag %.2f, Δ = %.3f, threads = %d\n", ustrip(diag), Δ, Threads.nthreads())

Random.seed!(20260611)
octree = TriangleOctree(mesh; classify_leaves = true)
bnd = PointBoundary(mesh, spacing)            # Poisson-disk surface sampling
alg = Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
cloud0 = discretize(bnd, spacing; alg, max_points = 10_000)
@printf("seeded: %d boundary + %d volume points\n",
    length(points(WhatsThePoint.boundary(cloud0))), length(WhatsThePoint.volume(cloud0)))

_raw(p) = Float64.(ustrip.(Meshes.to(p)))
function quality(cloud)
    coords = _raw.(points(cloud))
    tree = KDTree(coords)
    _, dists = knn(tree, coords, 2, true)
    dnn = [d[2] for d in dists]
    f = spacing_fidelity_metrics(cloud, spacing)
    return (; sep = minimum(dnn) / Δ, cv = f.cv, mean = f.mean_dnn_h, coord = f.coordination)
end

q0 = quality(cloud0)
@printf("%-26s  sep/Δ=%.3f  CV=%.3f  mean=%.3f  coord=%.1f\n", "raw direct pipeline", q0...)

println("\n-- repel, fixed 300 iterations (preserve check) --")
conv1 = Float64[]
t1 = @elapsed c1 = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, kick_after = 10, convergence = conv1)
q1 = quality(c1)
@printf("%-26s  sep/Δ=%.3f  CV=%.3f  mean=%.3f  coord=%.1f   (%d iters, %.1f s)\n",
    "repel 300", q1..., length(conv1), t1)

println("\n-- repel, production stopping (kick=10, cv_target=0.07, stall_after=50) --")
conv2 = Float64[]
t2 = @elapsed c2 = repel(cloud0, spacing, octree;
    max_iters = 300, tol = 1.0e-4, kick_after = 10, cv_target = 0.07, stall_after = 50,
    convergence = conv2)
q2 = quality(c2)
@printf("%-26s  sep/Δ=%.3f  CV=%.3f  mean=%.3f  coord=%.1f   (%d iters, %.1f s)\n",
    "repel auto-stop", q2..., length(conv2), t2)

println("\n==================== VERDICT ====================")
preserve = q1.cv <= q0.cv * 1.05 && q1.sep > 0.1
# cv_target promises "at least target quality", not "better than raw": a
# mid-run stop legitimately lands at cv ≈ 0.07 even when raw was better.
early = length(conv2) < 300 && q2.cv <= max(q0.cv * 1.05, 0.07) && q2.sep > 0.1
@printf("preserve (300 iters): CV %.3f -> %.3f, sep/Δ %.3f -> %.3f  => %s\n",
    q0.cv, q1.cv, q0.sep, q1.sep, preserve ? "PASS" : "FAIL")
@printf("auto-stop: %d iters, CV %.3f, sep/Δ %.3f  => %s\n",
    length(conv2), q2.cv, q2.sep, early ? "PASS" : "FAIL")
println(preserve && early ? "PASS — repel improves-or-preserves on box.stl." :
        "FAIL — see NODEGEN_FINDINGS.md open defects.")
