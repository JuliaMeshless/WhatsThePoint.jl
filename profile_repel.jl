# profile_repel.jl — Session-1 step 2: where do the repel allocations go?
#
# Measures, at ~45k points (cavity Δ=0.044, matching the 46.8k-pt measurement
# in NODEGEN_FINDINGS.md):
#   (a) full 10-iteration boundary-projected repel  — total time + GB allocated
#   (b) kd-tree rebuild alone (KNearestSearch)      — the per-iteration build cost
#   (c) one full sweep of searchdists queries       — the per-query allocation cost
#
# Usage:  jlrun profile_repel.jl [Δ]    (default Δ=0.044)
using Pkg
Pkg.activate(@__DIR__)

using WhatsThePoint
using WhatsThePoint: searchdists
using GeoIO, Meshes
using Unitful: m, ustrip
using LinearAlgebra: norm
using Printf

const STL_PATH = joinpath(@__DIR__, "test", "data", "cavity.stl")
isfile(STL_PATH) || error("run validate_cavity.jl first to generate $STL_PATH")

Δ = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.044

mesh_raw = GeoIO.load(STL_PATH).geometry
mesh = Meshes.SimpleMesh(
    [Meshes.Point((1.0 .* Meshes.to(v))...) for v in Meshes.vertices(mesh_raw)],
    Meshes.topology(mesh_raw),
)
boundary_ = PointBoundary(mesh)
spacing = ConstantSpacing(Δ * m)
octree = TriangleOctree(mesh; classify_leaves = true)
alg = Octree(mesh; spacing, alpha = 1.0, placement = :random)

max_pts = round(Int, (4 / 3) * π * (1.0^3 - 0.547^3) / Δ^3)
cloud0 = discretize(boundary_, spacing; alg, max_points = max_pts)
N = length(points(cloud0))
@printf("cloud: %d points (Δ=%.4f)\n", N, Δ)

k = 21

# --- warmup (compile both paths) ---
repel(cloud0, spacing, octree; max_iters = 1, tol = 0.0)
snap = copy(points(cloud0))
method = KNearestSearch(snap, k)
searchdists(snap[1], method)

# --- (a) full repel, 10 iterations ---
GC.gc()
stats_a = @timed repel(cloud0, spacing, octree; max_iters = 10, tol = 0.0)
@printf("\n(a) repel 10 iters:        %6.2f s   %7.3f GB   (%.1f%% gc)\n",
    stats_a.time, stats_a.bytes / 1e9, 100 * stats_a.gctime / stats_a.time)

# --- (b) kd-tree rebuild ×10 ---
GC.gc()
stats_b = @timed for _ in 1:10
    KNearestSearch(snap, k)
end
@printf("(b) kd rebuild ×10:        %6.2f s   %7.3f GB   (%.1f%% of repel time, %.1f%% of alloc)\n",
    stats_b.time, stats_b.bytes / 1e9,
    100 * stats_b.time / stats_a.time, 100 * stats_b.bytes / stats_a.bytes)

# --- (c) one full searchdists sweep ×10 (serial; time also reported per-sweep) ---
GC.gc()
stats_c = @timed for _ in 1:10
    for pt in snap
        searchdists(pt, method)
    end
end
@printf("(c) searchdists sweep ×10: %6.2f s   %7.3f GB   (%.1f%% of repel alloc)\n",
    stats_c.time, stats_c.bytes / 1e9, 100 * stats_c.bytes / stats_a.bytes)

@printf("\nresidual (force calc, projection, snapshots): %.3f GB\n",
    (stats_a.bytes - stats_b.bytes - stats_c.bytes) / 1e9)
