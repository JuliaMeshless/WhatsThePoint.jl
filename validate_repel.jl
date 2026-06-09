# validate_repel.jl — demonstrate the live-neighbor-graph repel fix.
# Octree :random fill is Poisson (close pairs + voids). Repel should relax it to
# a blue-noise-like cloud: separation ↑, mesh_ratio ↓. We compare the FROZEN
# tree (rebuild_every huge ≈ old behavior) against the LIVE tree (rebuild_every=1).
using Pkg
Pkg.activate(@__DIR__)
using WhatsThePoint
using GeoIO, Meshes
using LinearAlgebra: norm
using Printf

path = joinpath(@__DIR__, "test", "data", "box.stl")
mesh_raw = GeoIO.load(path).geometry
# box.stl loads as Float32; Octree discretization emits Float64 volume points, so
# promote the mesh to Float64 to avoid a (pre-existing) PointCloud type mismatch.
verts64 = [Meshes.Point((1.0 .* Meshes.to(v))...) for v in Meshes.vertices(mesh_raw)]
mesh = Meshes.SimpleMesh(verts64, Meshes.topology(mesh_raw))
boundary = PointBoundary(mesh)
diag = norm(Meshes.boundingbox(mesh).max - Meshes.boundingbox(mesh).min)
spacing = ConstantSpacing(diag / 18)

octree = TriangleOctree(mesh; classify_leaves = true)
alg = Octree(mesh; spacing, alpha = 1.0)

println("\n############ RAW OCTREE FILL (:random, Poisson) ############")
cloud0 = discretize(boundary, spacing; alg, max_points = 4000)
m0 = metrics(cloud0)

println("\n############ repel — FROZEN tree (rebuild_every=10_000 ≈ OLD) ############")
c_frozen = repel(cloud0, spacing, octree; max_iters = 400, tol = 1.0e-4, rebuild_every = 10_000)
mf = metrics(c_frozen)

println("\n############ repel — LIVE tree (rebuild_every=1, NEW default) ############")
c_live = repel(cloud0, spacing, octree; max_iters = 400, tol = 1.0e-4, rebuild_every = 1)
ml = metrics(c_live)

println("\n==================== SUMMARY ====================")
@printf("%-22s  %12s  %12s\n", "stage", "separation", "mesh_ratio")
@printf("%-22s  %12.4e  %12.2f\n", "raw octree fill", m0.separation, m0.mesh_ratio)
@printf("%-22s  %12.4e  %12.2f\n", "repel frozen (old)", mf.separation, mf.mesh_ratio)
@printf("%-22s  %12.4e  %12.2f\n", "repel live (new)", ml.separation, ml.mesh_ratio)
@printf("\nlive vs frozen: separation %.2fx better, mesh_ratio %.2fx lower\n",
    ml.separation / mf.separation, mf.mesh_ratio / ml.mesh_ratio)
