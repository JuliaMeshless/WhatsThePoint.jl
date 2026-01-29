using WhatsThePoint
using Unitful: m
using GLMakie

# 1. Load boundary from STL (extracts face centers as boundary points)
bnd = PointBoundary(joinpath(@__DIR__, "..", "bunny.stl"))
println("Loaded boundary with $(length(bnd)) points")

# 2. Create point cloud with boundary (volume initially empty)
cloud = PointCloud(bnd)

# 3. Build octree for fast isinside queries (~1000x speedup)
octree = TriangleOctree(
    joinpath(@__DIR__, "..", "bunny.stl");
    h_min = 0.5,
    classify_leaves = true,
)

# 4. Define spacing (bunny is ~86m across, 3m gives good density)
spacing = ConstantSpacing(3.0m)

# 5. Discretize volume with octree-accelerated SlakKosec
alg = SlakKosec(octree)
@time cloud2 = discretize(cloud, spacing; alg = alg, max_points = 200_000)

# 6. Print results
println("Boundary points: ", length(boundary(cloud2)))
println("Volume points: ", length(WhatsThePoint.volume(cloud2)))
println("Total points: ", length(cloud2))

visualize(cloud2; markersize = 0.3)
