using WhatsThePoint
using Unitful: m
using GLMakie
using GeoIO

mesh = GeoIO.load("bunny.stl").geometry

# Step 2: (Optional) Preprocess mesh with Meshes.jl
# mesh = refine(mesh)
# mesh = smooth(mesh)
# etc.

# create point boundary from mesh without storing the mesh
boundary = PointBoundary(mesh)
println("Loaded boundary with $(length(boundary)) points")

#explicit octree construction (later to be integrated into API)
octree = TriangleOctree(mesh; classify_leaves=true)
println("Constructed octree with $(length(octree)) nodes")

# Discretize volume with octree acceleration (bunny is ~86m across, 3m gives good density)
node_gen_alg = SlakKosec(octree)
cloud = discretize(boundary, 3.0m, max_points=50_000; alg=node_gen_alg)

# Print results
println("Boundary: $(length(boundary)) | Volume: $(length(WhatsThePoint.volume(cloud))) | Total: $(length(cloud))")

visualize(cloud; markersize=0.3)
