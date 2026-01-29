using WhatsThePoint
using Unitful: m
using GLMakie

# Load boundary from STL (stores mesh for octree reuse)
boundary = PointBoundary(joinpath(@__DIR__, "..", "bunny.stl"))
println("Loaded boundary with $(length(boundary)) points")

# Discretize volume with octree acceleration (bunny is ~86m across, 3m gives good density)
@time cloud = discretize(boundary, 3.0m; use_octree = true, max_points = 100_000)

# Print results
println("Boundary: $(length(boundary)) | Volume: $(length(WhatsThePoint.volume(cloud))) | Total: $(length(cloud))")

visualize(cloud; markersize = 0.3)
