using WhatsThePoint
using GLMakie
using GeoIO
using Meshes
using Unitful: m

# Load geometry
println("Loading mesh...")
@time mesh = GeoIO.load("bunny.stl").geometry
println("Mesh has $(nelements(mesh)) triangles")

println("\nCreating boundary...")
@time boundary = PointBoundary(mesh)
println("Boundary has $(length(WhatsThePoint.points(boundary))) points")

# Define boundary layer spacing: fine at walls, coarse in bulk
spacing = BoundaryLayerSpacing(
    WhatsThePoint.points(boundary);
    at_wall = 0.8m,
    bulk = 4.0m,
    layer_thickness = 3.0m
)

# AdaptiveOctree automatically adapts to variable spacing
# Pass spacing to constructor for automatic node_min_ratio computation
# Use smaller alpha for more aggressive subdivision with fine boundary layers
println("\nBuilding AdaptiveOctree (triangle octree construction)...")
@time alg = AdaptiveOctree(mesh; spacing, placement = :jittered, boundary_oversampling = 2.0, alpha = 1.0)
println("Node octree min_ratio: $(alg.node_min_ratio)")

println("\nDiscretizing volume (node octree + point generation)...")
@time cloud = discretize(boundary, spacing; alg, max_points = 600_000)

println("\nGenerated $(length(WhatsThePoint.volume(cloud))) volume points")

# Visualize
println("\nVisualizing...")
@time visualize(cloud; markersize = 0.15)
