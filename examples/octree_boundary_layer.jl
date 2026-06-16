using WhatsThePoint
using CairoMakie
using GeoIO
using Meshes
using Unitful: m, ustrip

# Load geometry
println("Loading mesh...")
@time mesh = GeoIO.load("bunny.stl").geometry
println("Mesh has $(nelements(mesh)) triangles")

# Boundary layer spacing tuned for the bunny's ~129k m³ interior volume.
# At h=0.8m we expect ~250k volume points; layer_thickness=15m keeps most
# of the volume in the fine boundary layer (the bunny is only ~43m radius).
spacing = BoundaryLayerSpacing(
    WhatsThePoint.points(PointBoundary(mesh));
    at_wall = 0.8m,
    bulk = 2.0m,
    layer_thickness = 15.0m,
)

# Poisson-disk boundary sampling (replaces face-center import)
println("\nCreating Poisson-disk boundary...")
@time boundary = PointBoundary(mesh, spacing)
println("Boundary has $(length(boundary)) points")

# Octree with Bridson global Poisson-disk volume placement
println("\nBuilding Octree...")
@time alg = Octree(mesh; spacing, placement = :bridson, alpha = 1.0, bridson_factor = 0.75)

println("\nDiscretizing volume...")
@time cloud = discretize(boundary, spacing; alg, max_points = 600_000)
vol_pts = WhatsThePoint.volume(cloud)
println("Generated $(length(vol_pts)) volume points")
println("Total cloud: $(length(cloud)) points (boundary + volume)")

# Metrics
println("\n--- Quality metrics ---")
cloud_metrics = metrics(cloud; k = 20)
sf = spacing_fidelity_metrics(cloud, spacing; k = 30)
h_min = ustrip(minimum(spacing.(WhatsThePoint.points(cloud))))
println("spacing CV:      $(round(sf.cv; digits=4))")
println("sep/Δ:           $(round(ustrip(cloud_metrics.separation) / h_min; digits=3))")
println("coordination:    $(round(sf.coordination; digits=1))")
println("fill max/Δ:      $(round(ustrip(cloud_metrics.fill) / h_min; digits=2))")

# Save visualization
println("\nSaving plot...")
fig = visualize(cloud; markersize = 0.15)
save("bunny_boundary_layer.png", fig)
println("Saved to bunny_boundary_layer.png")
