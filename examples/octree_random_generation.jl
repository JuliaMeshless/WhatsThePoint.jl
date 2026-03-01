using WhatsThePoint
using Unitful: m, ustrip
using Printf
using GeoIO
using StaticArrays
using Meshes
using GLMakie

# This example demonstrates the OctreeRandom node generation algorithm
# which uses octree classification to efficiently generate random points
# inside a mesh.

println("="^70)
println("Octree Random Node Generation Example")
println("="^70)
println()

# Load the bunny mesh
stl_file = "bunny.stl"
if !isfile(stl_file)
    error("Bunny mesh not found at $stl_file. Please ensure bunny.stl is in the project root.")
end

println("Loading mesh from: $stl_file")
mesh = GeoIO.load(stl_file).geometry
boundary = PointBoundary(mesh)
println("✓ Loaded boundary with $(length(boundary)) surface points")
println()

# Build octree with classification
println("Building octree with classification...")
@time octree = TriangleOctree(mesh; classify_leaves=true)

# Print octree statistics
n_leaves = num_leaves(octree)
n_triangles = num_triangles(octree)
println("✓ Built octree:")
println("  - Total leaves: $n_leaves")
println("  - Triangles: $n_triangles")

# # Count leaf types
# leaf_indices = collect(all_leaves(octree.tree))
# leaf_classes = octree.leaf_classification[leaf_indices]
# n_interior = count(==(Int8(2)), leaf_classes)
# n_boundary = count(==(Int8(1)), leaf_classes)
# n_exterior = count(==(Int8(0)), leaf_classes)

# println("  - Interior leaves: $n_interior ($(round(100 * n_interior / n_leaves, digits = 1))%)")
# println("  - Boundary leaves: $n_boundary ($(round(100 * n_boundary / n_leaves, digits = 1))%)")
# println("  - Exterior leaves: $n_exterior ($(round(100 * n_exterior / n_leaves, digits = 1))%)")
# println()

# Generate points using OctreeRandom algorithm
target_points = 300_000
println("Generating $target_points random interior points...")

alg = OctreeRandom(octree, 2.0; verify_interior=true)
@time cloud = discretize(boundary, 1.0m; alg=alg, max_points=target_points)

n_volume = length(WhatsThePoint.volume(cloud))
println("✓ Generated $n_volume volume points")
println()

# Verify all points are inside
println("Verifying point classification...")
vol_points = points(WhatsThePoint.volume(cloud))
inside_count = count(vol_points) do pt
    # Convert Point to SVector{3,Float64} for octree query
    c = Meshes.to(pt)
    sv = SVector{3,Float64}(ustrip(c[1]), ustrip(c[2]), ustrip(c[3]))
    isinside(sv, octree)
end

println("✓ Inside verification: $inside_count / $n_volume points")
if inside_count == n_volume
    println("  ✓ All points correctly classified as inside!")
else
    println("  ⚠ Warning: $(n_volume - inside_count) points classified as outside")
end
println()

# Performance comparison note
println("="^70)
println("Performance Notes:")
println("="^70)
println("✓ Interior leaves: 100% acceptance (no filtering)")
println("✓ Boundary leaves: ~50-80% acceptance (filtered with isinside)")
println("✓ Exterior leaves: Skipped entirely")
println()
println("This is much faster than naive bounding box rejection sampling,")
println("which typically has only 5-20% acceptance rate for complex meshes.")
println()

# Summary
println("="^70)
println("Summary:")
println("="^70)
println("Generated $n_volume points from target of $target_points")
println(@sprintf("Efficiency: %.1f%%", 100 * n_volume / target_points))
println()
println("The OctreeRandom algorithm is ideal for:")
println("  • Quick initial discretization")
println("  • Uniform random distributions")
println("  • Maximum point count targets")
println("  • Testing and prototyping")
println()
println("For spacing-controlled discretization, use SlakKosec instead.")
println("="^70)


visualize(cloud; markersize=0.3)
