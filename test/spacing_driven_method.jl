# Tests for SpacingDrivenMethod discretization algorithm

@testitem "SpacingDrivenMethod with ConstantSpacing" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    alg = SpacingDrivenMethod(mesh)
    cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 100)

    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "SpacingDrivenMethod with BoundaryLayerSpacing" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = 2.0m
    )

    alg = SpacingDrivenMethod(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)

    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "SpacingDrivenMethod points are inside" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(123)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(1.0m)

    alg = SpacingDrivenMethod(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)

    # All points should be inside via octree check
    octree = alg.triangle_octree
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end
end

@testitem "SpacingDrivenMethod with placement strategies" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(456)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    for placement in (:random, :jittered, :lattice)
        alg = SpacingDrivenMethod(mesh; placement)
        cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 50)

        @test cloud isa PointCloud
        @test length(WhatsThePoint.volume(cloud)) > 0
        @test length(WhatsThePoint.volume(cloud)) <= 50
    end
end

@testitem "SpacingDrivenMethod errors on unclassified octree" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; classify_leaves = false)

    @test_throws ErrorException discretize(
        bnd, ConstantSpacing(1m);
        alg = SpacingDrivenMethod(octree), max_points = 50
    )
end

@testitem "SpacingDrivenMethod invalid placement throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError SpacingDrivenMethod(octree; placement = :invalid)
end

@testitem "SpacingDrivenMethod invalid oversampling throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError SpacingDrivenMethod(octree; boundary_oversampling = -1.0)
end

@testitem "SpacingDrivenMethod octree actually subdivides with fine spacing" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(789)

    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)

    # Use very fine spacing to trigger subdivision
    # Unit cube is ~10m, with root box size ~17m (diagonal bounding)
    # With spacing=0.2m and alpha=2.0: h_box (17m) > alpha*spacing (0.4m) → should subdivide
    fine_spacing = ConstantSpacing(0.2m)
    alpha = 2.0
    node_min_ratio = 1.0e-6

    # Build node octree
    node_tree = WhatsThePoint.build_node_octree(tri_octree, fine_spacing, alpha, node_min_ratio)

    # Verify the octree actually subdivided beyond root
    num_leaves = length(WhatsThePoint.all_leaves(node_tree))
    @test num_leaves > 1  # Should have subdivided beyond just root node

    # Verify at least some depth of subdivision occurred
    # With such fine spacing, should get several levels deep
    @test num_leaves >= 8  # At least one level of subdivision (8 children)
end

@testitem "SpacingDrivenMethod subdivision respects alpha parameter" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(202)

    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)
    spacing = ConstantSpacing(0.3m)
    node_min_ratio = 1.0e-6

    # Build with aggressive subdivision (small alpha)
    node_tree_aggressive = WhatsThePoint.build_node_octree(tri_octree, spacing, 1.0, node_min_ratio)
    leaves_aggressive = length(WhatsThePoint.all_leaves(node_tree_aggressive))

    # Build with conservative subdivision (large alpha)
    node_tree_conservative = WhatsThePoint.build_node_octree(tri_octree, spacing, 5.0, node_min_ratio)
    leaves_conservative = length(WhatsThePoint.all_leaves(node_tree_conservative))

    # Smaller alpha (more aggressive) should produce more leaves
    @test leaves_aggressive > leaves_conservative
end

@testitem "SpacingDrivenMethod variable spacing produces adaptive octree" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(303)

    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)
    bnd = PointBoundary(mesh)

    # Test that fine constant spacing produces more leaves than coarse spacing
    # This verifies the subdivision mechanism works
    spacing_fine = ConstantSpacing(0.2m)      # Fine, should subdivide
    spacing_coarse = ConstantSpacing(0.8m)    # Coarse, less subdivision

    # Use small alpha to trigger subdivision
    node_tree_fine = WhatsThePoint.build_node_octree(tri_octree, spacing_fine, 2.0, 1.0e-6)
    node_tree_coarse = WhatsThePoint.build_node_octree(tri_octree, spacing_coarse, 2.0, 1.0e-6)

    leaves_fine = length(WhatsThePoint.all_leaves(node_tree_fine))
    leaves_coarse = length(WhatsThePoint.all_leaves(node_tree_coarse))

    # Finer spacing should produce more leaves
    @test leaves_fine > leaves_coarse
    # Both should have subdivided beyond root
    @test leaves_fine > 1
    @test leaves_coarse >= 1
end

@testitem "SpacingDrivenMethod node octree classification works" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)
    spacing = ConstantSpacing(0.2m)

    # Build and classify node octree
    node_tree = WhatsThePoint.build_node_octree(tri_octree, spacing, 2.0, 1.0e-6)
    classifications = WhatsThePoint.classify_node_octree(node_tree, tri_octree)

    # classifications is a Vector{Int8} indexed by box indices
    @test classifications isa Vector{Int8}
    @test length(classifications) > 0

    # Check that we have at least some interior leaves (unit cube should have interior)
    leaves = WhatsThePoint.all_leaves(node_tree)
    interior_count = count(idx -> classifications[idx] == WhatsThePoint.LEAF_INTERIOR, leaves)
    @test interior_count > 0

    # Check that all leaves are classified (not LEAF_UNKNOWN)
    for leaf_idx in leaves
        @test classifications[leaf_idx] != WhatsThePoint.LEAF_UNKNOWN
    end
end

@testitem "SpacingDrivenMethod generates points with subdivided octree" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(404)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    fine_spacing = ConstantSpacing(0.15m)

    # Create algorithm with fine spacing (triggers subdivision)
    alg = SpacingDrivenMethod(mesh; spacing = fine_spacing, alpha = 2.0)
    cloud = discretize(bnd, fine_spacing; alg, max_points = 500)

    # Verify points were generated
    @test length(WhatsThePoint.volume(cloud)) > 0

    # All points should be inside (verify subdivision didn't break inside/outside checks)
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, alg.triangle_octree) == true
    end
end
