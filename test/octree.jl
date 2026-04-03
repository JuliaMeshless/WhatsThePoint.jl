# Tests for Octree discretization algorithm

@testitem "Octree with different spacing types" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    # Test ConstantSpacing
    alg = Octree(mesh)
    cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 100)
    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100

    # Test BoundaryLayerSpacing
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = 2.0m
    )
    alg = Octree(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)
    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "Octree points are inside" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(123)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(1.0m)

    alg = Octree(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)

    # All points should be inside via octree check
    octree = alg.triangle_octree
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end
end

@testitem "Octree with placement strategies" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(456)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    for placement in (:random, :jittered, :lattice)
        alg = Octree(mesh; placement)
        cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 50)

        @test cloud isa PointCloud
        @test length(WhatsThePoint.volume(cloud)) > 0
        @test length(WhatsThePoint.volume(cloud)) <= 50
    end
end

@testitem "Octree errors on unclassified octree" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; classify_leaves = false)

    @test_throws Exception discretize(
        bnd, ConstantSpacing(1m);
        alg = Octree(octree), max_points = 50
    )
end

@testitem "Octree invalid placement throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError Octree(octree; placement = :invalid)
end

@testitem "Octree invalid oversampling throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError Octree(octree; boundary_oversampling = -1.0)
end

@testitem "Octree octree subdivision behavior" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(789)

    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)
    node_min_ratio = 1.0e-6

    # Test 1: Fine spacing triggers subdivision
    fine_spacing = ConstantSpacing(0.2m)
    node_tree = WhatsThePoint.build_node_octree(tri_octree, fine_spacing, 2.0, node_min_ratio)
    num_leaves = length(WhatsThePoint.all_leaves(node_tree))
    @test num_leaves >= 8  # At least one level of subdivision

    # Test 2: Alpha parameter affects subdivision aggressiveness
    spacing = ConstantSpacing(0.3m)
    node_tree_aggressive = WhatsThePoint.build_node_octree(tri_octree, spacing, 1.0, node_min_ratio)
    node_tree_conservative = WhatsThePoint.build_node_octree(tri_octree, spacing, 5.0, node_min_ratio)
    @test length(WhatsThePoint.all_leaves(node_tree_aggressive)) > length(WhatsThePoint.all_leaves(node_tree_conservative))

    # Test 3: Finer spacing produces more leaves than coarse spacing
    spacing_coarse = ConstantSpacing(0.8m)
    node_tree_fine = WhatsThePoint.build_node_octree(tri_octree, fine_spacing, 2.0, node_min_ratio)
    node_tree_coarse = WhatsThePoint.build_node_octree(tri_octree, spacing_coarse, 2.0, node_min_ratio)
    @test length(WhatsThePoint.all_leaves(node_tree_fine)) > length(WhatsThePoint.all_leaves(node_tree_coarse))
    @test length(WhatsThePoint.all_leaves(node_tree_coarse)) >= 1
end

@testitem "Octree node octree classification works" setup = [CommonImports, OctreeTestData] begin
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


@testitem "Octree constructors" setup = [TestData, CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(606)

    # Test with BoundaryLayerSpacing (exercises _extract_min_spacing)
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.3m,
        bulk = 3.0m,
        layer_thickness = 1.5m
    )
    alg = Octree(mesh; spacing, alpha = 1.5)
    cloud = discretize(bnd, spacing; alg, max_points = 50)
    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test alg.node_min_ratio < 1.0

    # Test string filepath constructor
    alg2 = Octree(TestData.BOX_PATH)
    @test alg2 isa Octree
    @test alg2.triangle_octree isa WhatsThePoint.TriangleOctree
end
