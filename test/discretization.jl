# Discretization tests

@testitem "ConstantSpacing" setup = [CommonImports] begin
    s = ConstantSpacing(1.0m)
    @test s(Point(0.0, 0.0, 0.0)) == 1.0m
    @test s() == 1.0m  # Test no-argument callable
end

@testitem "BoundaryLayerSpacing monotonicity" setup = [CommonImports] begin
    points = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)]
    spacing = BoundaryLayerSpacing(points; at_wall = 1.0m, bulk = 10.0m, layer_thickness = 5.0m)

    # Monotonic: farther from boundary = larger spacing
    near = spacing(Point(0.1, 0.0, 0.0))
    far = spacing(Point(100.0, 0.0, 0.0))
    @test near < far
    @test near >= 1.0m
    @test far <= 10.0m
end

@testitem "SlakKosec with octree" setup = [TestData, CommonImports] begin
    bnd = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(bnd)

    cloud = discretize(bnd, spacing; alg = SlakKosec(octree), max_points = 50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 50

    # All points should be inside
    for p in points(volume(cloud))
        @test isinside(p, octree) == true
    end
end

@testitem "VanDerSandeFornberg" setup = [TestData, CommonImports] begin
    @test_skip "Temporarily skipped - slow without octree acceleration"

    bnd = PointBoundary(TestData.BOX_PATH)
    spacing = _relative_spacing(bnd)

    cloud = discretize(bnd, spacing; alg = VanDerSandeFornberg(), max_points = 50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 50
end

@testitem "FornbergFlyer (2D)" setup = [CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.5m)

    cloud = discretize(boundary, spacing; alg = FornbergFlyer(), max_points = 50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 50
end

@testitem "max_points cap" setup = [TestData, CommonImports] begin
    bnd = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(bnd)

    cloud = discretize(bnd, spacing; alg = SlakKosec(octree), max_points = 20)
    @test length(volume(cloud)) == 20  # Should hit cap
end

@testitem "discretize accepts bare Unitful.Length" setup = [TestData, CommonImports] begin
    # Test convenience overload that wraps bare Length in ConstantSpacing
    bnd = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)

    # Should accept 3.0m instead of ConstantSpacing(3.0m)
    cloud = discretize(bnd, 3.0m; alg = SlakKosec(octree), max_points = 30)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 30
end

@testitem "discretize works with PointCloud input" setup = [TestData, CommonImports] begin
    # Test discretize(cloud::PointCloud, ...) overload
    bnd = PointBoundary(TestData.BOX_PATH)
    cloud = PointCloud(bnd)  # Empty volume

    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(bnd)

    # Discretize the empty cloud
    filled_cloud = discretize(cloud, spacing; alg = SlakKosec(octree), max_points = 30)
    @test filled_cloud isa PointCloud
    @test length(volume(filled_cloud)) > 0
    @test length(volume(filled_cloud)) <= 30

    # Convenience overload with bare Length
    filled_cloud2 = discretize(cloud, 3.0m; alg = SlakKosec(octree), max_points = 25)
    @test filled_cloud2 isa PointCloud
    @test length(volume(filled_cloud2)) > 0
end

@testitem "SlakKosec with BoundaryLayerSpacing" setup = [TestData, CommonImports] begin
    # Test SlakKosec with variable spacing (exercises calculate_ninit for VariableSpacing)
    bnd = PointBoundary(TestData.BOX_PATH)
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = 2.0m
    )

    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    cloud = discretize(bnd, spacing; alg = SlakKosec(octree), max_points = 30)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 30
end

@testitem "SlakKosec with variable spacing exercises calculate_ninit" setup = [TestData, CommonImports] begin
    # Test SlakKosec with variable spacing to exercise calculate_ninit(::VariableSpacing)
    bnd = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.8m,
        bulk = 4.0m,
        layer_thickness = 2.0m
    )

    # This exercises calculate_ninit(cloud::PointCloud{𝔼{3}}, s::VariableSpacing)
    cloud = discretize(bnd, spacing; alg = SlakKosec(octree), max_points = 30)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 30
end

@testitem "LogLike spacing" setup = [CommonImports] begin
    # Test LogLike spacing type with small boundary (LogLike is O(n) per query)
    points = [Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0)]
    cloud = PointCloud(PointBoundary(points))

    # Create LogLike spacing (growth_rate = 1.5, NOT 2.0 which gives constant spacing)
    spacing = WhatsThePoint.LogLike(cloud, 0.5m, 1.5)

    # Test spacing evaluation
    test_point = Point(5.0, 0.0, 0.0)
    h = spacing(test_point)
    @test h isa Unitful.Length
    @test h > 0m

    # Verify spacing increases with distance from boundary
    near_point = Point(0.1, 0.0, 0.0)  # Very close to first boundary point
    far_point = Point(5.0, 0.0, 0.0)   # Midpoint between boundary points
    @test spacing(near_point) < spacing(far_point)
end

@testitem "BoundaryLayerSpacing validation" setup = [TestData, CommonImports] begin
    # Test that BoundaryLayerSpacing validates layer_thickness > 0
    bnd = PointBoundary(TestData.BOX_PATH)

    @test_throws ArgumentError BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = -1.0m  # Invalid: negative thickness
    )

    @test_throws ArgumentError BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = 0.0m  # Invalid: zero thickness
    )
end
