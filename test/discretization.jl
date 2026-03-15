# Discretization tests

@testitem "ConstantSpacing" setup = [CommonImports] begin
    s = ConstantSpacing(1.0m)
    @test s(Point(0.0, 0.0, 0.0)) == 1.0m
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
