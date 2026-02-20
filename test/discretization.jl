@testitem "ConstantSpacing" setup = [TestData, CommonImports] begin
    s = ConstantSpacing(1.0m)
    @test s.Δx == 1.0m

    @test s() == 1.0m

    p = Point(0.0, 0.0, 0.0)
    @test s(p) == 1.0m

    s2 = ConstantSpacing(0.5m)
    @test s2() == 0.5m
    @test s2(p) == 0.5m
end

@testitem "LogLike Spacing" setup = [TestData, CommonImports] begin
    points = Point.(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
    )
    cloud = PointCloud(PointBoundary(points))

    base_size = 0.1m
    growth_rate = 1.5
    s = LogLike(cloud, base_size, growth_rate)

    @test s.base_size == base_size
    @test s.growth_rate == growth_rate

    # Test specific value at center of unit cube
    # Formula: base_size * x / (base_size * inv_growth + x)
    # where inv_growth = 1 - (growth_rate - 1) = 0.5
    # and x = distance to nearest boundary = sqrt(0.75)m for center point
    test_point = Point(0.5, 0.5, 0.5)
    spacing_value = s(test_point)
    x_center = sqrt(0.75)m
    inv_growth = 1 - (growth_rate - 1)
    expected_center = base_size * x_center / (base_size * inv_growth + x_center)
    @test spacing_value ≈ expected_center
    @test spacing_value isa typeof(base_size)

    # Test close point - distance to (0,0,0) is 0.01m
    close_point = Point(0.01, 0.0, 0.0)
    x_close = 0.01m
    expected_close = base_size * x_close / (base_size * inv_growth + x_close)
    @test s(close_point) ≈ expected_close

    # Test far point - distance to nearest corner (1,1,1) is sqrt(48)m
    far_point = Point(5.0, 5.0, 5.0)
    x_far = sqrt((5 - 1)^2 + (5 - 1)^2 + (5 - 1)^2)m
    expected_far = base_size * x_far / (base_size * inv_growth + x_far)
    @test s(far_point) ≈ expected_far

    # Verify ordering still holds
    @test s(close_point) < s(far_point)
end

@testitem "calculate_ninit 3D" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    cloud = PointCloud(PointBoundary(points))
    spacing = ConstantSpacing(0.1m)

    # Formula: (ceil(Int, extent[1] * 10 / Δx), ceil(Int, extent[2] * 10 / Δx))
    # For unit tetrahedron: extent = (1m, 1m, 1m), Δx = 0.1m
    # Expected: (ceil(1m * 10 / 0.1m), ceil(1m * 10 / 0.1m)) = (100, 100)
    ninit = WhatsThePoint.calculate_ninit(cloud, spacing)
    @test ninit isa Tuple{Int, Int}
    @test ninit == (100, 100)
end

@testitem "calculate_ninit 2D" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    cloud = PointCloud(PointBoundary(points))
    spacing = ConstantSpacing(0.1m)

    # Formula: ceil(Int, extent[1] * 10 / Δx)
    # For unit square: extent[1] = 1m, Δx = 0.1m
    # Expected: ceil(1m * 10 / 0.1m) = 100
    ninit = WhatsThePoint.calculate_ninit(cloud, spacing)
    @test ninit isa Int
    @test ninit == 100
end

@testitem "discretize with SlakKosec (3D)" setup = [TestData, CommonImports] begin
    using WhatsThePoint: boundary
    bnd = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(0.5m)

    cloud = discretize(bnd, spacing; alg = SlakKosec(), max_points = 50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 50
    @test length(boundary(cloud)) == length(bnd)

    cloud2 = PointCloud(bnd)
    @test length(volume(cloud2)) == 0
    cloud2 = discretize(cloud2, spacing; alg = SlakKosec(), max_points = 50)
    @test length(volume(cloud2)) <= 50
    @test length(volume(cloud2)) > 0
end

@testitem "discretize with VanDerSandeFornberg (3D)" setup = [TestData, CommonImports] begin
    bnd = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(5.0m)

    cloud = discretize(bnd, spacing; alg = VanDerSandeFornberg(), max_points = 100)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 100

    cloud2 = PointCloud(bnd)
    cloud2 = discretize(cloud2, spacing; alg = VanDerSandeFornberg(), max_points = 100)
    @test length(volume(cloud2)) <= 100
    @test length(volume(cloud2)) > 0
end

@testitem "discretize with FornbergFlyer (2D)" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.2m)

    cloud = @test_logs (
        :warn,
        "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it.",
    ) discretize(boundary, spacing; alg = FornbergFlyer(), max_points = 100)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 100

    cloud2 = PointCloud(boundary)
    cloud2 = discretize(cloud2, spacing; alg = FornbergFlyer(), max_points = 100)
    @test length(volume(cloud2)) <= 100
end

@testitem "discretize default algorithms" setup = [TestData, CommonImports] begin
    @testset "3D default (SlakKosec)" begin
        points =
            Point.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        boundary = PointBoundary(points)
        spacing = ConstantSpacing(0.5m)

        cloud = discretize(boundary, spacing; max_points = 50)
        @test cloud isa PointCloud
        @test length(volume(cloud)) <= 50
    end

    @testset "2D default (FornbergFlyer)" begin
        points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        boundary = PointBoundary(points)
        spacing = ConstantSpacing(0.2m)

        cloud = @test_logs (
            :warn,
            "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it.",
        ) discretize(boundary, spacing; max_points = 100)
        @test cloud isa PointCloud
        @test length(volume(cloud)) <= 100
    end
end

@testitem "Algorithm constructors" setup = [TestData, CommonImports] begin
    alg1 = SlakKosec()
    @test alg1.n == 10

    alg2 = SlakKosec(15)
    @test alg2.n == 15

    alg3 = VanDerSandeFornberg()
    @test alg3 isa VanDerSandeFornberg

    alg4 = FornbergFlyer()
    @test alg4 isa FornbergFlyer
end

@testitem "max_points limit" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.1m)

    max_pts = 20
    cloud = discretize(boundary, spacing; alg = SlakKosec(), max_points = max_pts)
    @test length(volume(cloud)) <= max_pts
end

@testitem "discretize with bare Unitful.Length" setup = [TestData, CommonImports] begin
    using WhatsThePoint: boundary
    bnd = PointBoundary(TestData.BOX_PATH)

    # Test PointBoundary overload
    cloud1 = discretize(bnd, 0.5m; max_points = 50)
    @test cloud1 isa PointCloud
    @test length(volume(cloud1)) <= 50
    @test length(boundary(cloud1)) == length(bnd)

    # Test PointCloud overload
    cloud2 = PointCloud(bnd)
    cloud2 = discretize(cloud2, 0.5m; max_points = 50)
    @test cloud2 isa PointCloud
    @test length(volume(cloud2)) <= 50
end

@testitem "_check_inside dispatch" setup = [TestData, CommonImports] begin
    using WhatsThePoint: _check_inside

    bnd = PointBoundary(TestData.BOX_PATH)
    cloud = PointCloud(bnd)
    octree = TriangleOctree(TestData.BOX_PATH; h_min = 0.5, classify_leaves = true)

    # Centroid of the box boundary (guaranteed interior)
    interior = centroid(bnd)
    exterior = Point(1000.0m, 1000.0m, 1000.0m)

    # Octree dispatch path
    @test _check_inside(interior, cloud, octree) == true
    @test _check_inside(exterior, cloud, octree) == false

    # Green's function dispatch path (nothing octree)
    @test _check_inside(interior, cloud, nothing) == true
    @test _check_inside(exterior, cloud, nothing) == false

    # Both paths agree
    @test _check_inside(interior, cloud, octree) == _check_inside(interior, cloud, nothing)
    @test _check_inside(exterior, cloud, octree) == _check_inside(exterior, cloud, nothing)
end

@testitem "discretize with SlakKosec(octree) (3D)" setup = [TestData, CommonImports] begin
    using WhatsThePoint: boundary

    bnd = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; h_min = 0.5, classify_leaves = true)
    spacing = ConstantSpacing(0.5m)

    cloud = discretize(bnd, spacing; alg = SlakKosec(octree), max_points = 50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) > 0
    @test length(volume(cloud)) <= 50
    @test length(boundary(cloud)) == length(bnd)

    # All volume points should be inside the domain
    vol_pts = points(volume(cloud))
    for p in vol_pts
        @test isinside(p, octree) == true
    end
end

