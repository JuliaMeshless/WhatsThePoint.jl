@testitem "ConstantSpacing" setup=[TestData, CommonImports] begin
    s = ConstantSpacing(1.0m)
    @test s.Δx == 1.0m

    @test s() == 1.0m

    p = Point(0.0, 0.0, 0.0)
    @test s(p) == 1.0m

    s2 = ConstantSpacing(0.5m)
    @test s2() == 0.5m
    @test s2(p) == 0.5m
end

@testitem "LogLike Spacing" setup=[TestData, CommonImports] begin
    points =
        Point.([
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ])
    cloud = PointCloud(PointBoundary(points))

    base_size = 0.1m
    growth_rate = 1.5
    s = LogLike(cloud, base_size, growth_rate)

    @test s.base_size == base_size
    @test s.growth_rate == growth_rate

    test_point = Point(0.5, 0.5, 0.5)
    spacing_value = s(test_point)
    @test spacing_value > 0m
    @test spacing_value isa typeof(base_size)

    close_point = Point(0.01, 0.0, 0.0)
    far_point = Point(5.0, 5.0, 5.0)
    @test s(close_point) < s(far_point)
end

@testitem "calculate_ninit 3D" setup=[TestData, CommonImports] begin
    points =
        Point.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    cloud = PointCloud(PointBoundary(points))
    spacing = ConstantSpacing(0.1m)

    ninit = WhatsThePoint.calculate_ninit(cloud, spacing)
    @test ninit isa Tuple{Int,Int}
    @test ninit[1] > 0
    @test ninit[2] > 0
end

@testitem "calculate_ninit 2D" setup=[TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    cloud = PointCloud(PointBoundary(points))
    spacing = ConstantSpacing(0.1m)

    ninit = WhatsThePoint.calculate_ninit(cloud, spacing)
    @test ninit isa Int
    @test ninit > 0
end

@testitem "discretize with SlakKosec (3D)" setup=[TestData, CommonImports] begin
    using WhatsThePoint: boundary
    bnd = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(0.5m)

    cloud = discretize(bnd, spacing; alg=SlakKosec(), max_points=50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 50
    @test length(boundary(cloud)) == length(bnd)

    cloud2 = PointCloud(bnd)
    @test length(volume(cloud2)) == 0
    discretize!(cloud2, spacing; alg=SlakKosec(), max_points=50)
    @test length(volume(cloud2)) <= 50
    @test length(volume(cloud2)) > 0
end

@testitem "discretize with VanDerSandeFornberg (3D)" setup=[TestData, CommonImports] begin
    bnd = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(5.0m)

    cloud = discretize(bnd, spacing; alg=VanDerSandeFornberg(), max_points=100)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 100

    cloud2 = PointCloud(bnd)
    discretize!(cloud2, spacing; alg=VanDerSandeFornberg(), max_points=100)
    @test length(volume(cloud2)) <= 100
    @test length(volume(cloud2)) > 0
end

@testitem "discretize with FornbergFlyer (2D)" setup=[TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.2m)

    cloud = @test_logs (
        :warn, "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it."
    ) discretize(boundary, spacing; alg=FornbergFlyer(), max_points=100)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 100

    cloud2 = PointCloud(boundary)
    discretize!(cloud2, spacing; alg=FornbergFlyer(), max_points=100)
    @test length(volume(cloud2)) <= 100
end

@testitem "discretize default algorithms" setup=[TestData, CommonImports] begin
    @testset "3D default (SlakKosec)" begin
        points =
            Point.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        boundary = PointBoundary(points)
        spacing = ConstantSpacing(0.5m)

        cloud = discretize(boundary, spacing; max_points=50)
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
        ) discretize(boundary, spacing; max_points=100)
        @test cloud isa PointCloud
        @test length(volume(cloud)) <= 100
    end
end

@testitem "Algorithm constructors" setup=[TestData, CommonImports] begin
    alg1 = SlakKosec()
    @test alg1.n == 10

    alg2 = SlakKosec(15)
    @test alg2.n == 15

    alg3 = VanDerSandeFornberg()
    @test alg3 isa VanDerSandeFornberg

    alg4 = FornbergFlyer()
    @test alg4 isa FornbergFlyer
end

@testitem "max_points limit" setup=[TestData, CommonImports] begin
    points = Point.([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.1m)

    max_pts = 20
    cloud = discretize(boundary, spacing; alg=SlakKosec(), max_points=max_pts)
    @test length(volume(cloud)) <= max_pts
end
