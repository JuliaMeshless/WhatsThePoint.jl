using WhatsThePoint
using WhatsThePoint: volume, boundary
using Meshes
using Unitful: m

@testset "Spacing Types" begin
    @testset "ConstantSpacing" begin
        s = ConstantSpacing(1.0m)
        @test s.Δx == 1.0m

        # Test calling operator with no arguments
        @test s() == 1.0m

        # Test calling operator with point argument
        p = Point(0.0, 0.0, 0.0)
        @test s(p) == 1.0m

        # Test with different constant value
        s2 = ConstantSpacing(0.5m)
        @test s2() == 0.5m
        @test s2(p) == 0.5m
    end

    @testset "LogLike" begin
        # Create a simple cloud for LogLike spacing with enough points for k-nearest neighbors
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

        # Test calling operator on a point
        test_point = Point(0.5, 0.5, 0.5)
        spacing_value = s(test_point)
        @test spacing_value > 0m
        @test spacing_value isa typeof(base_size)

        # Point closer to boundary should have smaller spacing
        close_point = Point(0.01, 0.0, 0.0)
        far_point = Point(5.0, 5.0, 5.0)
        @test s(close_point) < s(far_point)
    end
end

@testset "calculate_ninit" begin
    @testset "3D ConstantSpacing" begin
        # Create simple 3D boundary
        points =
            Point.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        cloud = PointCloud(PointBoundary(points))
        spacing = ConstantSpacing(0.1m)

        ninit = WhatsThePoint.calculate_ninit(cloud, spacing)
        @test ninit isa Tuple{Int,Int}
        @test ninit[1] > 0
        @test ninit[2] > 0
    end

    @testset "2D ConstantSpacing" begin
        # Create simple 2D boundary
        points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        cloud = PointCloud(PointBoundary(points))
        spacing = ConstantSpacing(0.1m)

        ninit = WhatsThePoint.calculate_ninit(cloud, spacing)
        @test ninit isa Int
        @test ninit > 0
    end
end

@testset "discretize with SlakKosec (3D)" begin
    # Use STL file to get proper areas for isinside to work correctly
    # SlakKosec requires spacing < distance to nearest boundary point, so use small spacing
    bnd = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(0.5m)

    # Test discretize (creates new cloud)
    cloud = discretize(bnd, spacing; alg=SlakKosec(), max_points=50)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 50
    @test length(boundary(cloud)) == length(bnd)

    # Test discretize! (modifies existing cloud)
    cloud2 = PointCloud(bnd)
    @test length(volume(cloud2)) == 0
    discretize!(cloud2, spacing; alg=SlakKosec(), max_points=50)
    @test length(volume(cloud2)) <= 50
    @test length(volume(cloud2)) > 0
end

@testset "discretize with VanDerSandeFornberg (3D)" begin
    # Use STL file to get proper areas for isinside to work correctly
    bnd = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(5.0m)

    # Test discretize
    cloud = discretize(bnd, spacing; alg=VanDerSandeFornberg(), max_points=100)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 100

    # Test discretize!
    cloud2 = PointCloud(bnd)
    discretize!(cloud2, spacing; alg=VanDerSandeFornberg(), max_points=100)
    @test length(volume(cloud2)) <= 100
    @test length(volume(cloud2)) > 0
end

@testset "discretize with FornbergFlyer (2D)" begin
    # Create a simple 2D square boundary
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.2m)

    # Test discretize - should automatically use FornbergFlyer for 2D
    cloud = @test_logs (
        :warn, "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it."
    ) discretize(boundary, spacing; alg=FornbergFlyer(), max_points=100)
    @test cloud isa PointCloud
    @test length(volume(cloud)) <= 100

    # Test discretize!
    cloud2 = PointCloud(boundary)
    discretize!(cloud2, spacing; alg=FornbergFlyer(), max_points=100)
    @test length(volume(cloud2)) <= 100
end

@testset "discretize default algorithms" begin
    @testset "3D default (SlakKosec)" begin
        points =
            Point.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        boundary = PointBoundary(points)
        spacing = ConstantSpacing(0.5m)

        # Should use SlakKosec by default for 3D
        cloud = discretize(boundary, spacing; max_points=50)
        @test cloud isa PointCloud
        @test length(volume(cloud)) <= 50
    end

    @testset "2D default (FornbergFlyer)" begin
        points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        boundary = PointBoundary(points)
        spacing = ConstantSpacing(0.2m)

        # Should use FornbergFlyer by default for 2D and warn
        cloud = @test_logs (
            :warn,
            "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it.",
        ) discretize(boundary, spacing; max_points=100)
        @test cloud isa PointCloud
        @test length(volume(cloud)) <= 100
    end
end

@testset "Algorithm constructors" begin
    # Test SlakKosec constructors
    alg1 = SlakKosec()
    @test alg1.n == 10  # default value

    alg2 = SlakKosec(15)
    @test alg2.n == 15

    # Test VanDerSandeFornberg constructor
    alg3 = VanDerSandeFornberg()
    @test alg3 isa VanDerSandeFornberg

    # Test FornbergFlyer constructor
    alg4 = FornbergFlyer()
    @test alg4 isa FornbergFlyer
end

@testset "max_points limit" begin
    # Test that discretization respects max_points limit
    points = Point.([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)])
    boundary = PointBoundary(points)
    spacing = ConstantSpacing(0.1m)  # Small spacing to potentially generate many points

    max_pts = 20
    cloud = discretize(boundary, spacing; alg=SlakKosec(), max_points=max_pts)
    @test length(volume(cloud)) <= max_pts
end
