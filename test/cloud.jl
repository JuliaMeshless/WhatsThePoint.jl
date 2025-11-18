using WhatsThePoint
using Meshes
using Random

N = 10

@testset "PointCloud with PointBoundary" begin
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test cloud.volume isa PointVolume
    @test WhatsThePoint.boundary(cloud)[:surface1] == b[:surface1]
end

@testset "PointCloud from file" begin
    cloud = PointCloud(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(cloud) == 24780
    @test hassurface(cloud, :surface1)
end

@testset "PointCloud from PointBoundary" begin
    cloud = PointCloud(PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl")))
    @test length(cloud) == 24780
    @test hassurface(cloud, :surface1)
end

@testset "Base Methods" begin
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test length(cloud) == N
    @test size(cloud) == (N,)

    points = rand(Point, N)
    surf = PointSurface(points)
    @test_throws ArgumentError cloud[:surface1] = surf
    cloud[:surface2] = surf
    @test cloud[:surface2] == surf

    # Test the iterate method
    @testset "iterate" begin
        points = rand(Point, N)
        b = PointBoundary(points)
        cloud = PointCloud(b)
        for (i, p) in enumerate(cloud)
            @test p.point == points[i]
        end
    end
end

@testset "normal and area functions" begin
    # Test that normal(cloud) and area(cloud) work correctly (issue #49)
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    @test_nowarn normal(cloud)
    @test_nowarn area(cloud)
    @test length(normal(cloud)) == N
    @test length(area(cloud)) == N
end
