using PointClouds
using Meshes
using Random

N = 10

@testset "PointCloud with PointBoundary" begin
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test cloud.volume isa PointVolume
    @test PointClouds.boundary(cloud)[:surface1] == b[:surface1]
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
        for (i, point) in enumerate(cloud)
            @test point == points[i]
        end
    end
end
