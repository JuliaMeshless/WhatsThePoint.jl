using PointClouds
using Meshes
using Random

N = 10

@testset "PointCloud with PointPart" begin
    part = PointPart(rand(Point, N))
    cloud = PointCloud(part)
    @test cloud.volume isa PointVolume
    @test cloud.points == part.points
    @test cloud.surfaces[:surface1] == part.surfaces[:surface1]
end

@testset "PointCloud from file" begin
    cloud = PointCloud(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(cloud) == 24780
    @test haskey(cloud.surfaces, :surface1)
end

@testset "Base Methods" begin
    part = PointPart(rand(Point, N))
    cloud = PointCloud(part)
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
        part = PointPart(points)
        cloud = PointCloud(part)
        for (i, point) in enumerate(cloud)
            @test point == points[i]
        end
    end
end

@testset "make_memory_contiguous!" begin
    points = rand(Point, N)
    part = PointPart(points)
    test_points = copy(points)
    cloud = PointCloud(part)
    permutations = randperm(N)
    PointClouds.make_memory_contiguous!(cloud, permutations)
    @test all(cloud.points .== points[permutations])
end
