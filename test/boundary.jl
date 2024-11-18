using PointClouds
using Meshes
using Random

N = 10

@testset "PointCloud with PointBoundary" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    @test all(b.points .== points)
    @test point(b[:surface1]) == points
end

@testset "PointCloud from file" begin
    b = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(b) == 24780
    @test hassurface(b, :surface1)
end

@testset "Base Methods" begin
    b = PointBoundary(rand(Point, N))
    @test length(b) == N
    @test size(b) == (N,)

    points = rand(Point, N)
    surf = PointSurface(points)
    @test_throws ArgumentError b[:surface1] = surf
    b[:surface2] = surf
    @test b[:surface2] == surf

    # Test the iterate method
    @testset "iterate" begin
        points = rand(Point, N)
        b = PointBoundary(points)
        for (i, point) in enumerate(b)
            @test point == points[i]
        end
    end
end
