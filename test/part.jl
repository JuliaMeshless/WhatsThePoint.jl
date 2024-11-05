using PointClouds
using Meshes
using Random

N = 10

@testset "PointCloud with PointPart" begin
    part = PointPart(rand(Point, N))
    @test part.points == part.points
    @test part.surfaces[:surface1] == part.surfaces[:surface1]
end

@testset "PointCloud from file" begin
    part = PointPart(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(part) == 24780
    @test haskey(part.surfaces, :surface1)
end

@testset "Base Methods" begin
    part = PointPart(rand(Point, N))
    @test length(part) == N
    @test size(part) == (N,)

    points = rand(Point, N)
    surf = PointSurface(points)
    @test_throws ArgumentError part[:surface1] = surf
    part[:surface2] = surf
    @test part[:surface2] == surf

    # Test the iterate method
    @testset "iterate" begin
        points = rand(Point, N)
        part = PointPart(points)
        for (i, point) in enumerate(part)
            @test point == points[i]
        end
    end
end
