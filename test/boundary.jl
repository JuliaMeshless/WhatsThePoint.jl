using WhatsThePoint
using Meshes
using Random

N = 10

@testset "PointCloud with PointBoundary" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    @test all(pointify(b) .== points)
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
        for (i, p) in enumerate(b)
            @test p.point == points[i]
        end
    end
end

@testset "add_surface!" begin
    # Test that add_surface! works correctly (issue #48)
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    # Add a new surface with a different name - should succeed
    points2 = rand(Point, N)
    @test_nowarn add_surface!(b, points2, :newsurface)
    @test hassurface(b, :newsurface)
    @test point(b[:newsurface]) == points2

    # Try to add a surface with existing name - should throw error
    @test_throws ArgumentError add_surface!(b, points2, :newsurface)
end
