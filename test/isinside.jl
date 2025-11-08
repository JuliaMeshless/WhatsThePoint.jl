using WhatsThePoint
using Meshes

@testset "2D" begin
    points = PointSet(Point.([(0, 0), (1, 0), (1, 1), (0, 1)]))
    @test isinside(Point(0.5, 0.5), points)
    @test !isinside(Point(0.5, 1.5), points)
    @test !isinside(Point(0.5, 1 + eps()), points)
end

@testset "3D" begin
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    display(boundingbox(boundary))
    @test isinside(Point(0.5, 0.5, 0.5), boundary)
    for p in [Point(0.0, 0.5, 0.5), Point(1.0, 0.5, 0.5), Point(0.5, 0.0, 0.5), Point(0.5, 1.0, 0.5), Point(0.5, 0.5, 0.0), Point(0.5, 0.5, 1.0)]
        @test isinside(p, boundary)
    end
    @test isinside(Point(10.0, 10.0, 10.0), boundary)
    @test !isinside(Point(0.5, 0.5, -0.5), boundary)
    @test !isinside(Point(0.5, 0.5, -0.001), boundary)
end
