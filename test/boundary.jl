using WhatsThePoint
using Meshes
using Random
using Unitful: m

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

@testset "to(boundary)" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    coords = to(b)
    @test length(coords) == N
    @test all(coords .== to.(points))
end

@testset "centroid(boundary)" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    c = centroid(b)
    @test c isa Point
    expected_centroid = centroid(PointSet(points))
    @test c == expected_centroid
end

@testset "boundingbox(boundary)" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    bbox = boundingbox(b)
    @test bbox isa Box
    expected_bbox = boundingbox(PointSet(points))
    @test bbox == expected_bbox
end

@testset "normal(boundary) and area(boundary)" begin
    points = rand(Point, N)
    normals = [rand(3) .* m for _ in 1:N]
    areas = rand(N) * m^2
    surf = PointSurface(points, normals, areas)

    # Create boundary with single surface
    b = PointBoundary(points, normals, areas)
    @test normal(b) == normals
    @test area(b) == areas

    # Create boundary with multiple surfaces
    points2 = rand(Point, N)
    normals2 = [rand(3) .* m for _ in 1:N]
    areas2 = rand(N) * m^2
    surf2 = PointSurface(points2, normals2, areas2)
    b[:surface2] = surf2

    all_normals = normal(b)
    all_areas = area(b)
    @test length(all_normals) == 2 * N
    @test length(all_areas) == 2 * N
    @test all_normals == vcat(normals, normals2)
    @test all_areas == vcat(areas, areas2)
end

@testset "Meshes.pointify(boundary)" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    pointified = Meshes.pointify(b)
    @test length(pointified) == N
    @test all(pointified .== points)

    # Test with multiple surfaces
    points2 = rand(Point, N)
    surf2 = PointSurface(points2)
    b[:surface2] = surf2
    pointified = Meshes.pointify(b)
    @test length(pointified) == 2 * N
    @test all(pointified .== vcat(points, points2))
end

@testset "Meshes.nelements(boundary)" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    @test Meshes.nelements(b) == N

    # Test with multiple surfaces
    points2 = rand(Point, N ÷ 2)
    surf2 = PointSurface(points2)
    b[:surface2] = surf2
    @test Meshes.nelements(b) == N + N ÷ 2
end

@testset "delete!(boundary, name)" begin
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    # Add additional surfaces
    points2 = rand(Point, N)
    b[:surface2] = PointSurface(points2)
    points3 = rand(Point, N)
    b[:surface3] = PointSurface(points3)

    @test length(b) == 3 * N
    @test hassurface(b, :surface2)

    # Delete surface2
    delete!(b, :surface2)
    @test length(b) == 2 * N
    @test !hassurface(b, :surface2)
    @test hassurface(b, :surface1)
    @test hassurface(b, :surface3)

    # Delete surface3
    delete!(b, :surface3)
    @test length(b) == N
    @test !hassurface(b, :surface3)
    @test hassurface(b, :surface1)
end

@testset "names(boundary)" begin
    points = rand(Point, N)
    b = PointBoundary(points)

    # Single surface
    surface_names = collect(names(b))
    @test length(surface_names) == 1
    @test :surface1 in surface_names

    # Multiple surfaces
    b[:surface2] = PointSurface(rand(Point, N))
    b[:mysurface] = PointSurface(rand(Point, N))
    surface_names = collect(names(b))
    @test length(surface_names) == 3
    @test :surface1 in surface_names
    @test :surface2 in surface_names
    @test :mysurface in surface_names
end

@testset "surfaces(boundary)" begin
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    # Single surface
    surfs = collect(surfaces(b))
    @test length(surfs) == 1
    @test surfs[1] isa PointSurface
    @test length(surfs[1]) == N

    # Multiple surfaces
    points2 = rand(Point, N ÷ 2)
    surf2 = PointSurface(points2)
    b[:surface2] = surf2

    points3 = rand(Point, N * 2)
    surf3 = PointSurface(points3)
    b[:surface3] = surf3

    surfs = collect(surfaces(b))
    @test length(surfs) == 3
    @test all(s -> s isa PointSurface, surfs)
    total_points = sum(length, surfs)
    @test total_points == N + N ÷ 2 + N * 2
end
