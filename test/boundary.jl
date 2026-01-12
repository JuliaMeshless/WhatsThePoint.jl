@testitem "PointBoundary with Points" setup=[TestData, CommonImports] begin
    N = 10
    pts = rand(Point, N)
    b = PointBoundary(pts)
    @test all(points(b) .== pts)
    @test point(b[:surface1]) == pts
end

@testitem "PointBoundary from file" setup=[TestData, CommonImports] begin
    b = PointBoundary(TestData.BIFURCATION_PATH)
    @test length(b) == 24780
    @test hassurface(b, :surface1)
end

@testitem "PointBoundary Base Methods" setup=[TestData, CommonImports] begin
    N = 10
    b = PointBoundary(rand(Point, N))
    @test length(b) == N
    @test size(b) == (N,)

    points = rand(Point, N)
    surf = PointSurface(points)
    @test_throws ArgumentError b[:surface1] = surf
    b[:surface2] = surf
    @test b[:surface2] == surf

    @testset "iterate" begin
        points = rand(Point, N)
        b = PointBoundary(points)
        for (i, p) in enumerate(b)
            @test p.point == points[i]
        end
    end
end

@testitem "PointBoundary setindex!" setup=[TestData, CommonImports] begin
    N = 10
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    points2 = rand(Point, N)
    surf2 = PointSurface(points2)
    @test_nowarn b[:newsurface] = surf2
    @test hassurface(b, :newsurface)
    @test point(b[:newsurface]) == points2

    @test_throws ArgumentError b[:newsurface] = surf2
end

@testitem "PointBoundary to()" setup=[TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    b = PointBoundary(points)
    coords = to(b)
    @test length(coords) == N
    @test all(coords .== to.(points))
end

@testitem "PointBoundary centroid()" setup=[TestData, CommonImports] begin
    N = 10
    pts = rand(Point, N)
    b = PointBoundary(pts)
    c = centroid(b)
    @test c isa Point
    expected_centroid = centroid(pts)
    @test c == expected_centroid
end

@testitem "PointBoundary boundingbox()" setup=[TestData, CommonImports] begin
    N = 10
    pts = rand(Point, N)
    b = PointBoundary(pts)
    bbox = boundingbox(b)
    @test bbox isa Box
    expected_bbox = boundingbox(pts)
    @test bbox == expected_bbox
end

@testitem "PointBoundary normal() and area()" setup=[TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    normals = [rand(3) .* m for _ in 1:N]
    areas = rand(N) * m^2
    surf = PointSurface(points, normals, areas)

    b = PointBoundary(points, normals, areas)
    @test normal(b) == normals
    @test area(b) == areas

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

@testitem "PointBoundary points()" setup=[TestData, CommonImports] begin
    N = 10
    pts = rand(Point, N)
    b = PointBoundary(pts)
    result = points(b)
    @test length(result) == N
    @test all(result .== pts)

    pts2 = rand(Point, N)
    surf2 = PointSurface(pts2)
    b[:surface2] = surf2
    result = points(b)
    @test length(result) == 2 * N
    @test all(result .== vcat(pts, pts2))
end

@testitem "PointBoundary Meshes.nelements()" setup=[TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    b = PointBoundary(points)
    @test Meshes.nelements(b) == N

    points2 = rand(Point, N ÷ 2)
    surf2 = PointSurface(points2)
    b[:surface2] = surf2
    @test Meshes.nelements(b) == N + N ÷ 2
end

@testitem "PointBoundary delete!()" setup=[TestData, CommonImports] begin
    N = 10
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    points2 = rand(Point, N)
    b[:surface2] = PointSurface(points2)
    points3 = rand(Point, N)
    b[:surface3] = PointSurface(points3)

    @test length(b) == 3 * N
    @test hassurface(b, :surface2)

    delete!(b, :surface2)
    @test length(b) == 2 * N
    @test !hassurface(b, :surface2)
    @test hassurface(b, :surface1)
    @test hassurface(b, :surface3)

    delete!(b, :surface3)
    @test length(b) == N
    @test !hassurface(b, :surface3)
    @test hassurface(b, :surface1)
end

@testitem "PointBoundary names()" setup=[TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    b = PointBoundary(points)

    surface_names = collect(names(b))
    @test length(surface_names) == 1
    @test :surface1 in surface_names

    b[:surface2] = PointSurface(rand(Point, N))
    b[:mysurface] = PointSurface(rand(Point, N))
    surface_names = collect(names(b))
    @test length(surface_names) == 3
    @test :surface1 in surface_names
    @test :surface2 in surface_names
    @test :mysurface in surface_names
end

@testitem "PointBoundary surfaces()" setup=[TestData, CommonImports] begin
    N = 10
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    surfs = collect(surfaces(b))
    @test length(surfs) == 1
    @test surfs[1] isa PointSurface
    @test length(surfs[1]) == N

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

@testitem "PointBoundary Pretty Printing" setup=[TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    b = PointBoundary(points)
    io = IOBuffer()
    show(io, MIME("text/plain"), b)
    output = String(take!(io))
    @test contains(output, "PointBoundary")
    @test contains(output, "$(N) points")
    @test contains(output, "Surfaces")
    @test contains(output, "surface1")

    points2 = rand(Point, N)
    b[:surface2] = PointSurface(points2)
    b[:mysurface] = PointSurface(rand(Point, N))
    io = IOBuffer()
    show(io, MIME("text/plain"), b)
    output = String(take!(io))
    @test contains(output, "$(3 * N) points")
    @test contains(output, "surface1")
    @test contains(output, "surface2")
    @test contains(output, "mysurface")
end
