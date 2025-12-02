@testitem "add_surface!" setup=[TestData, CommonImports] begin
    N = 10
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    points2 = rand(Point, N)
    @test_nowarn add_surface!(b, points2, :newsurface)
    @test hassurface(b, :newsurface)
    @test point(b[:newsurface]) == points2

    @test_throws ArgumentError add_surface!(b, points2, :newsurface)

    @test point(b[:surface1]) == points1
end

@testitem "combine_surfaces!" setup=[TestData, CommonImports] begin
    N = 10
    points1 = rand(Point, N)
    points2 = rand(Point, N)
    points3 = rand(Point, N)

    b = PointBoundary(points1)
    add_surface!(b, points2, :surface2)
    add_surface!(b, points3, :surface3)

    @test length(surfaces(b)) == 3
    original_total = length(b)

    combine_surfaces!(b, :surface2, :surface3)

    @test length(surfaces(b)) == 2
    @test hassurface(b, :surface1)
    @test hassurface(b, :surface2)
    @test !hassurface(b, :surface3)

    @test length(b) == original_total

    @test length(b[:surface2]) == 2 * N
end

@testitem "combine_surfaces! - multiple surfaces" setup=[TestData, CommonImports] begin
    N = 10
    b = PointBoundary(rand(Point, N))
    add_surface!(b, rand(Point, N), :surface2)
    add_surface!(b, rand(Point, N), :surface3)
    add_surface!(b, rand(Point, N), :surface4)

    original_total = length(b)

    combine_surfaces!(b, :surface2, :surface3, :surface4)

    @test length(surfaces(b)) == 2
    @test hassurface(b, :surface1)
    @test hassurface(b, :surface2)
    @test !hassurface(b, :surface3)
    @test !hassurface(b, :surface4)
    @test length(b) == original_total
end

@testitem "combine_surfaces! - nonexistent surface" setup=[TestData, CommonImports] begin
    N = 10
    b = PointBoundary(rand(Point, N))
    add_surface!(b, rand(Point, N), :surface2)

    @test_throws AssertionError combine_surfaces!(b, :surface1, :nonexistent)
end

@testitem "split_surface! - single surface" setup=[TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BIFURCATION_PATH)
    @test length(surfaces(boundary)) == 1

    split_surface!(boundary, 80°)

    @test length(surfaces(boundary)) > 1

    @test length(boundary) == 24780
end

@testitem "split_surface! - by target surface name" setup=[TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BIFURCATION_PATH)

    split_surface!(boundary, :surface1, 80°)

    @test length(surfaces(boundary)) > 1

    @test length(boundary) == 24780
end

@testitem "split_surface! - by PointSurface object" setup=[TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BIFURCATION_PATH)
    surf = boundary[:surface1]
    original_length = length(boundary)

    delete!(boundary, :surface1)

    split_surface!(boundary, surf, 80°)

    @test length(surfaces(boundary)) > 1

    @test length(boundary) == original_length
end

@testitem "split_surface! - with PointCloud" setup=[TestData, CommonImports] begin
    using WhatsThePoint: boundary
    cloud = PointCloud(TestData.BIFURCATION_PATH)
    @test length(surfaces(cloud)) == 1

    split_surface!(cloud, 80°)

    @test length(surfaces(cloud)) > 1

    @test length(boundary(cloud)) == 24780
end

@testitem "split_surface! - multiple surfaces error" setup=[TestData, CommonImports] begin
    N = 10
    boundary = PointBoundary(rand(Point, N))
    add_surface!(boundary, rand(Point, N), :surface2)

    @test length(surfaces(boundary)) == 2
    @test_throws AssertionError split_surface!(boundary, 80°)
end

@testitem "split_surface! - nonexistent surface error" setup=[TestData, CommonImports] begin
    N = 10
    boundary = PointBoundary(rand(Point, N))

    @test_throws AssertionError split_surface!(boundary, :nonexistent, 80°)
end

@testitem "split_surface! - different angles" setup=[TestData, CommonImports] begin
    boundary1 = PointBoundary(TestData.BIFURCATION_PATH)
    boundary2 = PointBoundary(TestData.BIFURCATION_PATH)

    split_surface!(boundary1, 45°)
    split_surface!(boundary2, 60°)

    n_surfaces_45 = length(surfaces(boundary1))
    n_surfaces_60 = length(surfaces(boundary2))

    @test n_surfaces_45 >= n_surfaces_60
    @test n_surfaces_45 > 1
end

@testitem "split_surface! - custom k parameter" setup=[TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BIFURCATION_PATH)

    @test_nowarn split_surface!(boundary, 80°; k=15)
    @test length(surfaces(boundary)) > 1
end

@testitem "surface operations integration" setup=[TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BIFURCATION_PATH)
    original_length = length(boundary)

    split_surface!(boundary, 80°)
    num_surfaces_after_split = length(surfaces(boundary))
    @test num_surfaces_after_split > 1

    surface_names = collect(names(boundary))
    @test length(surface_names) == num_surfaces_after_split

    if length(surface_names) >= 2
        combine_surfaces!(boundary, surface_names[1], surface_names[2])
        @test length(surfaces(boundary)) == num_surfaces_after_split - 1
    end

    @test length(boundary) == original_length
end
