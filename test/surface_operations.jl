using WhatsThePoint
using Meshes
using Unitful: m, °

N = 10

@testset "add_surface!" begin
    # Test adding a surface to an existing boundary
    points1 = rand(Point, N)
    b = PointBoundary(points1)

    # Add a new surface with a different name - should succeed
    points2 = rand(Point, N)
    @test_nowarn add_surface!(b, points2, :newsurface)
    @test hassurface(b, :newsurface)
    @test point(b[:newsurface]) == points2

    # Try to add a surface with existing name - should throw error
    @test_throws ArgumentError add_surface!(b, points2, :newsurface)

    # Verify original surface is unchanged
    @test point(b[:surface1]) == points1
end

@testset "combine_surfaces!" begin
    # Create boundary with multiple surfaces
    points1 = rand(Point, N)
    points2 = rand(Point, N)
    points3 = rand(Point, N)

    b = PointBoundary(points1)
    add_surface!(b, points2, :surface2)
    add_surface!(b, points3, :surface3)

    @test length(surfaces(b)) == 3
    original_total = length(b)

    # Combine two surfaces
    combine_surfaces!(b, :surface2, :surface3)

    # Should now have 2 surfaces (surface1 and surface2)
    @test length(surfaces(b)) == 2
    @test hassurface(b, :surface1)
    @test hassurface(b, :surface2)
    @test !hassurface(b, :surface3)

    # Total points should remain the same
    @test length(b) == original_total

    # Combined surface should have points from both original surfaces
    @test length(b[:surface2]) == 2 * N
end

@testset "combine_surfaces! - multiple surfaces" begin
    # Test combining more than 2 surfaces
    b = PointBoundary(rand(Point, N))
    add_surface!(b, rand(Point, N), :surface2)
    add_surface!(b, rand(Point, N), :surface3)
    add_surface!(b, rand(Point, N), :surface4)

    original_total = length(b)

    # Combine three surfaces
    combine_surfaces!(b, :surface2, :surface3, :surface4)

    @test length(surfaces(b)) == 2
    @test hassurface(b, :surface1)
    @test hassurface(b, :surface2)
    @test !hassurface(b, :surface3)
    @test !hassurface(b, :surface4)
    @test length(b) == original_total
end

@testset "combine_surfaces! - nonexistent surface" begin
    b = PointBoundary(rand(Point, N))
    add_surface!(b, rand(Point, N), :surface2)

    # Try to combine with a surface that doesn't exist
    @test_throws AssertionError combine_surfaces!(b, :surface1, :nonexistent)
end

@testset "split_surface! - single surface" begin
    # Test splitting when there's only one surface
    boundary = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(surfaces(boundary)) == 1

    split_surface!(boundary, 80°)

    # Should be split into multiple surfaces
    @test length(surfaces(boundary)) > 1

    # All points should still be accounted for
    @test length(boundary) == 24780
end

@testset "split_surface! - by target surface name" begin
    # Test splitting a specific surface by name
    boundary = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))

    # Split the surface
    split_surface!(boundary, :surface1, 80°)

    # Should have multiple new surfaces (surface names are reused starting from surface1)
    @test length(surfaces(boundary)) > 1

    # All points should still be accounted for
    @test length(boundary) == 24780
end

@testset "split_surface! - by PointSurface object" begin
    # Test splitting by passing PointSurface directly
    boundary = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))
    surf = boundary[:surface1]
    original_length = length(boundary)

    # When passing PointSurface directly, the original surface must be manually removed first
    delete!(boundary, :surface1)

    # Split the surface
    split_surface!(boundary, surf, 80°)

    # Should have multiple new surfaces
    @test length(surfaces(boundary)) > 1

    # All points should still be accounted for
    @test length(boundary) == original_length
end

@testset "split_surface! - with PointCloud" begin
    # Test that split_surface! works with PointCloud
    cloud = PointCloud(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(surfaces(cloud)) == 1

    split_surface!(cloud, 80°)

    # Should be split into multiple surfaces
    @test length(surfaces(cloud)) > 1

    # Boundary points should still be accounted for
    @test length(WhatsThePoint.boundary(cloud)) == 24780
end

@testset "split_surface! - multiple surfaces error" begin
    # Test that splitting without specifying target fails when multiple surfaces exist
    boundary = PointBoundary(rand(Point, N))
    add_surface!(boundary, rand(Point, N), :surface2)

    @test length(surfaces(boundary)) == 2
    @test_throws AssertionError split_surface!(boundary, 80°)
end

@testset "split_surface! - nonexistent surface error" begin
    boundary = PointBoundary(rand(Point, N))

    # Try to split a surface that doesn't exist
    @test_throws AssertionError split_surface!(boundary, :nonexistent, 80°)
end

@testset "split_surface! - different angles" begin
    # Test that different angles produce different splits
    boundary1 = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))
    boundary2 = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))

    split_surface!(boundary1, 45°)
    split_surface!(boundary2, 60°)

    # Different angles should generally produce different numbers of surfaces
    # (though not guaranteed for all geometries)
    n_surfaces_45 = length(surfaces(boundary1))
    n_surfaces_60 = length(surfaces(boundary2))

    # Smaller angles should produce more surfaces (stricter grouping)
    # 45° should produce more or equal surfaces than 60°
    @test n_surfaces_45 >= n_surfaces_60
    # At minimum, 45° should produce more than 1 surface for this geometry
    @test n_surfaces_45 > 1
end

@testset "split_surface! - custom k parameter" begin
    # Test that custom k parameter works
    boundary = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))

    # Use a different k value for nearest neighbors
    @test_nowarn split_surface!(boundary, 80°; k=15)
    @test length(surfaces(boundary)) > 1
end

@testset "surface operations integration" begin
    # Test a realistic workflow: split then combine
    boundary = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))
    original_length = length(boundary)

    # Split the surface
    split_surface!(boundary, 80°)
    num_surfaces_after_split = length(surfaces(boundary))
    @test num_surfaces_after_split > 1

    # Get the names of surfaces to combine
    surface_names = collect(names(boundary))
    @test length(surface_names) == num_surfaces_after_split

    # Combine first two surfaces
    if length(surface_names) >= 2
        combine_surfaces!(boundary, surface_names[1], surface_names[2])
        @test length(surfaces(boundary)) == num_surfaces_after_split - 1
    end

    # Total points should be preserved
    @test length(boundary) == original_length
end
