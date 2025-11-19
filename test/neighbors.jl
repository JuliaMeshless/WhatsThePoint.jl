using WhatsThePoint
using Meshes
using Unitful: m
using LinearAlgebra

N = 20

@testitem "KNearestSearch Constructor" begin
    # Test with PointCloud
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    k = 5

    @testset "PointCloud" begin
        method = KNearestSearch(cloud, k)
        @test method isa KNearestSearch
        @test method.k == k
    end

    @testset "PointBoundary" begin
        boundary = PointBoundary(points)
        method = KNearestSearch(boundary, k)
        @test method isa KNearestSearch
        @test method.k == k
    end

    @testset "PointSurface" begin
        surf = PointSurface(points)
        method = KNearestSearch(surf, k)
        @test method isa KNearestSearch
        @test method.k == k
    end

    @testset "with custom metric" begin
        cloud = PointCloud(PointBoundary(points))
        method = KNearestSearch(cloud, k; metric=Euclidean())
        @test method isa KNearestSearch
    end
end

@testitem "search with PointCloud" begin
    # Create a simple 2D circle for predictable neighbor relationships
    circle_points = Point.([(cos(θ), sin(θ)) for θ in range(0, 2π, length=N+1)[1:end-1]])
    cloud = PointCloud(PointBoundary(circle_points))

    k = 3
    method = KNearestSearch(cloud, k)
    neighbors = search(cloud, method)

    @testset "Return type and structure" begin
        @test neighbors isa Vector
        @test length(neighbors) == N
        @test all(length(n) == k for n in neighbors)
    end

    @testset "Neighbor indices validity" begin
        # All indices should be valid (within 1:N)
        @test all(all(1 <= idx <= N for idx in n) for n in neighbors)
    end

    @testset "Self as nearest neighbor" begin
        # Each point should be its own nearest neighbor (index 1)
        @test all(neighbors[i][1] == i for i in 1:N)
    end
end

@testitem "search with PointBoundary" begin
    points = rand(Point, N)
    boundary = PointBoundary(points)

    k = 4
    method = KNearestSearch(boundary, k)
    neighbors = search(boundary, method)

    @test neighbors isa Vector
    @test length(neighbors) == N
    @test all(length(n) == k for n in neighbors)
end

@testitem "search with PointSurface" begin
    points = rand(Point, N)
    surf = PointSurface(points)

    k = 5
    method = KNearestSearch(surf, k)
    neighbors = search(surf, method)

    @test neighbors isa Vector
    @test length(neighbors) == N
    @test all(length(n) == k for n in neighbors)
end

@testitem "searchdists with PointCloud" begin
    circle_points = Point.([(cos(θ), sin(θ)) for θ in range(0, 2π, length=N+1)[1:end-1]])
    cloud = PointCloud(PointBoundary(circle_points))

    k = 3
    method = KNearestSearch(cloud, k)
    dists = searchdists(cloud, method)

    @testset "Return type and structure" begin
        @test dists isa Vector
        @test length(dists) == N
        @test all(length(d) == k for d in dists)
    end

    @testset "Distance properties" begin
        # All distances should be non-negative
        @test all(all(d >= 0 for d in ds) for ds in dists)

        # Distance to self should be zero (first element)
        @test all(dists[i][1] ≈ 0 atol=1e-10 for i in 1:N)

        # Distances should be sorted (nearest to farthest)
        @test all(issorted(d) for d in dists)
    end
end

@testitem "searchdists with PointBoundary" begin
    points = rand(Point, N)
    boundary = PointBoundary(points)

    k = 4
    method = KNearestSearch(boundary, k)
    dists = searchdists(boundary, method)

    @test dists isa Vector
    @test length(dists) == N
    @test all(all(d >= 0 for d in ds) for ds in dists)
end

@testitem "searchdists with PointSurface" begin
    points = rand(Point, N)
    surf = PointSurface(points)

    k = 5
    method = KNearestSearch(surf, k)
    dists = searchdists(surf, method)

    @test dists isa Vector
    @test length(dists) == N
    @test all(all(d >= 0 for d in ds) for ds in dists)
end

@testitem "searchdists with Vec point" begin
    # Create a simple point cloud
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    k = 3
    method = KNearestSearch(cloud, k)

    # Test with a Vec point (this uses Meshes.Vec which is a StaticArray)
    test_point = Vec(0.5, 0.5)
    dists = searchdists(test_point, method)

    @testset "Return type and structure" begin
        @test dists isa Vector
        @test length(dists) == k
    end

    @testset "Distance properties" begin
        # All distances should be non-negative
        @test all(d >= 0 for d in dists)

        # Distances should be sorted (nearest to farthest)
        @test issorted(dists)
    end
end

@testitem "Integration with real geometry" begin
    # Load a real geometry from test data
    cloud = PointCloud(joinpath(@__DIR__, "data", "box.stl"))

    @testset "Search on imported geometry" begin
        k = 10
        method = KNearestSearch(cloud, k)

        # Test search
        neighbors = search(cloud, method)
        @test length(neighbors) == length(cloud)
        @test all(length(n) == k for n in neighbors)

        # Test searchdists
        dists = searchdists(cloud, method)
        @test length(dists) == length(cloud)
        @test all(length(d) == k for d in dists)
        @test all(all(d >= 0 for d in ds) for ds in dists)
    end
end

@testitem "Edge cases" begin
    @testset "k equals number of points" begin
        points = rand(Point, 5)
        cloud = PointCloud(PointBoundary(points))

        k = 5
        method = KNearestSearch(cloud, k)
        neighbors = search(cloud, method)

        @test all(length(n) == k for n in neighbors)
    end

    @testset "k = 1 (only self)" begin
        points = rand(Point, N)
        cloud = PointCloud(PointBoundary(points))

        k = 1
        method = KNearestSearch(cloud, k)
        neighbors = search(cloud, method)

        @test all(length(n) == 1 for n in neighbors)
        @test all(neighbors[i][1] == i for i in 1:N)

        dists = searchdists(cloud, method)
        @test all(dists[i][1] ≈ 0 atol=1e-10 for i in 1:N)
    end
end

@testitem "Units support" begin
    # Test with unitful points
    points_with_units = [Point(rand() * m, rand() * m) for _ in 1:N]
    cloud = PointCloud(PointBoundary(points_with_units))

    k = 5
    method = KNearestSearch(cloud, k)

    @testset "search with units" begin
        neighbors = search(cloud, method)
        @test length(neighbors) == N
        @test all(length(n) == k for n in neighbors)
    end

    @testset "searchdists with units" begin
        dists = searchdists(cloud, method)
        @test length(dists) == N
        @test all(length(d) == k for d in dists)
        # Distances should have units
        @test all(all(d isa Unitful.Length for d in ds) for ds in dists)
    end
end
