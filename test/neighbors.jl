using WhatsThePoint
using Meshes
using Meshes: Euclidean
using Unitful
using Unitful: m
using LinearAlgebra

N = 20

@testset "KNearestSearch Constructor" begin
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

@testset "search with PointCloud" begin
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

@testset "search with PointBoundary" begin
    points = rand(Point, N)
    boundary = PointBoundary(points)

    k = 4
    method = KNearestSearch(boundary, k)
    neighbors = search(boundary, method)

    @test neighbors isa Vector
    @test length(neighbors) == N
    @test all(length(n) == k for n in neighbors)
end

@testset "search with PointSurface" begin
    points = rand(Point, N)
    surf = PointSurface(points)

    k = 5
    method = KNearestSearch(surf, k)
    neighbors = search(surf, method)

    @test neighbors isa Vector
    @test length(neighbors) == N
    @test all(length(n) == k for n in neighbors)
end

@testset "searchdists with PointCloud" begin
    circle_points = Point.([(cos(θ), sin(θ)) for θ in range(0, 2π, length=N+1)[1:end-1]])
    cloud = PointCloud(PointBoundary(circle_points))

    k = 3
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)

    @testset "Return type and structure" begin
        @test results isa Vector
        @test length(results) == N
        @test all(length(idxs) == k && length(ds) == k for (idxs, ds) in results)
    end

    @testset "Distance properties" begin
        # All distances should be non-negative
        @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)

        # Distance to self should be zero (first element)
        @test all(isapprox(ds[1], 0.0m; atol=1e-10m) for (_, ds) in results)

        # Distances should be sorted (nearest to farthest)
        @test all(issorted(ds) for (_, ds) in results)
    end
end

@testset "searchdists with PointBoundary" begin
    points = rand(Point, N)
    boundary = PointBoundary(points)

    k = 4
    method = KNearestSearch(boundary, k)
    results = searchdists(boundary, method)

    @test results isa Vector
    @test length(results) == N
    @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
end

@testset "searchdists with PointSurface" begin
    points = rand(Point, N)
    surf = PointSurface(points)

    k = 5
    method = KNearestSearch(surf, k)
    results = searchdists(surf, method)

    @test results isa Vector
    @test length(results) == N
    @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
end

@testset "searchdists with Vec point" begin
    # Create a simple 3D point cloud (rand(Point, N) returns 3D points with units)
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    k = 3
    method = KNearestSearch(cloud, k)

    # Test with a 3D Vec point (must match cloud dimensionality)
    test_point = Vec(0.5m, 0.5m, 0.5m)
    idxs, dists = searchdists(test_point, method)

    @testset "Return type and structure" begin
        @test idxs isa AbstractVector{Int}
        @test dists isa AbstractVector
        @test length(idxs) == k
        @test length(dists) == k
    end

    @testset "Distance properties" begin
        # All distances should be non-negative
        @test all(d >= 0.0m for d in dists)

        # Distances should be sorted (nearest to farthest)
        @test issorted(dists)
    end
end

@testset "Integration with real geometry" begin
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
        results = searchdists(cloud, method)
        @test length(results) == length(cloud)
        @test all(length(ds) == k for (_, ds) in results)
        @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
    end
end

@testset "Edge cases" begin
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

        results = searchdists(cloud, method)
        @test all(isapprox(ds[1], 0.0m; atol=1e-10m) for (_, ds) in results)
    end
end

@testset "Units support" begin
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
        results = searchdists(cloud, method)
        @test length(results) == N
        @test all(length(ds) == k for (_, ds) in results)
        # Distances should have units
        @test all(all(d isa Unitful.Length for d in ds) for (_, ds) in results)
    end
end
