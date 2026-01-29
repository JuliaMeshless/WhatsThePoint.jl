@testitem "KNearestSearch Constructor" setup = [TestData, CommonImports] begin
    N = 20
    points = rand(Point, N)
    k = 5

    @testset "PointCloud" begin
        cloud = PointCloud(PointBoundary(points))
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
        method = KNearestSearch(cloud, k; metric = Euclidean())
        @test method isa KNearestSearch
    end
end

@testitem "search with PointCloud" setup = [TestData, CommonImports] begin
    N = 20
    circle_points =
        Point.([(cos(θ), sin(θ)) for θ in range(0, 2π; length = N + 1)[1:(end-1)]])
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
        @test all(all(1 <= idx <= N for idx in n) for n in neighbors)
    end

    @testset "Self as nearest neighbor" begin
        @test all(neighbors[i][1] == i for i = 1:N)
    end
end

@testitem "search with PointBoundary" setup = [TestData, CommonImports] begin
    N = 20
    points = rand(Point, N)
    boundary = PointBoundary(points)

    k = 4
    method = KNearestSearch(boundary, k)
    neighbors = search(boundary, method)

    @test neighbors isa Vector
    @test length(neighbors) == N
    @test all(length(n) == k for n in neighbors)
end

@testitem "search with PointSurface" setup = [TestData, CommonImports] begin
    N = 20
    points = rand(Point, N)
    surf = PointSurface(points)

    k = 5
    method = KNearestSearch(surf, k)
    neighbors = search(surf, method)

    @test neighbors isa Vector
    @test length(neighbors) == N
    @test all(length(n) == k for n in neighbors)
end

@testitem "searchdists with PointCloud" setup = [TestData, CommonImports] begin
    N = 20
    circle_points =
        Point.([(cos(θ), sin(θ)) for θ in range(0, 2π; length = N + 1)[1:(end-1)]])
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
        @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
        @test all(isapprox(ds[1], 0.0m; atol = 1.0e-10m) for (_, ds) in results)
        @test all(issorted(ds) for (_, ds) in results)
    end
end

@testitem "searchdists with PointBoundary" setup = [TestData, CommonImports] begin
    N = 20
    points = rand(Point, N)
    boundary = PointBoundary(points)

    k = 4
    method = KNearestSearch(boundary, k)
    results = searchdists(boundary, method)

    @test results isa Vector
    @test length(results) == N
    @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
end

@testitem "searchdists with PointSurface" setup = [TestData, CommonImports] begin
    N = 20
    points = rand(Point, N)
    surf = PointSurface(points)

    k = 5
    method = KNearestSearch(surf, k)
    results = searchdists(surf, method)

    @test results isa Vector
    @test length(results) == N
    @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
end

@testitem "KNearestSearch with real geometry" setup = [TestData, CommonImports] begin
    cloud = PointCloud(TestData.BOX_PATH)

    @testset "Search on imported geometry" begin
        k = 10
        method = KNearestSearch(cloud, k)

        neighbors = search(cloud, method)
        @test length(neighbors) == length(cloud)
        @test all(length(n) == k for n in neighbors)

        results = searchdists(cloud, method)
        @test length(results) == length(cloud)
        @test all(length(ds) == k for (_, ds) in results)
        @test all(all(d >= 0.0m for d in ds) for (_, ds) in results)
    end
end

@testitem "KNearestSearch edge cases" setup = [TestData, CommonImports] begin
    N = 20
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
        @test all(neighbors[i][1] == i for i = 1:N)

        results = searchdists(cloud, method)
        @test all(isapprox(ds[1], 0.0m; atol = 1.0e-10m) for (_, ds) in results)
    end
end

@testitem "KNearestSearch units support" setup = [TestData, CommonImports] begin
    N = 20
    points_with_units = [Point(rand() * m, rand() * m) for _ = 1:N]
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
        @test all(all(d isa Unitful.Length for d in ds) for (_, ds) in results)
    end
end
