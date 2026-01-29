@testitem "isinside 2D Vector{Point}" setup = [TestData, CommonImports] begin
    pts = Point.([(0, 0), (1, 0), (1, 1), (0, 1)])
    @test isinside(Point(0.5, 0.5), pts)
    @test !isinside(Point(0.5, 1.5), pts)
    @test !isinside(Point(0.5, 1 + eps()), pts)
end

@testitem "isinside 2D AbstractVector{Point}" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    @test isinside(Point(0.5, 0.5), points)
    @test !isinside(Point(1.5, 0.5), points)
    @test !isinside(Point(0.5, -0.5), points)
end

@testitem "isinside 2D PointSurface" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    normals = [[0.0, -1.0] * m for _ = 1:4]
    areas = [0.5, 0.5, 0.5, 0.5] * m
    surf = PointSurface(points, normals, areas)

    @test isinside(Point(1.0, 1.0), surf)
    @test !isinside(Point(3.0, 1.0), surf)
    @test !isinside(Point(1.0, -1.0), surf)
end

@testitem "isinside 2D PointCloud" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)])
    normals = [[0.0, -1.0] * m for _ = 1:4]
    areas = [0.75, 0.75, 0.75, 0.75] * m
    boundary = PointBoundary(points, normals, areas)
    cloud = PointCloud(boundary)

    @test isinside(Point(1.5, 1.5), cloud)
    @test !isinside(Point(4.0, 1.5), cloud)
    @test !isinside(Point(1.5, -1.0), cloud)
end

@testitem "isinside 2D testpoint as AbstractVector" setup = [TestData, CommonImports] begin
    points = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    normals = [[0.0, -1.0] * m for _ = 1:4]
    areas = [0.25, 0.25, 0.25, 0.25] * m
    surf = PointSurface(points, normals, areas)

    inside_point = Point(0.5, 0.5)
    outside_point = Point(1.5, 0.5)

    @test_throws MethodError isinside([0.5, 0.5], surf)
end

@testitem "isinside 3D PointBoundary" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    @test isinside(Point(0.5, 0.5, 0.5), boundary)
    @test !isinside(Point(0.5, 0.5, -0.5), boundary)
    @test !isinside(Point(0.5, 0.5, -0.001), boundary)
end

@testitem "isinside 3D PointCloud" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    cloud = PointCloud(boundary)

    @test isinside(Point(12.5, 12.5, 12.5), cloud)
    @test !isinside(Point(12.5, 12.5, -10.0), cloud)
    @test !isinside(Point(30.0, 12.5, 12.5), cloud)
end

@testitem "isinside 3D PointSurface MethodError" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    surf = boundary[:surface1]

    @test_throws MethodError isinside(Point(12.5, 12.5, 12.5), surf)
end

@testitem "isinside 3D testpoint as AbstractVector" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    surf = boundary[:surface1]

    @test_throws MethodError isinside([0.5, 0.5, 0.5], surf)
end
