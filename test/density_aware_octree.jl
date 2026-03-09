# Tests for DensityAwareOctree discretization algorithm

@testitem "BoundaryLayerSpacing basic evaluation" setup = [CommonImports] begin
    points = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)]
    spacing = BoundaryLayerSpacing(points; at_wall = 1.0m, bulk = 10.0m, layer_thickness = 5.0m)

    # At boundary: close to at_wall
    @test spacing(Point(0.0, 0.0, 0.0)) ≈ 1.0m atol = 0.5m

    # Far from boundary: close to bulk
    @test spacing(Point(100.0, 100.0, 100.0)) ≈ 10.0m atol = 0.5m
end

@testitem "DensityAwareOctree generates points" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = 2.0m
    )

    alg = DensityAwareOctree(mesh)
    cloud = discretize(bnd, spacing; alg = alg, max_points = 100)

    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "DensityAwareOctree points are inside" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(123)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(1.0m)

    alg = DensityAwareOctree(mesh)
    cloud = discretize(bnd, spacing; alg = alg, max_points = 100)

    # All points should be inside via octree check
    octree = alg.triangle_octree
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end
end
