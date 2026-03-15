# Tests for AdaptiveOctree discretization algorithm

@testitem "AdaptiveOctree with ConstantSpacing" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    alg = AdaptiveOctree(mesh)
    cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 100)

    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "AdaptiveOctree with BoundaryLayerSpacing" setup = [CommonImports, OctreeTestData] begin
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

    alg = AdaptiveOctree(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)

    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "AdaptiveOctree points are inside" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(123)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(1.0m)

    alg = AdaptiveOctree(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)

    # All points should be inside via octree check
    octree = alg.triangle_octree
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end
end

@testitem "AdaptiveOctree with placement strategies" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(456)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    for placement in (:random, :jittered, :lattice)
        alg = AdaptiveOctree(mesh; placement)
        cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 50)

        @test cloud isa PointCloud
        @test length(WhatsThePoint.volume(cloud)) > 0
        @test length(WhatsThePoint.volume(cloud)) <= 50
    end
end

@testitem "AdaptiveOctree errors on unclassified octree" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; classify_leaves = false)

    @test_throws ErrorException discretize(
        bnd, ConstantSpacing(1m);
        alg = AdaptiveOctree(octree), max_points = 50
    )
end

@testitem "AdaptiveOctree invalid placement throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError AdaptiveOctree(octree; placement = :invalid)
end

@testitem "AdaptiveOctree invalid oversampling throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError AdaptiveOctree(octree; boundary_oversampling = -1.0)
end
