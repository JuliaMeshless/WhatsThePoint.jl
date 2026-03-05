# Tests for OctreeRandom discretization algorithm

@testitem "OctreeRandom generates points" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    cloud = discretize(bnd, ConstantSpacing(1m); alg=OctreeRandom(mesh), max_points=100)

    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "OctreeRandom points are inside" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    alg = OctreeRandom(mesh)

    cloud = discretize(bnd, ConstantSpacing(1m); alg=alg, max_points=100)

    # Verify all points pass isinside check
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, alg.octree) == true
    end
end

@testitem "OctreeRandom errors on unclassified octree" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; classify_leaves=false)

    @test_throws ErrorException discretize(
        bnd, ConstantSpacing(1m);
        alg=OctreeRandom(octree), max_points=50
    )
end
