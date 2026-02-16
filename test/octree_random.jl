# Tests for OctreeRandom discretization algorithm and helpers

@testitem "OctreeRandom constructors" setup = [CommonImports] begin
    pts = [
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0), Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0), Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle), connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle), connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle), connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle), connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle), connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle), connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)
    octree = TriangleOctree(mesh; h_min=0.05, max_triangles_per_box=5, classify_leaves=true)

    # Default oversampling = 2.0
    alg1 = OctreeRandom(octree)
    @test alg1 isa OctreeRandom
    @test alg1.boundary_oversampling == 2.0
    @test alg1.octree === octree

    # Custom oversampling
    alg2 = OctreeRandom(octree, 3.5)
    @test alg2.boundary_oversampling == 3.5

    # Integer oversampling gets converted to Float64
    alg3 = OctreeRandom(octree, 4)
    @test alg3.boundary_oversampling === 4.0

    # Type parameters propagate from octree
    @test alg1 isa OctreeRandom{Meshes.𝔼{3}}
end

@testitem "_allocate_counts_by_volume — basic" setup = [CommonImports] begin
    using WhatsThePoint: _allocate_counts_by_volume

    # Empty volumes → empty result
    result = _allocate_counts_by_volume(Float64[], 100)
    @test result == Int[]

    # Zero total_count → all zeros
    result = _allocate_counts_by_volume([1.0, 2.0, 3.0], 0)
    @test result == [0, 0, 0]

    # Proportional allocation: volumes [3.0, 1.0] with total 100
    # Expected: 75 for first, 25 for second
    result = _allocate_counts_by_volume([3.0, 1.0], 100)
    @test result == [75, 25]
    @test sum(result) == 100

    # Sum always preserved exactly for various inputs
    for total in [1, 7, 13, 99, 1000]
        vols = [1.5, 2.3, 0.7, 3.1]
        r = _allocate_counts_by_volume(vols, total)
        @test sum(r) == total
    end

    # ensure_one: every leaf gets at least 1 when total_count >= n
    result = _allocate_counts_by_volume([10.0, 0.001, 0.001], 100; ensure_one=true)
    @test all(result .>= 1)
    @test sum(result) == 100

    # Single element → gets all points
    result = _allocate_counts_by_volume([5.0], 42)
    @test result == [42]

    # Equal volumes → equal split
    result = _allocate_counts_by_volume([1.0, 1.0, 1.0, 1.0], 100)
    @test result == [25, 25, 25, 25]
    @test sum(result) == 100

    # Equal volumes, non-divisible → still sums correctly
    result = _allocate_counts_by_volume([1.0, 1.0, 1.0], 10)
    @test sum(result) == 10
    # Each gets either 3 or 4
    @test all(3 .<= result .<= 4)
end

@testitem "_allocate_counts_by_volume — edge cases" setup = [CommonImports] begin
    using WhatsThePoint: _allocate_counts_by_volume

    # All zero volumes → uniform distribution
    result = _allocate_counts_by_volume([0.0, 0.0, 0.0], 9)
    @test sum(result) == 9
    @test result == [3, 3, 3]

    # ensure_one with total_count < n → can't guarantee one per leaf
    result = _allocate_counts_by_volume([1.0, 1.0, 1.0, 1.0, 1.0], 3; ensure_one=true)
    @test sum(result) == 3

    # Large count with many leaves → sum still exact
    n_leaves = 500
    vols = rand(n_leaves) .+ 0.01
    total = 100_000
    result = _allocate_counts_by_volume(vols, total)
    @test sum(result) == total
    @test length(result) == n_leaves
    @test all(result .>= 0)

    # Negative total → zeros
    result = _allocate_counts_by_volume([1.0, 2.0], -5)
    @test result == [0, 0]

    # Very skewed volumes
    result = _allocate_counts_by_volume([1000.0, 0.001], 100)
    @test sum(result) == 100
    @test result[1] > result[2]
end

@testitem "discretize with OctreeRandom (unit cube)" setup = [CommonImports] begin
    using Random
    Random.seed!(42)

    pts = [
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0), Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0), Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle), connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle), connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle), connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle), connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle), connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle), connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; h_min=0.05, max_triangles_per_box=5, classify_leaves=true)

    cloud = discretize(bnd, ConstantSpacing(1.0m); alg=OctreeRandom(octree), max_points=200)

    @test cloud isa PointCloud
    vol = WhatsThePoint.volume(cloud)
    @test length(vol) > 0
    @test length(vol) <= 200

    # All volume points should be inside the unit cube
    for pt in vol
        c = to(pt)
        x, y, z = c[1] / m, c[2] / m, c[3] / m
        @test -0.01 <= x <= 1.01
        @test -0.01 <= y <= 1.01
        @test -0.01 <= z <= 1.01
    end

    # Verify all points pass isinside check
    for pt in vol
        c = to(pt)
        sv = SVector(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end
end

@testitem "discretize with OctreeRandom — error on unclassified octree" setup = [CommonImports] begin
    pts = [
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0), Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0), Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle), connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle), connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle), connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle), connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle), connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle), connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

    bnd = PointBoundary(mesh)
    octree_no_class = TriangleOctree(mesh; h_min=0.05, max_triangles_per_box=5, classify_leaves=false)

    @test_throws ErrorException discretize(
        bnd, ConstantSpacing(1.0m);
        alg=OctreeRandom(octree_no_class), max_points=50,
    )
end

@testitem "discretize with OctreeRandom (box.stl)" setup = [CommonImports, TestData] begin
    using Random
    Random.seed!(123)

    if isfile(TestData.BOX_PATH)
        bnd = PointBoundary(TestData.BOX_PATH)
        octree = TriangleOctree(
            TestData.BOX_PATH;
            h_min=0.1,
            max_triangles_per_box=50,
            classify_leaves=true,
        )

        cloud = discretize(bnd, ConstantSpacing(1.0m); alg=OctreeRandom(octree), max_points=500)

        @test cloud isa PointCloud
        vol = WhatsThePoint.volume(cloud)
        @test length(vol) > 0
        @test length(vol) <= 500

        # All generated points should be inside the mesh
        for pt in vol
            c = to(pt)
            sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
            @test isinside(sv, octree) == true
        end
    else
        @test_skip "box.stl not available"
    end
end

@testitem "OctreeRandom boundary_oversampling effect" setup = [CommonImports] begin
    using Random

    pts = [
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0), Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0), Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle), connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle), connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle), connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle), connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle), connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle), connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)
    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; h_min=0.05, max_triangles_per_box=5, classify_leaves=true)

    max_pts = 200

    # Low oversampling
    Random.seed!(99)
    cloud_low = discretize(bnd, ConstantSpacing(1.0m); alg=OctreeRandom(octree, 1.0), max_points=max_pts)
    n_low = length(WhatsThePoint.volume(cloud_low))

    # High oversampling
    Random.seed!(99)
    cloud_high = discretize(bnd, ConstantSpacing(1.0m); alg=OctreeRandom(octree, 3.0), max_points=max_pts)
    n_high = length(WhatsThePoint.volume(cloud_high))

    # Both should produce valid interior points
    @test n_low > 0
    @test n_high > 0
    @test n_low <= max_pts
    @test n_high <= max_pts

    # Higher oversampling should yield count closer to max_points (or equal)
    @test n_high >= n_low
end
