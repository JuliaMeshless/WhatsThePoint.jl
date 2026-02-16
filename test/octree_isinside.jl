# Layer 4: Fast isinside queries using TriangleOctree

@testitem "TriangleOctree isinside - Basic" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = true,
    )

    # Test 1: Interior point (center of cube)
    @test isinside(SVector(0.5, 0.5, 0.5), octree) == true

    # Test 2: Exterior points
    @test isinside(SVector(-0.5, 0.5, 0.5), octree) == false
    @test isinside(SVector(1.5, 0.5, 0.5), octree) == false
    @test isinside(SVector(0.5, -0.5, 0.5), octree) == false
    @test isinside(SVector(0.5, 1.5, 0.5), octree) == false
    @test isinside(SVector(0.5, 0.5, -0.5), octree) == false
    @test isinside(SVector(0.5, 0.5, 1.5), octree) == false

    # Test 3: Points near boundary
    @test isinside(SVector(0.1, 0.5, 0.5), octree) == true
    @test isinside(SVector(0.9, 0.5, 0.5), octree) == true

    # Test 4: Corner cases
    @test isinside(SVector(0.1, 0.1, 0.1), octree) == true
    @test isinside(SVector(0.9, 0.9, 0.9), octree) == true
end

@testitem "TriangleOctree isinside - Error Handling" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()

    # Build octree WITHOUT classification
    octree_no_class = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = false,
    )

    @test isnothing(octree_no_class.leaf_classification)

    # Verify that boundary points work without classification
    boundary_point = SVector(0.05, 0.5, 0.5)
    result = isinside(boundary_point, octree_no_class)
    @test result isa Bool  # Should work without error (boundary leaf has triangles)
end

@testitem "TriangleOctree isinside - Correctness" setup = [CommonImports, OctreeTestData] begin
    using Random
    using WhatsThePoint: _get_triangle_vertices, _get_triangle_normal,
        closest_point_on_triangle

    mesh = OctreeTestData.unit_cube_mesh()

    octree = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = true,
    )

    # Brute-force O(M) signed distance for ground-truth comparison
    function isinside_bruteforce(point::SVector{3, T}, octree) where {T <: Real}
        min_dist = T(Inf)
        n = Meshes.nelements(octree.mesh)
        for i in 1:n
            v1, v2, v3 = _get_triangle_vertices(T, octree.mesh, i)
            cp = closest_point_on_triangle(point, v1, v2, v3)
            dist = norm(point - cp)
            if dist < abs(min_dist)
                to_point = point - cp
                normal_i = _get_triangle_normal(T, octree.mesh, i)
                sign = dot(to_point, normal_i) >= 0 ? 1 : -1
                min_dist = sign * dist
            end
        end
        return min_dist < 0
    end

    # Test random points
    Random.seed!(42)
    n_test = 100

    for _ in 1:n_test
        # Random point in [-0.5, 1.5]³
        point = SVector(rand(3)...) .* 2.0 .- 0.5

        octree_result = isinside(point, octree)
        brute_result = isinside_bruteforce(point, octree)

        @test octree_result == brute_result
    end
end

@testitem "TriangleOctree isinside - Real STL" setup = [CommonImports, TestData] begin
    using Random

    # Load box.stl (46,786 triangles)
    if isfile(TestData.BOX_PATH)
        # Use TriangleOctree file path constructor
        octree = TriangleOctree(
            TestData.BOX_PATH;
            h_min = 0.1,
            max_triangles_per_box = 50,
            classify_leaves = true,
        )

        # Compute bounding box from mesh
        bbox_min, bbox_max = WhatsThePoint._compute_bbox(Float64, octree.mesh)

        # Test 1: Point at mesh center should be interior
        center = SVector{3}((bbox_min + bbox_max) / 2)
        @test isinside(center, octree) == true

        # Test 2: Point outside bounding box should be exterior
        outside = SVector{3}(bbox_max + SVector(1.0, 1.0, 1.0))
        @test isinside(outside, octree) == false

        # Test 3: Random points within bounding box
        Random.seed!(123)
        for _ in 1:20
            # Random point in bounding box
            t = rand(3)
            point = SVector{3}(bbox_min .+ t .* (bbox_max - bbox_min))

            # Just verify no errors (correctness hard to verify without ground truth)
            result = isinside(point, octree)
            @test result isa Bool
        end
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree isinside - Batch Queries" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()

    octree = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = true,
    )

    # Create test points
    test_points = [
        SVector(0.5, 0.5, 0.5),   # Interior
        SVector(1.5, 0.5, 0.5),   # Exterior
        SVector(0.1, 0.1, 0.1),   # Interior near corner
        SVector(-0.1, 0.5, 0.5),  # Exterior
    ]

    # Batch query
    results = isinside(test_points, octree)

    @test results isa Vector{Bool}
    @test length(results) == 4
    @test results[1] == true   # Interior
    @test results[2] == false  # Exterior
    @test results[3] == true   # Interior
    @test results[4] == false  # Exterior

    # Verify matches individual queries
    for (i, point) in enumerate(test_points)
        @test results[i] == isinside(point, octree)
    end
end

@testitem "TriangleOctree isinside - Edge Cases" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()

    octree = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = true,
    )

    # Test 1: Points exactly on boundary (may be inside or outside depending on tolerance)
    # We test that the query doesn't crash
    boundary_points = [
        SVector(0.0, 0.5, 0.5),  # On left face
        SVector(1.0, 0.5, 0.5),  # On right face
        SVector(0.5, 0.0, 0.5),  # On front face
        SVector(0.5, 1.0, 0.5),  # On back face
        SVector(0.5, 0.5, 0.0),  # On bottom face
        SVector(0.5, 0.5, 1.0),  # On top face
    ]

    for point in boundary_points
        result = isinside(point, octree)
        @test result isa Bool  # Should return without error
    end

    # Test 2: Points at corners
    corners = [
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(1.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0),
        SVector(1.0, 0.0, 1.0),
        SVector(0.0, 1.0, 1.0),
        SVector(1.0, 1.0, 1.0),
    ]

    for corner in corners
        result = isinside(corner, octree)
        @test result isa Bool  # Should return without error
    end
end

@testitem "TriangleOctree isinside - Point bridge" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()

    octree = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = true,
    )

    # Single Point should match SVector result
    @test isinside(Point(0.5, 0.5, 0.5), octree) == isinside(SVector(0.5, 0.5, 0.5), octree)
    @test isinside(Point(0.5, 0.5, 0.5), octree) == true
    @test isinside(Point(5.0, 5.0, 5.0), octree) == false

    # Batch Point query
    pts = [Point(0.5, 0.5, 0.5), Point(5.0, 5.0, 5.0), Point(0.1, 0.1, 0.1)]
    results = isinside(pts, octree)
    @test results == [true, false, true]
end
