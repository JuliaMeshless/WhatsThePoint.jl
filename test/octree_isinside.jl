# Layer 4: Fast isinside queries using TriangleOctree

@testitem "TriangleOctree isinside - Basic" setup = [CommonImports] begin
    # Create unit cube mesh
    pts = [
        Point(0.0, 0.0, 0.0),  # 1
        Point(1.0, 0.0, 0.0),  # 2
        Point(1.0, 1.0, 0.0),  # 3
        Point(0.0, 1.0, 0.0),  # 4
        Point(0.0, 0.0, 1.0),  # 5
        Point(1.0, 0.0, 1.0),  # 6
        Point(1.0, 1.0, 1.0),  # 7
        Point(0.0, 1.0, 1.0),  # 8
    ]

    # Define triangles (2 per face, counterclockwise when viewed from outside)
    connec = [
        # Bottom face (z=0)
        connect((1, 3, 2), Meshes.Triangle),
        connect((1, 4, 3), Meshes.Triangle),
        # Top face (z=1)
        connect((5, 6, 7), Meshes.Triangle),
        connect((5, 7, 8), Meshes.Triangle),
        # Front face (y=0)
        connect((1, 2, 6), Meshes.Triangle),
        connect((1, 6, 5), Meshes.Triangle),
        # Back face (y=1)
        connect((3, 4, 8), Meshes.Triangle),
        connect((3, 8, 7), Meshes.Triangle),
        # Left face (x=0)
        connect((1, 5, 8), Meshes.Triangle),
        connect((1, 8, 4), Meshes.Triangle),
        # Right face (x=1)
        connect((2, 3, 7), Meshes.Triangle),
        connect((2, 7, 6), Meshes.Triangle),
    ]

    mesh = SimpleMesh(pts, connec)
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

@testitem "TriangleOctree isinside - Error Handling" setup = [CommonImports] begin
    # Create unit cube mesh
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0),
        Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0),
        Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle),
        connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle),
        connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle),
        connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle),
        connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle),
        connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle),
        connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

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

@testitem "TriangleOctree isinside - Correctness" setup = [CommonImports] begin
    using Random

    # Create unit cube mesh
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0),
        Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0),
        Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle),
        connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle),
        connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle),
        connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle),
        connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle),
        connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle),
        connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(
        mesh;
        h_min = 0.05,
        max_triangles_per_box = 5,
        classify_leaves = true,
    )

    function isinside_bruteforce(point::SVector{3, T}, octree) where {T <: Real}
        dist = WhatsThePoint._compute_signed_distance(point, octree.triangles)
        return dist < 0
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
    using GeoIO

    # Load box.stl (46,786 triangles)
    if isfile(TestData.BOX_PATH)
        # Use TriangleOctree file path constructor
        octree = TriangleOctree(
            TestData.BOX_PATH;
            h_min = 0.1,
            max_triangles_per_box = 50,
            classify_leaves = true,
        )

        # Compute bounding box from cached triangles
        bbox_min, bbox_max = WhatsThePoint._compute_bbox(octree.triangles)

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

@testitem "TriangleOctree isinside - Batch Queries" setup = [CommonImports] begin
    # Create unit cube mesh
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0),
        Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0),
        Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle),
        connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle),
        connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle),
        connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle),
        connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle),
        connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle),
        connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

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

@testitem "TriangleOctree isinside - Edge Cases" setup = [CommonImports] begin
    # Create unit cube mesh
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0),
        Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0),
        Point(0.0, 1.0, 1.0),
    ]
    connec = [
        connect((1, 3, 2), Meshes.Triangle),
        connect((1, 4, 3), Meshes.Triangle),
        connect((5, 6, 7), Meshes.Triangle),
        connect((5, 7, 8), Meshes.Triangle),
        connect((1, 2, 6), Meshes.Triangle),
        connect((1, 6, 5), Meshes.Triangle),
        connect((3, 4, 8), Meshes.Triangle),
        connect((3, 8, 7), Meshes.Triangle),
        connect((1, 5, 8), Meshes.Triangle),
        connect((1, 8, 4), Meshes.Triangle),
        connect((2, 3, 7), Meshes.Triangle),
        connect((2, 7, 6), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

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
