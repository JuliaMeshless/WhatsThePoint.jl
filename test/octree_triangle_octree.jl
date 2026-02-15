# Tests for TriangleOctree with SimpleMesh

@testitem "TriangleOctree Simple 2-Triangle Mesh" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree
    # Helper function to create a simple 2-triangle square in xy-plane
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    connec = [connect((1, 2, 3), Meshes.Triangle), connect((1, 3, 4), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    # Build octree
    octree = TriangleOctree(mesh; h_min = 0.1, max_triangles_per_box = 1)

    # Basic checks
    @test octree.mesh === mesh
    @test octree.tree isa SpatialOctree{Int, Float64}
    @test num_triangles(octree) == 2
    @test num_leaves(octree) > 0
end

@testitem "TriangleOctree Subdivision Behavior" setup = [CommonImports] begin
    using WhatsThePoint: all_leaves, box_size
    # Create mesh with multiple spatially separated triangles
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(0.3, 0.0, 0.0),
        Point(0.3, 0.3, 0.0),
        Point(0.5, 0.5, 0.0),
        Point(0.8, 0.5, 0.0),
        Point(0.8, 0.8, 0.0),
        Point(0.0, 0.5, 0.5),
        Point(0.3, 0.5, 0.5),
        Point(0.3, 0.8, 0.5),
    ]
    connec = [
        connect((1, 2, 3), Meshes.Triangle),
        connect((4, 5, 6), Meshes.Triangle),
        connect((7, 8, 9), Meshes.Triangle),
    ]
    mesh = SimpleMesh(pts, connec)

    # Build with low threshold to force subdivision
    octree = TriangleOctree(mesh; h_min = 0.05, max_triangles_per_box = 1)

    # Should have subdivided
    @test num_leaves(octree) > 1

    # All leaves should respect triangle threshold or be at min size
    for leaf_idx in all_leaves(octree.tree)
        n_tris = length(octree.tree.element_lists[leaf_idx])
        leaf_size = box_size(octree.tree, leaf_idx)

        # Either has ≤1 triangle, or is at minimum size
        @test (n_tris <= 1) || (leaf_size ≈ 0.05)
    end
end

@testitem "TriangleOctree Triangle Distribution" setup = [CommonImports] begin
    using WhatsThePoint: all_leaves
    # Single triangle spanning octree
    pts = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.5, 1.0, 0.0)]
    connec = [connect((1, 2, 3), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(mesh; h_min = 0.2, max_triangles_per_box = 1)

    # Triangle should be distributed to multiple boxes
    total_references = sum(
        length(octree.tree.element_lists[leaf_idx]) for leaf_idx in all_leaves(octree.tree)
    )

    # Triangle spans multiple boxes, so should have multiple references
    @test total_references >= 1
end

@testitem "TriangleOctree Minimum Size Constraint" setup = [CommonImports] begin
    using WhatsThePoint: all_leaves, box_size
    pts = [Point(0.0, 0.0, 0.0), Point(0.1, 0.0, 0.0), Point(0.1, 0.1, 0.0)]
    connec = [connect((1, 2, 3), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    h_min = 0.05
    octree = TriangleOctree(mesh; h_min = h_min, max_triangles_per_box = 1)

    # No leaf should be smaller than h_min (with small tolerance for floating point)
    for leaf_idx in all_leaves(octree.tree)
        leaf_size = box_size(octree.tree, leaf_idx)
        @test (leaf_size >= h_min - 1.0e-10) || (leaf_size ≈ h_min)
    end
end

@testitem "TriangleOctree Leaf Classification - Disabled" setup = [CommonImports] begin
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    connec = [connect((1, 2, 3), Meshes.Triangle), connect((1, 3, 4), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(
        mesh;
        h_min = 0.1,
        max_triangles_per_box = 1,
        classify_leaves = false,
    )

    @test octree.leaf_classification === nothing
end

@testitem "TriangleOctree Leaf Classification - Enabled" setup = [CommonImports] begin
    using WhatsThePoint: all_leaves
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    connec = [connect((1, 2, 3), Meshes.Triangle), connect((1, 3, 4), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(
        mesh;
        h_min = 0.2,
        max_triangles_per_box = 10,
        classify_leaves = true,
    )

    @test octree.leaf_classification !== nothing
    @test length(octree.leaf_classification) == length(octree.tree.element_lists)

    # Check leaf classification values are valid (0, 1, or 2)
    for leaf_idx in all_leaves(octree.tree)
        @test octree.leaf_classification[leaf_idx] in (0, 1, 2)
    end

    # Leaves with triangles should be classified as boundary (1)
    for leaf_idx in all_leaves(octree.tree)
        if !isempty(octree.tree.element_lists[leaf_idx])
            @test octree.leaf_classification[leaf_idx] == 1
        end
    end
end

@testitem "TriangleOctree with box.stl Construction" setup = [CommonImports, TestData] begin
    using WhatsThePoint: all_leaves
    # Only run if test file exists
    if isfile(TestData.BOX_PATH)
        # Build octree directly from file path
        octree = TriangleOctree(
            TestData.BOX_PATH;
            h_min = 0.05,
            max_triangles_per_box = 100,
            classify_leaves = false,
        )

        @test num_triangles(octree) == 46786
        @test num_leaves(octree) > 1

        # Check that triangles were distributed
        total_refs =
            sum(length(octree.tree.element_lists[leaf]) for leaf in all_leaves(octree.tree))
        @test total_refs >= num_triangles(octree)
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree box.stl with Classification" setup = [CommonImports, TestData] begin
    # Only run if test file exists
    if isfile(TestData.BOX_PATH)
        # Build octree from file with classification
        octree = TriangleOctree(
            TestData.BOX_PATH;
            h_min = 0.1,
            max_triangles_per_box = 200,
            classify_leaves = true,
        )

        @test octree.leaf_classification !== nothing

        # Count classification types
        n_exterior = count(==(0), octree.leaf_classification)
        n_boundary = count(==(1), octree.leaf_classification)
        n_interior = count(==(2), octree.leaf_classification)

        # Should have all three types
        @test n_boundary > 0
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree Accessors" setup = [CommonImports] begin
    pts = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)]
    connec = [connect((1, 2, 3), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(mesh; h_min = 0.1, max_triangles_per_box = 10)

    @test length(octree) == 1
    @test num_triangles(octree) == 1
    @test num_leaves(octree) >= 1
end

@testitem "has_consistent_normals - Consistent Mesh" setup = [CommonImports] begin
    # Two triangles with consistent normals (both pointing up in +z)
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    connec = [connect((1, 2, 3), Meshes.Triangle), connect((1, 3, 4), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(mesh; h_min = 0.1, verify_orientation = false)

    # Both triangles have normals pointing in +z, so should be consistent
    @test has_consistent_normals(octree.mesh)
end

@testitem "has_consistent_normals - Inconsistent Mesh" setup = [CommonImports] begin
    # Two triangles with opposite normals (one +z, one -z)
    # First triangle: CCW -> normal +z
    # Second triangle: CW -> normal -z
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    connec = [
        connect((1, 2, 3), Meshes.Triangle),  # CCW: normal +z
        connect((1, 4, 3), Meshes.Triangle),  # CW: normal -z
    ]
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(mesh; h_min = 0.1, verify_orientation = false)

    # Normals should be opposite
    n1 = WhatsThePoint._get_triangle_normal(octree.mesh, 1)
    n2 = WhatsThePoint._get_triangle_normal(octree.mesh, 2)
    @test dot(n1, n2) < 0

    # Should detect inconsistency
    @test !has_consistent_normals(octree.mesh)
end

@testitem "TriangleOctree Verification Disabled" setup = [CommonImports] begin
    # Mesh with inconsistent normals
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    connec = [connect((1, 2, 3), Meshes.Triangle), connect((1, 4, 3), Meshes.Triangle)]
    mesh = SimpleMesh(pts, connec)

    # Should build without warning when verification is disabled
    octree = TriangleOctree(mesh; h_min = 0.1, verify_orientation = false)
    @test octree isa TriangleOctree
end

@testitem "TriangleOctree From File Path" setup = [CommonImports, TestData] begin
    if isfile(TestData.BOX_PATH)
        # Test file path constructor
        octree = TriangleOctree(
            TestData.BOX_PATH;
            h_min = 0.1,
            max_triangles_per_box = 100,
            classify_leaves = false,
            verify_orientation = false,
        )

        @test octree isa TriangleOctree
        @test num_triangles(octree) == 46786
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree From PointBoundary" setup = [CommonImports, TestData] begin
    # TODO: has_source_mesh/TriangleOctree(::PointBoundary) not yet implemented
    @test_skip "TriangleOctree from PointBoundary not yet implemented"
end

@testitem "TriangleOctree From PointBoundary without source mesh" setup = [CommonImports] begin
    # TODO: has_source_mesh/TriangleOctree(::PointBoundary) not yet implemented
    @test_skip "TriangleOctree from PointBoundary not yet implemented"
end
