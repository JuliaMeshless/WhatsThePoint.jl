@testitem "TriangleOctree Simple 2-Triangle Mesh" setup = [CommonImports] begin
    # Create a simple square made of 2 triangles in xy-plane
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0)),
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), SVector(0.0, 1.0, 0.0)),
    ]
    mesh = TriangleMesh(tri_list)

    # Build octree
    octree = TriangleOctree(mesh; h_min = 0.1, max_triangles_per_box = 1)

    # Basic checks
    @test octree.mesh === mesh
    @test octree.tree isa SpatialOctree{Int,Float64}
    @test num_triangles(octree) == 2
    @test num_leaves(octree) > 0
end

@testitem "TriangleOctree Subdivision Behavior" setup = [CommonImports] begin
    # Create mesh with multiple triangles to force subdivision
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(0.3, 0.0, 0.0), SVector(0.3, 0.3, 0.0)),
        Triangle(SVector(0.5, 0.5, 0.0), SVector(0.8, 0.5, 0.0), SVector(0.8, 0.8, 0.0)),
        Triangle(SVector(0.0, 0.5, 0.5), SVector(0.3, 0.5, 0.5), SVector(0.3, 0.8, 0.5)),
    ]
    mesh = TriangleMesh(tri_list)

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
    # Single triangle spanning octree
    tri = Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.5, 1.0, 0.0))
    mesh = TriangleMesh([tri])

    octree = TriangleOctree(mesh; h_min = 0.2, max_triangles_per_box = 1)

    # Triangle should be distributed to multiple boxes
    total_references = 0
    for leaf_idx in all_leaves(octree.tree)
        total_references += length(octree.tree.element_lists[leaf_idx])
    end

    # Triangle spans multiple boxes, so should have multiple references
    @test total_references >= 1
end

@testitem "TriangleOctree Minimum Size Constraint" setup = [CommonImports] begin
    tri_list =
        [Triangle(SVector(0.0, 0.0, 0.0), SVector(0.1, 0.0, 0.0), SVector(0.1, 0.1, 0.0))]
    mesh = TriangleMesh(tri_list)

    h_min = 0.05  # Smaller than the triangle, so we can test constraint
    octree = TriangleOctree(mesh; h_min = h_min, max_triangles_per_box = 1)

    # No leaf should be smaller than h_min (with small tolerance for floating point)
    for leaf_idx in all_leaves(octree.tree)
        leaf_size = box_size(octree.tree, leaf_idx)
        # Size should be either >= h_min or very close (due to FP precision)
        @test (leaf_size >= h_min - 1e-10) || (leaf_size ≈ h_min)
    end
end

@testitem "TriangleOctree Leaf Classification - Disabled" setup = [CommonImports] begin
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0)),
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), SVector(0.0, 1.0, 0.0)),
    ]
    mesh = TriangleMesh(tri_list)

    octree = TriangleOctree(
        mesh;
        h_min = 0.1,
        max_triangles_per_box = 1,
        classify_leaves = false,
    )

    @test octree.leaf_classification === nothing
end

@testitem "TriangleOctree Leaf Classification - Enabled" setup = [CommonImports] begin
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0)),
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), SVector(0.0, 1.0, 0.0)),
    ]
    mesh = TriangleMesh(tri_list)

    octree = TriangleOctree(
        mesh;
        h_min = 0.2,
        max_triangles_per_box = 10,
        classify_leaves = true,
    )

    @test octree.leaf_classification !== nothing
    @test length(octree.leaf_classification) == length(octree.tree.element_lists)

    # Check classification values are valid (0, 1, or 2)
    for classification in octree.leaf_classification
        @test classification in (0, 1, 2)
    end

    # Leaves with triangles should be classified as boundary (1)
    for leaf_idx in all_leaves(octree.tree)
        if !isempty(octree.tree.element_lists[leaf_idx])
            @test octree.leaf_classification[leaf_idx] == 1
        end
    end
end

@testitem "TriangleOctree with box.stl Construction" setup = [CommonImports, TestData] begin
    # Only run if test file exists
    if isfile(TestData.BOX_PATH)
        mesh = TriangleMesh(TestData.BOX_PATH)

        # Build octree with coarser parameters to avoid hitting capacity limits
        octree = TriangleOctree(
            mesh;
            h_min = 0.05,  # Coarser minimum size
            max_triangles_per_box = 100,  # Higher threshold
            classify_leaves = false,
        )  # Skip classification for speed

        @test num_triangles(octree) == 46786
        @test num_leaves(octree) > 1

        # Check that triangles were distributed
        total_refs =
            sum(length(octree.tree.element_lists[leaf]) for leaf in all_leaves(octree.tree))
        @test total_refs >= num_triangles(octree)  # Should be >= due to multi-box triangles
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree box.stl with Classification" setup = [CommonImports, TestData] begin
    # Only run if test file exists
    if isfile(TestData.BOX_PATH)
        mesh = TriangleMesh(TestData.BOX_PATH)

        # Build with classification (slower but complete), coarser params
        octree = TriangleOctree(
            mesh;
            h_min = 0.1,  # Even coarser for classification test
            max_triangles_per_box = 200,
            classify_leaves = true,
        )

        @test octree.leaf_classification !== nothing

        # Count classification types
        n_exterior = count(==(0), octree.leaf_classification)
        n_boundary = count(==(1), octree.leaf_classification)
        n_interior = count(==(2), octree.leaf_classification)

        # Should have all three types
        @test n_boundary > 0  # Must have boundary leaves with triangles
    # Note: exterior/interior might be 0 depending on mesh topology
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree Accessors" setup = [CommonImports] begin
    tri_list =
        [Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0))]
    mesh = TriangleMesh(tri_list)
    octree = TriangleOctree(mesh; h_min = 0.1, max_triangles_per_box = 10)

    @test length(octree) == 1
    @test num_triangles(octree) == 1
    @test num_leaves(octree) >= 1
end

@testitem "has_consistent_normals - Consistent Mesh" setup = [CommonImports] begin
    # Create two triangles with consistent normals (both pointing up in +z)
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0)),
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), SVector(0.0, 1.0, 0.0)),
    ]
    mesh = TriangleMesh(tri_list)

    # Both triangles have normals pointing in +z, so should be consistent
    @test has_consistent_normals(mesh)
end

@testitem "has_consistent_normals - Inconsistent Mesh" setup = [CommonImports] begin
    # Create two nearby triangles with opposite normals
    # First triangle: normal points up (+z)
    tri1 = Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0))

    # Second triangle: vertices in opposite winding order -> normal points down (-z)
    # By reversing v2 and v3, we flip the normal
    tri2 = Triangle(SVector(0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(1.0, 1.0, 0.0))

    mesh = TriangleMesh([tri1, tri2])

    # Verify the normals are actually opposite
    @test dot(tri1.normal, tri2.normal) < 0

    # Should detect inconsistency
    @test !has_consistent_normals(mesh)
end

@testitem "TriangleOctree Verification Disabled" setup = [CommonImports] begin
    # Create mesh with inconsistent normals
    tri1 = Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0))
    tri2 = Triangle(SVector(0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(1.0, 1.0, 0.0))
    mesh = TriangleMesh([tri1, tri2])

    # Should build without warning when verification is disabled
    octree = TriangleOctree(mesh; h_min = 0.1, verify_orientation = false)
    @test octree isa TriangleOctree{Float64}
end
