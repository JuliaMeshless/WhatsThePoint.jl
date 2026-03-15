# Tests for TriangleOctree construction

@testitem "TriangleOctree basic construction" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.simple_square_mesh()
    octree = TriangleOctree(mesh)

    @test octree isa TriangleOctree
    @test octree.mesh === mesh
    @test num_triangles(octree) == 2
    @test num_leaves(octree) > 0
end

@testitem "TriangleOctree classification" setup = [CommonImports, OctreeTestData] begin
    using WhatsThePoint: all_leaves

    mesh = OctreeTestData.unit_cube_mesh()

    # Without classification
    octree1 = TriangleOctree(mesh; classify_leaves = false)
    @test octree1.leaf_classification === nothing

    # With classification
    octree2 = TriangleOctree(mesh; classify_leaves = true)
    @test octree2.leaf_classification !== nothing

    # Classification values are valid (0=exterior, 1=boundary, 2=interior)
    for leaf_idx in all_leaves(octree2.tree)
        @test octree2.leaf_classification[leaf_idx] in (0, 1, 2)
    end
end

@testitem "TriangleOctree orientation verification" setup = [CommonImports] begin
    # Mesh with inconsistent normals (flipped faces)
    pts = [
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0),
    ]
    connec = [
        connect((1, 2, 3), Meshes.Triangle),  # CCW: +z normal
        connect((1, 4, 3), Meshes.Triangle),   # CW: -z normal (flipped)
    ]
    mesh = SimpleMesh(pts, connec)

    # Should error by default
    @test_throws ArgumentError TriangleOctree(mesh)

    # Can disable check
    octree = TriangleOctree(mesh; verify_orientation = false)
    @test octree isa TriangleOctree
    @test !has_consistent_normals(mesh)
end

@testitem "TriangleOctree from file" setup = [CommonImports, TestData] begin
    if isfile(TestData.BOX_PATH)
        octree = TriangleOctree(TestData.BOX_PATH)
        @test octree isa TriangleOctree
        @test num_triangles(octree) == 46786
    else
        @test_skip "box.stl not available"
    end
end
