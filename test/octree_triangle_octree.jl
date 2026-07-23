# Tests for TriangleOctree construction

@testitem "TriangleOctree basic construction" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.simple_square_mesh()
    octree = TriangleOctree(mesh)

    @test octree isa TriangleOctree
    @test num_triangles(octree.index) == 2
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
        octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"))
        @test octree isa TriangleOctree
        @test num_triangles(octree) == 46786
    else
        @test_skip "box.stl not available"
    end
end

@testitem "TriangleOctree length" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh)
    @test length(octree) == 12
end

@testitem "isinside with Meshes.jl Point types" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    # Single Point
    @test isinside(Point(0.5, 0.5, 0.5), octree) == true
    @test isinside(Point(-0.5, 0.5, 0.5), octree) == false

    # Vector of Points
    results = isinside([Point(0.5, 0.5, 0.5), Point(-0.5, 0.5, 0.5)], octree)
    @test results == [true, false]
end

@testitem "TriangleOctree balancing redistributes triangles" setup = [CommonImports] begin
    using WhatsThePoint: all_leaves, NearestTriangleState, _nearest_triangle_octree!

    # Regression: `balance_octree!` subdivides leaves without moving their
    # element lists to the children — triangles of a balance-subdivided leaf
    # vanish from the queryable tree, and nearest-triangle search then returns
    # far-away triangles. Exposed on multi-density geometry (coarse wall +
    # fine patch), where 2:1 balancing subdivides occupied wall leaves.
    #
    # Mesh: one NARROW tall sliver (5mm × 2m, 2 triangles, x=0 plane — the
    # kind CAD exports produce on vessel walls) flanked by a dense patch of
    # tiny triangles 1cm away spanning its full height. The sliver's mid-span
    # leaves stay coarse (no local vertices) while the patch goes deep, so
    # 2:1 balancing subdivides them — pre-fix the sliver was orphaned out of
    # the queryable tree entirely.
    pts = [Point(0.0, -0.0025, 0.0), Point(0.0, 0.0025, 0.0), Point(0.0, 0.0025, 2.0), Point(0.0, -0.0025, 2.0)]
    connec = [connect((1, 2, 3), Meshes.Triangle), connect((1, 3, 4), Meshes.Triangle)]
    patch_d = 0.002
    ny, nz = 30, 1000
    grid = [
        Point(0.01, -0.03 + iy * patch_d, iz * patch_d)
            for iz in 0:nz for iy in 0:ny
    ]
    offset = length(pts)
    append!(pts, grid)
    v(iy, iz) = offset + iz * (ny + 1) + iy + 1
    for iz in 1:nz, iy in 1:ny
        a, b, c, d = v(iy - 1, iz - 1), v(iy, iz - 1), v(iy, iz), v(iy - 1, iz)
        push!(connec, connect((a, b, c), Meshes.Triangle))
        push!(connec, connect((a, c, d), Meshes.Triangle))
    end
    mesh = SimpleMesh(pts, connec)

    octree = TriangleOctree(mesh; verify_orientation = false, classify_leaves = false)

    # Invariant: every triangle is present in at least one leaf's element list.
    present = falses(num_triangles(octree))
    for leaf in all_leaves(octree.tree)
        for ti in octree.tree.element_lists[leaf]
            present[ti] = true
        end
    end
    @test all(present)

    # Query 2.5mm off the sliver at mid-height: the nearest triangle must be a
    # sliver triangle (2.5mm away), not a patch triangle (7.5mm away).
    p = SVector(0.0025, 0.0, 1.0)
    state = NearestTriangleState{Float64}(p)
    _nearest_triangle_octree!(p, octree.tree, octree.index, 1, state)
    @test state.closest_idx in (1, 2)
end
