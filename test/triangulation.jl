# Tests for Triangulation — multi-STL triangle geometry (load, merge, orient).

@testitem "Triangulation - single container" setup = [CommonImports, TestData] begin
    using WhatsThePoint: patch_range, role, num_triangles

    tri = load_triangulation(TestData.BOX_PATH; units = u"m")

    @test npatches(tri) == 1
    @test patches(tri) == [:box]                     # name derived from file stem
    @test role(tri, :box) == :container
    @test num_triangles(tri) > 0
    @test length(patch_range(tri, :box)) == num_triangles(tri)

    # merged index is queryable: interior classifies inside, far corner outside
    oct = TriangleOctree(tri)
    center = (tri.index.bbox_min .+ tri.index.bbox_max) ./ 2
    @test isinside(center, oct)
    @test !isinside(tri.index.bbox_max, oct)
end

@testitem "Triangulation - merge mechanics and obstacle winding flip" setup = [CommonImports, TestData] begin
    using WhatsThePoint: patch_range, num_triangles

    box = TestData.BOX_PATH
    n1 = num_triangles(load_triangulation(box; units = u"m"))

    # Same box as container + obstacle is a synthetic zero-fluid case (it emits a
    # non-positive-volume warning — expected here); we assert only the mechanics.
    tri = load_triangulation(:outer => (box, :container), :inner => (box, :obstacle); units = u"m")

    @test npatches(tri) == 2
    @test patches(tri) == [:outer, :inner]
    @test num_triangles(tri) == 2 * n1

    ro = patch_range(tri, :outer)
    ri = patch_range(tri, :inner)
    @test ro == 1:n1
    @test ri == (n1 + 1):(2 * n1)

    # Obstacle winding is reversed → its rebuilt face normals are negated.
    @test tri.index.face[first(ri)] ≈ -tri.index.face[first(ro)]
    @test tri.index.face[last(ri)] ≈ -tri.index.face[last(ro)]
end

@testitem "Triangulation - input validation" setup = [CommonImports, TestData] begin
    box = TestData.BOX_PATH

    @test_throws ArgumentError load_triangulation(; units = u"m")                       # empty
    @test_throws ArgumentError load_triangulation(:a => (box, :wall); units = u"m")     # bad role
    @test_throws ArgumentError load_triangulation(                                      # duplicate name
        :a => (box, :container), :a => (box, :obstacle); units = u"m",
    )
    @test_throws ArgumentError load_triangulation(:a => box; units = u"m")              # spec not (path, role)
    @test_throws ArgumentError load_triangulation(box; units = u"m", role = :wall)      # bad single role
end

@testitem "Triangulation - orientation guard rejects inside-out" setup = [CommonImports, TestData, OctreeTestData] begin
    using WhatsThePoint: _verify_patch_orientation, TriangleIndex
    using Meshes: SimpleMesh, connect, vertices
    import Meshes

    # Outward-wound closed cube: signed volume positive → passes.
    cube = OctreeTestData.unit_cube_mesh(Float64)
    idx_out = TriangleIndex(Float64, cube)
    @test _verify_patch_orientation(idx_out, :cube, :container) === nothing

    # Reverse every triangle's winding → inside-out → guard must reject.
    rev = [
        connect(
                (Meshes.indices(c)[1], Meshes.indices(c)[3], Meshes.indices(c)[2]),
                Meshes.Triangle,
            )
            for c in Meshes.elements(Meshes.topology(cube))
    ]
    idx_in = TriangleIndex(Float64, SimpleMesh(vertices(cube), rev))
    @test_throws ArgumentError _verify_patch_orientation(idx_in, :cube, :container)
end
