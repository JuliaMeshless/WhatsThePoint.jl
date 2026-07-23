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

@testitem "PointBoundary(tri, spacing) - named per-patch surfaces" setup = [
    CommonImports, TestData, STLHelpers,
] begin
    Random.seed!(11)

    # Container = box.stl (binary, Float32); obstacle = 2×2×2 ASCII-STL box
    # (Float64) inside it — also exercises mactype promotion to Float64.
    dir = mktempdir()
    inner_path = joinpath(dir, "inner.stl")
    inner_mesh = Meshes.simplexify(
        Meshes.boundary(Box(Meshes.Point(3.0, 1.0, 1.0), Meshes.Point(5.0, 3.0, 3.0))),
    )
    STLHelpers.write_ascii_stl(inner_path, inner_mesh)

    tri = load_triangulation(
        :outer => (TestData.BOX_PATH, :container),
        :inner => (inner_path, :obstacle);
        units = u"m",
    )
    @test tri isa Triangulation{Float64}

    bnd = PointBoundary(tri, ConstantSpacing(0.5m))
    surfs = namedsurfaces(bnd)

    # Patch names become surface names, in load order (BCs stay taggable).
    @test collect(keys(surfs)) == [:outer, :inner]

    # Container normals point out of the box (away from its center).
    c_out = SVector(12.5, 12.5, 12.5)
    outer = surfs[:outer]
    @test all(zip(normal(outer), points(outer))) do (n, p)
        dot(n, SVector{3}(ustrip.(to(p))) - c_out) > 0
    end

    # Obstacle normals point into the obstacle (out of the fluid).
    c_in = SVector(4.0, 2.0, 2.0)
    inner = surfs[:inner]
    @test all(zip(normal(inner), points(inner))) do (n, p)
        dot(n, c_in - SVector{3}(ustrip.(to(p)))) > 0
    end

    # Per-patch area preservation (2×2×2 box → 24 m²).
    @test sum(area(inner)) ≈ 24.0m^2 atol = 1.0e-6m^2
end

@testitem "Triangulation - obstacle flush on container floor (seam veto)" setup = [
    CommonImports, TestData, STLHelpers,
] begin
    # Obstacle resting flush on the container floor (its base is coincident
    # with the floor plane). At the seam the container and obstacle surfaces
    # tie in the nearest-feature search; the floor's outward normal then
    # classifies the obstacle interior above the seam as fluid. The seam veto
    # (flagged container triangles + obstacle cross-check) must resolve it.
    dir = mktempdir()
    inner_path = joinpath(dir, "inner.stl")
    inner_mesh = Meshes.simplexify(
        Meshes.boundary(Box(Meshes.Point(10.0, 10.0, 0.0), Meshes.Point(12.0, 12.0, 2.0))),
    )
    STLHelpers.write_ascii_stl(inner_path, inner_mesh)

    tri = load_triangulation(
        :outer => (TestData.BOX_PATH, :container),
        :inner => (inner_path, :obstacle);
        units = u"m",
    )
    oct = TriangleOctree(tri)

    # seam detected on the floor triangles under the obstacle
    @test oct.seam_veto !== nothing
    @test any(oct.seam_veto.flags)

    # obstacle interior (above the flush base) is NOT fluid…
    @test !isinside(SVector(11.0, 11.0, 1.0), oct)
    @test !isinside(SVector(11.0, 11.0, 0.01), oct)
    # …but the fluid above and beside it is
    @test isinside(SVector(11.0, 11.0, 3.0), oct)
    @test isinside(SVector(11.0, 13.5, 0.01), oct)

    # end-to-end: no volume point lands inside the flush obstacle
    Random.seed!(5)
    spacing = ConstantSpacing(1.0m)
    bnd = PointBoundary(tri, spacing)
    cloud = discretize(bnd, spacing; alg = Octree(tri; spacing))
    n_inside = count(points(volume(cloud))) do p
        x, y, z = ustrip.(to(p))
        10.0 < x < 12.0 && 10.0 < y < 12.0 && 0.0 < z < 2.0
    end
    @test n_inside == 0
    @test length(volume(cloud)) > 1000
end

@testitem "Octree(tri) algorithm constructor" setup = [CommonImports, TestData] begin
    tri = load_triangulation(TestData.BOX_PATH; units = u"m")
    spacing = ConstantSpacing(1.0m)

    alg = Octree(tri; spacing)
    @test alg isa Octree
    @test alg.triangle_octree isa TriangleOctree
    @test alg.placement == :bridson

    # Parity with the mesh path on the same single-patch geometry.
    alg_mesh = Octree(import_mesh(TestData.BOX_PATH, m); spacing)
    @test typeof(alg) == typeof(alg_mesh)
    @test alg.node_min_ratio ≈ alg_mesh.node_min_ratio

    # Same kwargs validation as the mesh constructor.
    @test_throws ArgumentError Octree(tri; placement = :bogus)
    @test_throws ArgumentError Octree(tri; alpha = 0.0)
    @test_throws ArgumentError Octree(tri; boundary_oversampling = 0.0)
    @test_throws ArgumentError Octree(tri; bridson_factor = 0.0)
    @test_throws ArgumentError Octree(tri; max_growth = -1.0)
end

@testitem "Triangulation end-to-end: box container + box obstacle" setup = [
    CommonImports, TestData, STLHelpers,
] begin
    Random.seed!(7)

    dir = mktempdir()
    inner_path = joinpath(dir, "inner.stl")
    inner_mesh = Meshes.simplexify(
        Meshes.boundary(Box(Meshes.Point(3.0, 1.0, 1.0), Meshes.Point(5.0, 3.0, 3.0))),
    )
    STLHelpers.write_ascii_stl(inner_path, inner_mesh)

    tri = load_triangulation(
        :outer => (TestData.BOX_PATH, :container),
        :inner => (inner_path, :obstacle);
        units = u"m",
    )
    spacing = ConstantSpacing(1.0m)
    bnd = PointBoundary(tri, spacing)
    cloud = discretize(bnd, spacing; alg = Octree(tri; spacing))

    # Both patches are present as named surfaces.
    @test collect(keys(namedsurfaces(boundary(cloud)))) == [:outer, :inner]

    # No volume point inside the obstacle; fluid exists around it.
    n_inside = count(points(volume(cloud))) do p
        x, y, z = ustrip.(to(p))
        3.0 < x < 5.0 && 1.0 < y < 3.0 && 1.0 < z < 3.0
    end
    @test n_inside == 0
    @test length(volume(cloud)) > 1000

    # Merged-octree classification: obstacle interior is outside the fluid,
    # container center is inside.
    oct = TriangleOctree(tri)
    @test !isinside(SVector(4.0, 2.0, 2.0), oct)
    @test isinside(SVector(12.5, 12.5, 12.5), oct)
end
