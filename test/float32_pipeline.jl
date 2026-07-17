@testitem "Float32 pipeline: octree, discretize, sampling, guidance" setup = [CommonImports, OctreeTestData] begin
    # End-to-end machine-type preservation: a Float32 mesh must flow Float32
    # through TriangleOctree, the Octree algorithm, discretize, sample_surface,
    # and suggest_spacing — no silent promotion to Float64. Assertions are
    # type-level only; point quality on a coarse Float32 cube is not the target.
    mesh32 = OctreeTestData.unit_cube_mesh(Float32)
    @test CoordRefSystems.mactype(Meshes.crs(mesh32)) === Float32
    sp = ConstantSpacing(0.25f0u"m")

    tri = TriangleOctree(mesh32; classify_leaves = true)
    @test tri.tree isa WhatsThePoint.SpatialOctree{Int, Float32}
    @test eltype(tri.index.bbox_min) === Float32

    alg = Octree(mesh32; spacing = sp)
    @test alg.alpha isa Float32
    @test alg.boundary_oversampling isa Float32
    @test alg.bridson_factor isa Float32
    @test alg.max_growth isa Float32
    @test alg.node_min_ratio isa Float32

    cloud = discretize(PointBoundary(mesh32), sp; alg, max_points = 200)
    @test CoordRefSystems.mactype(Meshes.crs(first(points(cloud)))) === Float32
    @test length(WhatsThePoint.volume(cloud)) > 0

    surf = sample_surface(mesh32, sp)
    @test CoordRefSystems.mactype(Meshes.crs(first(points(surf)))) === Float32
    @test eltype(WhatsThePoint.normal(surf)) <: SVector{3, Float32}
    @test ustrip(first(WhatsThePoint.area(surf))) isa Float32

    g = suggest_spacing(mesh32; verbose = false)
    @test ustrip(g.h_ceiling) isa Float32
    @test ustrip(g.h_baseline) isa Float32
    @test ustrip(g.h_fine) isa Float32
end

@testitem "Float32 pipeline: repel and spacing fidelity metrics" setup = [CommonImports, OctreeTestData] begin
    mesh32 = OctreeTestData.unit_cube_mesh(Float32)
    sp = ConstantSpacing(0.25f0u"m")
    tri = TriangleOctree(mesh32; classify_leaves = true)
    cloud = discretize(
        PointBoundary(mesh32), sp; alg = Octree(mesh32; spacing = sp), max_points = 200
    )

    c_vol = repel(cloud, sp; β = 0.2f0, max_iters = 3)
    @test CoordRefSystems.mactype(Meshes.crs(first(points(c_vol)))) === Float32

    # A user-supplied Float64 convergence vector keeps working: entries are
    # computed in Float32 and convert on insertion.
    conv = Float64[]
    c_oct = repel(cloud, sp, tri; β = 0.2f0, max_iters = 3, convergence = conv)
    @test CoordRefSystems.mactype(Meshes.crs(first(points(c_oct)))) === Float32
    @test length(conv) == 3

    fm = spacing_fidelity_metrics(cloud, sp; k = 8)
    @test fm.mean_dnn_h isa Float32
    @test fm.cv isa Float32
    @test fm.p05 isa Float32 && fm.p50 isa Float32 && fm.p95 isa Float32
end
