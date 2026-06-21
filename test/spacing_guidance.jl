# Tests for suggest_spacing (the step-0 spacing probe) and the bridson
# coarse-spacing clamp-and-warn guard.

@testitem "suggest_spacing reports a fillable baseline" setup = [CommonImports, OctreeTestData] begin
    using Unitful: m, ustrip

    mesh = OctreeTestData.unit_cube_mesh()
    g = suggest_spacing(mesh; verbose = false)

    # Unit cube: shortest axis 1 m, ceiling = L_min/(2·0.75) = 0.667 m.
    @test ustrip(g.min_extent) ≈ 1.0
    @test ustrip(g.h_ceiling) ≈ 1 / 1.5 rtol = 1.0e-6
    @test ustrip(g.volume) ≈ 1.0 rtol = 1.0e-6

    # The recommended baseline must sit comfortably below the ceiling so it
    # actually fills, and finer still for h_fine.
    @test g.h_baseline < g.h_ceiling
    @test g.h_fine < g.h_baseline
    @test g.n_baseline > 0 && g.n_fine > g.n_baseline

    # The headline promise: discretizing at the suggested baseline yields a
    # non-empty cloud (the coarse-spacing clamp is exercised separately).
    bnd = PointBoundary(mesh)
    alg = Octree(mesh)
    cloud = discretize(bnd, g.h_baseline; alg, max_points = 5000)
    @test length(WhatsThePoint.volume(cloud)) > 0
end

@testitem "suggest_spacing budget mode targets a point count" setup = [
    CommonImports, OctreeTestData,
] begin
    mesh = OctreeTestData.unit_cube_mesh()
    g = suggest_spacing(mesh; n_points = 5000, verbose = false)
    # cbrt(V/N) inverts the spacing integral, so the estimate lands on target
    # (capped only if it would exceed the fillable ceiling, which 5000 does not).
    @test g.n_baseline ≈ 5000 rtol = 0.05
    @test g.h_baseline < g.h_ceiling
end

@testitem "suggest_spacing accepts boundary and file inputs" setup = [
    TestData, CommonImports,
] begin
    using Unitful: ustrip
    import GeoIO

    g_path = suggest_spacing(TestData.BOX_PATH; verbose = false)
    @test ustrip(g_path.h_ceiling) > 0
    @test g_path.n_triangles > 0

    bnd = PointBoundary(GeoIO.load(TestData.BOX_PATH).geometry)
    g_bnd = suggest_spacing(bnd; verbose = false)
    # Same bounding box ⇒ same extent-driven landmarks regardless of entry point.
    @test ustrip(g_bnd.min_extent) ≈ ustrip(g_path.min_extent) rtol = 1.0e-6
    @test ustrip(g_bnd.h_ceiling) ≈ ustrip(g_path.h_ceiling) rtol = 1.0e-6
end

@testitem "Octree bridson clamps and warns on too-coarse spacing" setup = [
    CommonImports, OctreeTestData,
] begin
    using Unitful: m

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    alg = Octree(mesh)

    # h = 1 m on a 1 m cube is above the Poisson-disk ceiling: the unguarded
    # bridson front is empty. The guard must warn loudly and still produce a
    # usable (non-empty) cloud rather than silently returning nothing.
    cloud = @test_logs (:warn,) match_mode = :any discretize(
        bnd, ConstantSpacing(1m); alg, max_points = 100,
    )
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end
