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

    g_path = suggest_spacing(TestData.BOX_PATH, u"m"; verbose = false)
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

@testitem "suggest_spacing verbose prints guidance" setup = [CommonImports, OctreeTestData] begin
    using Unitful: ustrip

    mesh = OctreeTestData.unit_cube_mesh()
    old_stdout = stdout
    rd, wr = redirect_stdout()
    g = suggest_spacing(mesh; verbose = true, name = "test_cube")
    redirect_stdout(old_stdout)
    close(wr)
    str = read(rd, String)
    close(rd)
    # The verbose print must mention the geometry name and the key landmarks
    @test occursin("test_cube", str)
    @test occursin("h_ceiling", str)
    @test occursin("h_baseline", str)
    @test occursin("h_fine", str)
    @test occursin("start here", str)
end

@testitem "suggest_spacing degenerate geometry throws" setup = [CommonImports, OctreeTestData] begin
    using Unitful: m
    # A flat boundary (all points in a plane) gives a zero-width bounding box
    # in one axis, which _spacing_guidance rejects.
    pts = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.5, 0.0, 0.0)]
    normals = [SVector(0.0, 0.0, 1.0) for _ in pts]
    areas = [0.0 * u"m^2" for _ in pts]
    surf = PointSurface(pts, normals, areas)
    bnd = PointBoundary(LittleDict(:s1 => surf))
    @test_throws ArgumentError suggest_spacing(bnd; verbose = false)
end

@testitem "_extract_min_spacing sees through the coarse-spacing clamp" setup = [CommonImports] begin
    using Unitful: m

    # The node octree resolution must follow whichever of inner minimum and
    # cap actually drives the fill.
    clamped = WhatsThePoint._ClampedSpacing(ConstantSpacing(2.0m), 0.1m)
    @test WhatsThePoint._extract_min_spacing(clamped) == 0.1

    fine = WhatsThePoint._ClampedSpacing(ConstantSpacing(0.05m), 0.1m)
    @test WhatsThePoint._extract_min_spacing(fine) == 0.05

    # An inner spacing with unknown minimum (LogLike) falls back to the cap.
    pts = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)]
    loglike = WhatsThePoint.LogLike(pts, 0.5m, 1.5)
    capped = WhatsThePoint._ClampedSpacing(loglike, 0.1m)
    @test WhatsThePoint._extract_min_spacing(capped) == 0.1
end
