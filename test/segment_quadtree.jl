# Tests for the 2D geometry index (SegmentQuadtree) and the 2D Octree
# discretization path.

@testitem "SegmentQuadtree construction and queries" setup = [CommonImports] begin
    # Unit square, CCW
    square = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
    sq = SegmentQuadtree(square)

    @test sq isa SegmentQuadtree
    @test num_segments(sq) == 4
    @test num_leaves(sq) >= 1

    @test isinside(SVector(0.5, 0.5), sq)
    @test !isinside(SVector(1.5, 0.5), sq)
    @test !isinside(SVector(-0.1, -0.1), sq)

    # Point queries near the boundary classify as boundary within tolerance
    @test WhatsThePoint.classify_point(sq, SVector(0.5, 1.0e-12), 1.0e-6) ==
        WhatsThePoint.LEAF_BOUNDARY
end

@testitem "SegmentQuadtree orientation invariance" setup = [CommonImports] begin
    # The same square, clockwise — construction must normalize orientation.
    square_cw = [SVector(0.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0), SVector(1.0, 0.0)]
    sq = SegmentQuadtree(square_cw)
    @test isinside(SVector(0.5, 0.5), sq)
    @test !isinside(SVector(2.0, 0.5), sq)
end

@testitem "SegmentQuadtree multiply-connected domain (hole)" setup = [CommonImports] begin
    outer = [SVector(2.0 * cos(t), 2.0 * sin(t)) for t in range(0, 2pi; length = 65)[1:64]]
    hole = [SVector(0.5 * cos(t), 0.5 * sin(t)) for t in range(0, 2pi; length = 33)[1:32]]
    sq = SegmentQuadtree([outer, hole])

    @test !isinside(SVector(0.0, 0.0), sq)      # inside the hole -> outside domain
    @test isinside(SVector(1.2, 0.0), sq)       # annulus -> inside
    @test !isinside(SVector(2.5, 0.0), sq)      # beyond outer -> outside

    # Hole orientation must not matter either
    sq2 = SegmentQuadtree([outer, reverse(hole)])
    @test !isinside(SVector(0.0, 0.0), sq2)
    @test isinside(SVector(1.2, 0.0), sq2)
end

@testitem "SegmentQuadtree degenerate input errors" setup = [CommonImports] begin
    @test_throws ArgumentError SegmentQuadtree([SVector(0.0, 0.0), SVector(1.0, 0.0)])
    collinear = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(2.0, 0.0)]
    @test_throws ArgumentError SegmentQuadtree(collinear)
    # Collinear in Float32 too: the area threshold must clear the shoelace's
    # own round-off noise, which the Float64-flavored 1e-10 does not.
    @test_throws ArgumentError SegmentQuadtree(
        [SVector(0.0f0, 0.0f0), SVector(1.0f0, 0.0f0), SVector(2.0f0, 0.0f0)]
    )
    # A loop that collapses to < 3 distinct vertices once duplicates are dropped
    @test_throws ArgumentError SegmentQuadtree(
        [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 0.0)]
    )
end

@testitem "SegmentQuadtree orientation is origin-independent" setup = [CommonImports] begin
    # The shoelace sum must be centered on the loop: on absolute coordinates a
    # small Float32 loop far from the origin loses its area to cancellation and
    # the orientation — hence which side is inside — becomes round-off noise.
    for off in (0.0f0, 1.0f3, 1.0f5)
        square = [
            SVector(off, off), SVector(off + 1.0f0, off),
            SVector(off + 1.0f0, off + 1.0f0), SVector(off, off + 1.0f0),
        ]
        sq = SegmentQuadtree(square)
        @test isinside(SVector(off + 0.5f0, off + 0.5f0), sq)
        @test !isinside(SVector(off + 1.5f0, off + 0.5f0), sq)
        @test !isinside(SVector(off - 0.5f0, off + 0.5f0), sq)
    end
end

@testitem "SegmentQuadtree accepts explicitly-closed loops" setup = [CommonImports] begin
    # `[p₁, …, pₙ, p₁]` is the standard convention: the closing duplicate must
    # be dropped, not turned into a zero-length segment (which leaves the two
    # coincident seam vertices with half-built pseudonormals).
    open_sq = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
    closed_sq = vcat(open_sq, [SVector(0.0, 0.0)])
    @test num_segments(SegmentQuadtree(closed_sq)) == 4
    @test all(!iszero, SegmentQuadtree(closed_sq).index.normal)

    # Sharp corner at the seam: a half-built pseudonormal there misclassifies an
    # angular sector of exterior points as interior. Check against the
    # independent even-odd crossing test.
    wedge = [SVector(0.0, 0.0), SVector(1.0, -0.05), SVector(1.0, 0.05)]
    sq = SegmentQuadtree(vcat(wedge, [wedge[1]]))
    probes = [
        SVector(r * cos(θ), r * sin(θ))
            for r in (0.001, 0.01, 0.05) for θ in range(0, 2pi; length = 73)[1:72]
    ]
    @test all(isinside(p, sq) == WhatsThePoint._point_in_loop(p, wedge) for p in probes)
end

@testitem "SegmentQuadtree degeneracy thresholds scale with the boundary" setup = [CommonImports] begin
    # A small-but-legitimate Float32 domain: segment lengths of 1e-5 coordinate
    # units sit under the old absolute `eps(T)*100` floor, which zeroed every
    # normal and made the whole domain classify as exterior.
    s = 1.0f-5
    sq = SegmentQuadtree(
        [SVector(0.0f0, 0.0f0), SVector(s, 0.0f0), SVector(s, s), SVector(0.0f0, s)]
    )
    @test all(n -> isapprox(norm(n), 1.0f0), sq.index.normal)
    @test isinside(SVector(s / 2, s / 2), sq)
    @test !isinside(SVector(3s, s / 2), sq)

    # Finely sampled Float32 circle: same floor, reached by segment count.
    n = 20_000
    circle = [SVector{2, Float32}(cos(t), sin(t)) for t in range(0, 2pi; length = n + 1)[1:n]]
    qc = SegmentQuadtree(circle)
    @test all(p -> isinside(p, qc), [SVector{2, Float32}(0.5f0 * cos(t), 0.5f0 * sin(t)) for t in range(0, 2pi; length = 64)])
    @test all(p -> !isinside(p, qc), [SVector{2, Float32}(1.5f0 * cos(t), 1.5f0 * sin(t)) for t in range(0, 2pi; length = 64)])
end

@testitem "SegmentQuadtree from PointBoundary" setup = [CommonImports] begin
    pts = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    bnd = PointBoundary(pts)
    sq = SegmentQuadtree(bnd)
    @test num_segments(sq) == 4
    @test isinside(Point(0.5, 0.5), sq)
    @test !isinside(Point(1.5, 0.5), sq)
end

@testitem "Octree discretization in 2D (bridson)" setup = [CommonImports] begin
    using Random
    Random.seed!(42)

    # Star-shaped domain exercises non-convexity
    pts = Point.(
        [
            (cos(t) * (1 + 0.3 * cos(5t)), sin(t) * (1 + 0.3 * cos(5t)))
                for t in range(0, 2pi; length = 121)[1:120]
        ]
    )
    bnd = PointBoundary(pts)
    spacing = ConstantSpacing(0.08m)

    alg = Octree(bnd; spacing)
    cloud = discretize(bnd, spacing; alg)

    vol = WhatsThePoint.volume(cloud)
    @test length(vol) > 100

    # Every volume point must lie inside the boundary polygon
    sq = alg.geometry
    @test all(isinside(p, sq) for p in vol)

    # Poisson-disk separation: min pairwise distance ≥ bridson_factor · h
    # between volume points (guaranteed by construction; small numerical slack)
    coords = [SVector(ustrip(to(p)[1]), ustrip(to(p)[2])) for p in vol]
    dmin = minimum(
        sqrt(sum(abs2, coords[i] - coords[j]))
            for i in eachindex(coords) for j in (i + 1):length(coords)
    )
    @test dmin >= 0.75 * 0.08 * (1 - 1.0e-9)
end

@testitem "Octree 2D respects max_points cap" setup = [CommonImports] begin
    using Random
    Random.seed!(7)
    pts = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    bnd = PointBoundary(pts)
    spacing = ConstantSpacing(0.05m)
    alg = Octree(bnd; spacing)
    cloud = @test_logs (:warn, r"truncated") match_mode = :any discretize(
        bnd, spacing; alg, max_points = 20,
    )
    @test length(WhatsThePoint.volume(cloud)) <= 20
end

@testitem "Octree 2D preserves the boundary length unit" setup = [CommonImports] begin
    using Random
    Random.seed!(11)
    # `Point(pt...)` on unit-stripped magnitudes attaches metres unconditionally,
    # so a non-metre boundary used to exit with metre-typed, mm-valued points —
    # a CRS mismatch at PointCloud assembly (or a silent 1000× mix).
    pts = Point.([(0.0u"mm", 0.0u"mm"), (10.0u"mm", 0.0u"mm"), (10.0u"mm", 10.0u"mm"), (0.0u"mm", 10.0u"mm")])
    bnd = PointBoundary(pts)
    spacing = ConstantSpacing(1.0u"mm")

    for placement in (:bridson, :random)
        alg = Octree(bnd; spacing, placement)
        cloud = discretize(bnd, spacing; alg, max_points = 60)
        vol = volume(cloud)
        @test length(vol) > 0
        @test all(Unitful.unit(to(p)[1]) == u"mm" for p in vol)
        @test all(p -> isinside(p, alg.geometry), vol)
    end
end

@testitem "2D discretize defaults to Octree" setup = [CommonImports] begin
    using Random
    Random.seed!(19)
    circle = Point.([(cos(t), sin(t)) for t in range(0, 2pi; length = 121)[1:120]])
    bnd = PointBoundary(circle)

    # No `alg`: the Poisson-disk Octree fill, not the FornbergFlyer height field.
    cloud = discretize(bnd, ConstantSpacing(0.1m))
    vol = volume(cloud)
    @test length(vol) > 100
    coords = [SVector(ustrip(to(p)[1]), ustrip(to(p)[2])) for p in vol]
    dmin = minimum(
        norm(coords[i] - coords[j])
            for i in eachindex(coords) for j in (i + 1):length(coords)
    )
    @test dmin >= 0.75 * 0.1 * (1 - 1.0e-9)

    # A graded spacing through the default path — the combination FornbergFlyer
    # has no method for, so it used to die in a raw MethodError.
    graded = BoundaryLayerSpacing(
        points(bnd); at_wall = 0.03m, bulk = 0.12m, layer_thickness = 0.3m
    )
    @test length(volume(discretize(bnd, graded))) > 0

    # The PointCloud entry point takes the same default (it used to hardcode
    # SlakKosec, so refining a 2D cloud was a MethodError).
    @test length(volume(discretize(PointCloud(bnd), ConstantSpacing(0.1m)))) > 100

    # FornbergFlyer stays available, explicitly
    @test length(volume(discretize(bnd, ConstantSpacing(0.1m); alg = FornbergFlyer(), max_points = 50))) > 0
end

@testitem "2D discretize rejects unusable algorithm/spacing combinations" setup = [
    CommonImports, TestData,
] begin
    pts = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    bnd = PointBoundary(pts)
    spacing = ConstantSpacing(0.2m)

    @test_throws ArgumentError discretize(bnd, spacing; alg = SlakKosec())
    @test_throws ArgumentError discretize(PointCloud(bnd), spacing; alg = SlakKosec())

    # An Octree built from 3D geometry cannot fill a 2D boundary: instructive
    # ArgumentError, not a raw MethodError out of _discretize_volume.
    mesh = import_mesh(TestData.BOX_PATH, m)
    @test_throws ArgumentError discretize(bnd, spacing; alg = Octree(mesh; spacing))

    # FornbergFlyer has only a ConstantSpacing method
    graded = BoundaryLayerSpacing(
        points(bnd); at_wall = 0.05m, bulk = 0.2m, layer_thickness = 0.3m
    )
    @test_throws ArgumentError discretize(bnd, graded; alg = FornbergFlyer())
end

@testitem "Octree 2D per-leaf placements" setup = [CommonImports] begin
    using Random
    Random.seed!(3)
    pts = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    bnd = PointBoundary(pts)
    spacing = ConstantSpacing(0.1m)

    for placement in (:random, :jittered, :lattice)
        alg = Octree(bnd; spacing, placement)
        cloud = discretize(bnd, spacing; alg, max_points = 60)
        vol = WhatsThePoint.volume(cloud)
        @test length(vol) > 0
        @test length(vol) <= 60
        sq = alg.geometry
        @test all(isinside(p, sq) for p in vol)
    end
end
