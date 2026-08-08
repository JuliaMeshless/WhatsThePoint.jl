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

@testitem "2D discretize rejects 3D-only algorithms" setup = [CommonImports] begin
    pts = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    bnd = PointBoundary(pts)
    @test_throws ArgumentError discretize(bnd, ConstantSpacing(0.2m); alg = SlakKosec())
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
