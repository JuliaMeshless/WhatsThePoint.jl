# Tests for Octree discretization algorithm helpers

@testitem "Octree invalid alpha throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError Octree(octree; alpha = -1.0)
    @test_throws ArgumentError Octree(octree; alpha = 0.0)
end

@testitem "_extract_min_spacing methods" setup = [CommonImports, OctreeTestData] begin
    using WhatsThePoint: _extract_min_spacing, AbstractSpacing

    # ConstantSpacing
    @test _extract_min_spacing(ConstantSpacing(2.5m)) ≈ 2.5

    # BoundaryLayerSpacing
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    bls = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.3m,
        bulk = 3.0m,
        layer_thickness = 1.5m,
    )
    @test _extract_min_spacing(bls) ≈ 0.3

    # Fallback for unknown spacing type
    struct CustomSpacing <: AbstractSpacing end
    @test _extract_min_spacing(CustomSpacing()) === nothing
end

@testitem "_allocate_counts_by_volume edge cases" setup = [CommonImports] begin
    using WhatsThePoint: _allocate_counts_by_volume

    # Empty volumes
    @test _allocate_counts_by_volume(Float64[], 10) == Int[]

    # Zero total count
    result = _allocate_counts_by_volume([1.0, 2.0, 3.0], 0)
    @test result == [0, 0, 0]

    # Normal allocation preserves total count
    result = _allocate_counts_by_volume([1.0, 2.0, 3.0], 60)
    @test sum(result) == 60
    @test all(result .>= 0)

    # All-zero volumes distributes uniformly
    result = _allocate_counts_by_volume([0.0, 0.0, 0.0], 9)
    @test sum(result) == 9
    @test all(result .>= 0)

    # ensure_one gives each at least 1 when total_count >= n
    result = _allocate_counts_by_volume([1.0, 1.0, 1.0], 6; ensure_one = true)
    @test sum(result) == 6
    @test all(result .>= 1)

    # ensure_one with exact count = n
    result = _allocate_counts_by_volume([1.0, 1.0], 2; ensure_one = true)
    @test result == [1, 1]

    # Single volume gets all points
    result = _allocate_counts_by_volume([5.0], 10)
    @test result == [10]
end

@testitem "_generate_points_in_box placement modes" setup = [CommonImports] begin
    using WhatsThePoint: _generate_points_in_box
    using Random
    Random.seed!(42)

    bbox_min = SVector(0.0, 0.0, 0.0)
    bbox_max = SVector(1.0, 1.0, 1.0)

    # n=0 returns empty
    @test isempty(_generate_points_in_box(bbox_min, bbox_max, 0, :random))

    for placement in (:random, :jittered, :lattice)
        pts = _generate_points_in_box(bbox_min, bbox_max, 10, placement)
        @test length(pts) == 10
        for p in pts
            @test all(bbox_min .<= p .<= bbox_max)
        end
    end

    # Negative n returns empty
    @test isempty(_generate_points_in_box(bbox_min, bbox_max, -1, :random))
end

@testitem "Octree mesh constructor with spacing hint" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()

    # With spacing hint: node_min_ratio computed from spacing
    alg_with_spacing = Octree(mesh; spacing = ConstantSpacing(0.5m), alpha = 2.0)
    @test alg_with_spacing.node_min_ratio > 0
    @test alg_with_spacing.alpha == 2.0

    # Without spacing: uses geometry default
    alg_no_spacing = Octree(mesh)
    @test alg_no_spacing.node_min_ratio > 0

    # The spacing-derived ratio should differ from the geometry-only ratio
    @test alg_with_spacing.node_min_ratio != alg_no_spacing.node_min_ratio
end
