# Tests for spacing-driven octree subdivision criterion

@testitem "SpacingCriterion construction and should_subdivide" setup = [CommonImports] begin
    using WhatsThePoint: SpacingCriterion, SpatialOctree, should_subdivide, subdivide!, box_size

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)
    spacing = ConstantSpacing(1.0m)

    c = SpacingCriterion(spacing, 10.0; alpha = 2.0, min_ratio = 1.0e-6)
    @test c.alpha == 2.0
    @test c.absolute_min ≈ 10.0 * 1.0e-6

    # Root box_size=10, h_local=1.0, alpha*h_local=2.0 → 10 > 2 → true
    @test should_subdivide(c, octree, 1) == true

    # With large absolute_min: h_box <= absolute_min → false (early exit)
    c_large_min = SpacingCriterion(spacing, 100.0; alpha = 2.0, min_ratio = 1.0)
    @test c_large_min.absolute_min ≈ 100.0
    @test should_subdivide(c_large_min, octree, 1) == false

    # With very large alpha: h_box <= alpha * h_local → false
    c_large_alpha = SpacingCriterion(spacing, 10.0; alpha = 20.0, min_ratio = 1.0e-6)
    @test should_subdivide(c_large_alpha, octree, 1) == false
end

@testitem "SpacingCriterion can_subdivide" setup = [CommonImports] begin
    using WhatsThePoint: SpacingCriterion, SpatialOctree, can_subdivide, box_size

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)
    spacing = ConstantSpacing(1.0m)

    # box_size=10 > absolute_min=1e-5 → true
    c = SpacingCriterion(spacing, 10.0; alpha = 2.0, min_ratio = 1.0e-6)
    @test can_subdivide(c, octree, 1) == true

    # box_size=10 <= absolute_min=100 → false
    c_large = SpacingCriterion(spacing, 100.0; alpha = 2.0, min_ratio = 1.0)
    @test can_subdivide(c_large, octree, 1) == false
end

@testitem "SpacingCriterion h_local near-zero early exit" setup = [CommonImports] begin
    using WhatsThePoint: SpacingCriterion, SpatialOctree, should_subdivide, AbstractSpacing

    # Spacing that returns near-zero to trigger h_local <= eps(T) path
    struct TinySpacing <: AbstractSpacing end
    (::TinySpacing)(_) = 0.0m

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    c = SpacingCriterion(TinySpacing(), 10.0; alpha = 2.0, min_ratio = 1.0e-6)
    @test should_subdivide(c, octree, 1) == false
end

@testitem "_box_may_contain_interior" setup = [CommonImports, OctreeTestData] begin
    using WhatsThePoint: _box_may_contain_interior, SpatialOctree, box_center

    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)

    # The root node tree shares bounds with the triangle octree and contains
    # the unit cube → should contain interior
    bbox_min, bbox_max = WhatsThePoint.bounding_box(tri_octree.tree)
    node_tree = SpatialOctree{Int, Float64}(bbox_min, tri_octree.tree.root_size)
    @test _box_may_contain_interior(node_tree, 1, tri_octree) == true

    # A box far outside the unit cube should not contain interior
    far_tree = SpatialOctree{Int, Float64}(SVector(10.0, 10.0, 10.0), 1.0)
    @test _box_may_contain_interior(far_tree, 1, tri_octree) == false
end

@testitem "_box_may_contain_interior high-aspect-ratio domain (#75)" setup = [CommonImports] begin
    using WhatsThePoint: _box_may_contain_interior, SpatialOctree, bounding_box
    using Meshes: SimpleMesh, connect, Triangle, Point

    # 20x7x3 cuboid — the cubic root node box centers all 9 original sample
    # points outside the thin domain, so subdivision relies on enriched
    # sampling (face centers / edge midpoints) or the spatial descent fallback.
    vertices = Point.(
        [
            (0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 7.0, 0.0), (0.0, 7.0, 0.0),
            (0.0, 0.0, 3.0), (20.0, 0.0, 3.0), (20.0, 7.0, 3.0), (0.0, 7.0, 3.0),
        ]
    )
    triangles = [
        connect((1, 3, 2), Triangle), connect((1, 4, 3), Triangle),
        connect((5, 6, 7), Triangle), connect((5, 7, 8), Triangle),
        connect((1, 2, 6), Triangle), connect((1, 6, 5), Triangle),
        connect((3, 4, 8), Triangle), connect((3, 8, 7), Triangle),
        connect((1, 5, 8), Triangle), connect((1, 8, 4), Triangle),
        connect((2, 3, 7), Triangle), connect((2, 7, 6), Triangle),
    ]
    mesh = SimpleMesh(vertices, triangles)
    tri_octree = TriangleOctree(mesh; classify_leaves = true)

    bbox_min, _ = bounding_box(tri_octree.tree)
    node_tree = SpatialOctree{Int, Float64}(bbox_min, tri_octree.tree.root_size)
    @test _box_may_contain_interior(node_tree, 1, tri_octree) == true

    # Fallback must still work when the triangle octree has no classification
    tri_octree_unclassified = TriangleOctree(mesh; classify_leaves = false)
    bbox_min_u, _ = bounding_box(tri_octree_unclassified.tree)
    node_tree_u = SpatialOctree{Int, Float64}(
        bbox_min_u, tri_octree_unclassified.tree.root_size
    )
    @test _box_may_contain_interior(node_tree_u, 1, tri_octree_unclassified) == true
end
