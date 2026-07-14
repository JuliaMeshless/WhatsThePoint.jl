# Tests for Octree discretization algorithm

@testitem "Octree with different spacing types" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(42)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    # Test ConstantSpacing
    alg = Octree(mesh)
    cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 100)
    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100

    # Test BoundaryLayerSpacing
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.5m,
        bulk = 5.0m,
        layer_thickness = 2.0m
    )
    alg = Octree(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)
    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test length(WhatsThePoint.volume(cloud)) <= 100
end

@testitem "Octree points are inside" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(123)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(1.0m)

    alg = Octree(mesh)
    cloud = discretize(bnd, spacing; alg, max_points = 100)

    # All points should be inside via octree check
    octree = alg.triangle_octree
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end
end

@testitem "Octree points are inside (high-aspect-ratio domain, #76)" setup = [CommonImports] begin
    using Random
    using Meshes: SimpleMesh, connect, Triangle, Point
    Random.seed!(76)

    # Regression test for #76: interior and deficit points were placed in
    # node-octree leaf bounding boxes with no isinside filter, so for
    # high-aspect-ratio domains the cubic leaves at the thin boundary
    # produced points far outside the actual geometry.
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
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(0.5m)

    alg = Octree(mesh; spacing)
    cloud = discretize(bnd, spacing; alg, max_points = 500)

    # Every volume point must pass the geometry check, not just the
    # near-surface candidates that happened to be filtered.
    octree = alg.triangle_octree
    for pt in WhatsThePoint.volume(cloud)
        c = to(pt)
        sv = SVector{3, Float64}(c[1] / m, c[2] / m, c[3] / m)
        @test isinside(sv, octree) == true
    end

    # Deficit filling must actually close the deficit when the interior is
    # reachable (a 20×7×3 box is well inside the octree resolution here).
    @test length(WhatsThePoint.volume(cloud)) >= 450
end

@testitem "Octree with placement strategies" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(456)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)

    for placement in (:random, :jittered, :lattice)
        alg = Octree(mesh; placement)
        cloud = discretize(bnd, ConstantSpacing(1m); alg, max_points = 50)

        @test cloud isa PointCloud
        @test length(WhatsThePoint.volume(cloud)) > 0
        @test length(WhatsThePoint.volume(cloud)) <= 50
    end
end

@testitem "Octree :bridson placement enforces global separation" setup = [
    CommonImports, OctreeTestData,
] begin
    using Random
    using LinearAlgebra: norm
    using Unitful: ustrip
    import Meshes
    Random.seed!(789)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(0.15m)
    alg = Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
    cloud = discretize(bnd, spacing; alg, max_points = 2000)

    vol = points(WhatsThePoint.volume(cloud))
    @test length(vol) > 50              # front saturates well below max_points
    @test length(vol) <= 2000

    # Poisson-disk guarantee: every volume point is at least
    # bridson_factor·h (default 0.75) from every other point in the cloud,
    # boundary included — global, not per-leaf.
    raw = [Float64.(ustrip.(Meshes.to(p))) for p in points(cloud)]
    n_bnd = length(points(WhatsThePoint.boundary(cloud)))
    min_sep = minimum(
        norm(raw[i] - raw[j])
            for i in (n_bnd + 1):length(raw) for j in eachindex(raw) if i != j
    )
    @test min_sep >= 0.75 * 0.15 - 1.0e-9
end

@testitem "Octree auto-estimates max_points when unset" setup = [
    CommonImports, OctreeTestData,
] begin
    using Random
    Random.seed!(2718)

    mesh = OctreeTestData.unit_cube_mesh()
    spacing = ConstantSpacing(0.15m)
    bnd = PointBoundary(mesh, spacing)
    alg = Octree(mesh; spacing, alpha = 1.0, placement = :bridson)

    # max_points unset → the Octree algorithm estimates the cap from the
    # spacing integral instead of erroring on `nothing` (regression: the
    # default path was previously unreachable and crashed with a MethodError).
    cloud = discretize(bnd, spacing; alg)
    vol = points(WhatsThePoint.volume(cloud))

    # A saturated Poisson-disk front fills ≈ 0.39× the spacing integral —
    # a healthy, non-empty fill that does not depend on a hand-set cap.
    @test length(vol) > 20

    # The estimator returns a positive Int, and its 1.1× pad keeps the cap
    # above the saturated count (so the front saturates, not truncates).
    node_tree = WhatsThePoint.build_node_octree(
        alg.triangle_octree, spacing, alg.alpha, alg.node_min_ratio,
    )
    classification = WhatsThePoint.classify_node_octree(node_tree, alg.triangle_octree)
    est = WhatsThePoint._estimate_volume_points(node_tree, classification, spacing)
    @test est isa Int
    @test est > length(vol)
end

@testitem "Octree :bridson placement with graded spacing" setup = [
    CommonImports, OctreeTestData,
] begin
    using Random
    using LinearAlgebra: norm
    using Unitful: ustrip
    import Meshes
    Random.seed!(101)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = BoundaryLayerSpacing(
        points(bnd); at_wall = 0.1m, bulk = 0.25m, layer_thickness = 0.2m,
    )
    alg = Octree(mesh; spacing, alpha = 1.0, placement = :bridson, bridson_factor = 1.0)
    cloud = discretize(bnd, spacing; alg, max_points = 2000)

    vol = points(WhatsThePoint.volume(cloud))
    @test length(vol) > 50

    # Graded guarantee at bridson_factor = 1: ‖xᵢ − xⱼ‖ ≥ min(h(xᵢ), h(xⱼ))
    # for volume points against the whole cloud.
    allp = points(cloud)
    raw = [Float64.(ustrip.(Meshes.to(p))) for p in allp]
    h = [Float64(ustrip(spacing(p))) for p in allp]
    n_bnd = length(points(WhatsThePoint.boundary(cloud)))
    ok = all(
        norm(raw[i] - raw[j]) >= min(h[i], h[j]) - 1.0e-9
            for i in (n_bnd + 1):length(raw) for j in eachindex(raw) if i != j
    )
    @test ok
end

@testitem "Octree bridson warns on truncation" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(999)

    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = ConstantSpacing(0.15m)
    alg = Octree(mesh; spacing, alpha = 1.0, placement = :bridson)

    # A tiny max_points forces truncation: the front has room to grow but is
    # capped early. The truncation probe should find acceptable candidates and
    # emit the warning.
    cloud = @test_logs (:warn, "Bridson front truncated by max_points before saturation — parts of the domain may be unfilled") match_mode = :any discretize(
        bnd, spacing; alg, max_points = 5,
    )
    @test length(WhatsThePoint.volume(cloud)) == 5
end

@testitem "Octree errors on unclassified octree" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    octree = TriangleOctree(mesh; classify_leaves = false)

    @test_throws Exception discretize(
        bnd, ConstantSpacing(1m);
        alg = Octree(octree), max_points = 50
    )
end

@testitem "Octree invalid placement throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError Octree(octree; placement = :invalid)
    @test_throws ArgumentError Octree(mesh; placement = :invalid)
end

@testitem "Octree invalid oversampling throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError Octree(octree; boundary_oversampling = -1.0)
end

@testitem "Octree invalid max_growth throws" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    @test_throws ArgumentError Octree(octree; max_growth = -0.1)
    @test_throws ArgumentError Octree(mesh; max_growth = -0.1)
end

@testitem "Octree max_growth smooths steep spacing gradients" setup = [
    CommonImports, OctreeTestData,
] begin
    using Random
    using LinearAlgebra: norm
    using Unitful: ustrip
    using NearestNeighbors: KDTree, knn
    import Meshes
    Random.seed!(424242)

    # Worst-neighbor spacing ratio: how abruptly d_NN changes between a point
    # and its nearest neighbor. The gradient limiter should shrink it.
    function max_neighbor_dnn_ratio(cloud)
        pts = [Float64.(ustrip.(Meshes.to(p))) for p in points(cloud)]
        tree = KDTree(pts)
        idxs, dists = knn(tree, pts, 2, true)
        dnn = [d[2] for d in dists]
        return maximum(
            max(dnn[i] / dnn[idxs[i][2]], dnn[idxs[i][2]] / dnn[i]) for i in eachindex(pts)
        )
    end

    mesh = OctreeTestData.unit_cube_mesh()
    # Steep boundary layer: fine 0.04 at the wall, coarse 0.22 in the bulk,
    # over a thin 0.15 layer — a gradient no meshless stencil wants raw.
    spacing = BoundaryLayerSpacing(
        points(PointBoundary(mesh)); at_wall = 0.04m, bulk = 0.22m, layer_thickness = 0.15m,
    )
    bnd = PointBoundary(mesh, spacing)

    alg_raw = Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
    alg_lim = Octree(mesh; spacing, alpha = 1.0, placement = :bridson, max_growth = 0.15)

    cloud_raw = discretize(bnd, spacing; alg = alg_raw)
    cloud_lim = discretize(bnd, spacing; alg = alg_lim)

    ratio_raw = max_neighbor_dnn_ratio(cloud_raw)
    ratio_lim = max_neighbor_dnn_ratio(cloud_lim)

    # The limiter must produce a strictly smoother field, and a clearly bounded
    # worst-case neighbor ratio (steep-but-smooth).
    @test ratio_lim < ratio_raw
    @test ratio_lim < 2.5

    # Limiting makes the transition band finer → at least as many points.
    @test length(WhatsThePoint.volume(cloud_lim)) >= length(WhatsThePoint.volume(cloud_raw))
end

@testitem "Octree octree subdivision behavior" setup = [CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(789)

    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)
    node_min_ratio = 1.0e-6

    # Test 1: Fine spacing triggers subdivision
    fine_spacing = ConstantSpacing(0.2m)
    node_tree = WhatsThePoint.build_node_octree(tri_octree, fine_spacing, 2.0, node_min_ratio)
    num_leaves = length(WhatsThePoint.all_leaves(node_tree))
    @test num_leaves >= 8  # At least one level of subdivision

    # Test 2: Alpha parameter affects subdivision aggressiveness
    spacing = ConstantSpacing(0.3m)
    node_tree_aggressive = WhatsThePoint.build_node_octree(tri_octree, spacing, 1.0, node_min_ratio)
    node_tree_conservative = WhatsThePoint.build_node_octree(tri_octree, spacing, 5.0, node_min_ratio)
    @test length(WhatsThePoint.all_leaves(node_tree_aggressive)) > length(WhatsThePoint.all_leaves(node_tree_conservative))

    # Test 3: Finer spacing produces more leaves than coarse spacing
    spacing_coarse = ConstantSpacing(0.8m)
    node_tree_fine = WhatsThePoint.build_node_octree(tri_octree, fine_spacing, 2.0, node_min_ratio)
    node_tree_coarse = WhatsThePoint.build_node_octree(tri_octree, spacing_coarse, 2.0, node_min_ratio)
    @test length(WhatsThePoint.all_leaves(node_tree_fine)) > length(WhatsThePoint.all_leaves(node_tree_coarse))
    @test length(WhatsThePoint.all_leaves(node_tree_coarse)) >= 1
end

@testitem "Octree node octree classification works" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    tri_octree = TriangleOctree(mesh; classify_leaves = true)
    spacing = ConstantSpacing(0.2m)

    # Build and classify node octree
    node_tree = WhatsThePoint.build_node_octree(tri_octree, spacing, 2.0, 1.0e-6)
    classifications = WhatsThePoint.classify_node_octree(node_tree, tri_octree)

    # classifications is a Vector{Int8} indexed by box indices
    @test classifications isa Vector{Int8}
    @test length(classifications) > 0

    # Check that we have at least some interior leaves (unit cube should have interior)
    leaves = WhatsThePoint.all_leaves(node_tree)
    interior_count = count(idx -> classifications[idx] == WhatsThePoint.LEAF_INTERIOR, leaves)
    @test interior_count > 0

    # Check that all leaves are classified (not LEAF_UNKNOWN)
    for leaf_idx in leaves
        @test classifications[leaf_idx] != WhatsThePoint.LEAF_UNKNOWN
    end
end


@testitem "Octree constructors" setup = [TestData, CommonImports, OctreeTestData] begin
    using Random
    Random.seed!(606)

    # Test with BoundaryLayerSpacing (exercises _extract_min_spacing)
    mesh = OctreeTestData.unit_cube_mesh()
    bnd = PointBoundary(mesh)
    spacing = BoundaryLayerSpacing(
        WhatsThePoint.points(bnd);
        at_wall = 0.3m,
        bulk = 3.0m,
        layer_thickness = 1.5m
    )
    alg = Octree(mesh; spacing, alpha = 1.5)
    cloud = discretize(bnd, spacing; alg, max_points = 50)
    @test cloud isa PointCloud
    @test length(WhatsThePoint.volume(cloud)) > 0
    @test alg.node_min_ratio < 1.0

    # Test construction from an imported mesh
    alg2 = Octree(import_mesh(TestData.BOX_PATH, u"m"))
    @test alg2 isa Octree
    @test alg2.triangle_octree isa WhatsThePoint.TriangleOctree
end

@testitem "Octree discretization preserves and promotes mactype" setup = [TestData, CommonImports] begin
    # GeoIO's output mactype is version-dependent (newer Meshes/CoordRefSystems
    # promote binary STL to Float64 on load), so we explicitly rebuild the mesh
    # at a known mactype to pin both paths regardless of the resolved
    # dependency versions.
    mesh_raw = GeoIO.load(TestData.BOX_PATH).geometry
    rebuild(T, mesh) = Meshes.SimpleMesh(
        [
            (c = Meshes.coords(p); Meshes.Point(T(ustrip(c.x)), T(ustrip(c.y)), T(ustrip(c.z))))
                for p in Meshes.vertices(mesh)
        ],
        Meshes.topology(mesh),
    )
    mesh_f32 = rebuild(Float32, mesh_raw)
    bnd = PointBoundary(mesh_f32)
    @test CoordRefSystems.mactype(Meshes.crs(first(points(bnd)))) === Float32

    # Type-consistent path: a Float32 mesh drives a Float32 algorithm, and the
    # whole pipeline stays Float32 — no silent promotion to Float64.
    alg32 = Octree(mesh_f32)
    @test alg32.alpha isa Float32
    @test alg32.boundary_oversampling isa Float32
    @test alg32.bridson_factor isa Float32
    @test alg32.max_growth isa Float32
    @test eltype(alg32.triangle_octree.index.bbox_min) === Float32
    cloud32 = discretize(bnd, ConstantSpacing(3.0f0m); alg = alg32, max_points = 50)
    @test cloud32 isa PointCloud
    @test length(WhatsThePoint.volume(cloud32)) > 0
    @test CoordRefSystems.mactype(Meshes.crs(first(points(cloud32)))) === Float32

    # Mixed path: a Float32 boundary combined with a deliberately Float64
    # algorithm still assembles — the PointCloud constructor promotes boundary
    # and volume to their common type (Float32 + Float64 -> Float64).
    mesh_f64 = rebuild(Float64, mesh_raw)
    cloud_mixed = discretize(bnd, ConstantSpacing(3.0m); alg = Octree(mesh_f64), max_points = 50)
    @test length(WhatsThePoint.volume(cloud_mixed)) > 0
    @test CoordRefSystems.mactype(Meshes.crs(first(points(cloud_mixed)))) === Float64

    # Promotion preserves surface names (and the boundary point count).
    @test names(WhatsThePoint.boundary(cloud_mixed)) == names(bnd)
    @test length(WhatsThePoint.boundary(cloud_mixed)) == length(bnd)

    # Non-Octree algorithms are untouched: same Float32 boundary still works
    # through SlakKosec.
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    cloud_sk = discretize(bnd, ConstantSpacing(3.0f0m); alg = SlakKosec(octree), max_points = 20)
    @test cloud_sk isa PointCloud
end
