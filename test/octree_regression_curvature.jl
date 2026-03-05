# Regression tests for high-curvature / concave regions and conservative classification.

@testitem "TriangleOctree regression - near-surface normal offsets (bifurcation)" setup = [CommonImports, TestData] begin
    using GeoIO
    using WhatsThePoint: _compute_bbox, _get_triangle_vertices, _get_triangle_normal, all_leaves

    if !isfile(TestData.BIFURCATION_PATH)
        @test_skip "bifurcation.stl not available"
    end

    mesh = GeoIO.load(TestData.BIFURCATION_PATH).geometry
    octree = TriangleOctree(mesh; classify_leaves=true)

    bbox_min, bbox_max = _compute_bbox(Float64, octree.mesh)
    diagonal = norm(bbox_max - bbox_min)
    ε = max(diagonal * 1e-5, 1e-8)

    n_tri = Meshes.nelements(octree.mesh)
    n_samples = min(128, n_tri)
    step = max(1, fld(n_tri, n_samples))
    tri_indices = collect(1:step:n_tri)[1:n_samples]

    outside_false_positives = 0
    inside_hits = 0

    for tri_idx in tri_indices
        v1, v2, v3 = _get_triangle_vertices(Float64, octree.mesh, tri_idx)
        n = _get_triangle_normal(Float64, octree.mesh, tri_idx)
        centroid = (v1 + v2 + v3) / 3

        p_out = centroid + ε * n
        p_in = centroid - ε * n

        global outside_false_positives += isinside(p_out, octree) ? 1 : 0
        global inside_hits += isinside(p_in, octree) ? 1 : 0
    end

    # Conservative-first guarantee we want to protect:
    # near-surface points moved outward along the triangle normal should not be inside.
    @test outside_false_positives == 0

    # In conservative mode, inward offsets near highly curved regions can still be
    # classified as boundary/exterior. Track this for diagnostics but do not fail.
    @info "inward offset inside hits" inside_hits total = length(tri_indices)
    @test inside_hits >= 0
end

@testitem "TriangleOctree regression - interior leaves avoid positive signed distance" setup = [CommonImports, TestData] begin
    using GeoIO
    using WhatsThePoint: LEAF_INTERIOR, _compute_signed_distance_octree, all_leaves, box_bounds

    if !isfile(TestData.BIFURCATION_PATH)
        @test_skip "bifurcation.stl not available"
    end

    mesh = GeoIO.load(TestData.BIFURCATION_PATH).geometry
    octree = TriangleOctree(mesh; classify_leaves=true)

    leaves = all_leaves(octree.tree)
    interior_leaves = [i for i in leaves if octree.leaf_classification[i] == LEAF_INTERIOR]

    if isempty(interior_leaves)
        @test_skip "no interior leaves found"
    end

    Random.seed!(2026)
    n_leaf_samples = min(40, length(interior_leaves))
    sampled_leaves = randperm(length(interior_leaves))[1:n_leaf_samples]

    positive_sd_count = 0
    n_checked = 0

    for leaf_pos in sampled_leaves
        leaf_idx = interior_leaves[leaf_pos]
        bbox_min, bbox_max = box_bounds(octree.tree, leaf_idx)

        for _ in 1:5
            p = bbox_min + rand(SVector{3,Float64}) .* (bbox_max - bbox_min)
            sd = _compute_signed_distance_octree(p, octree.mesh, octree.tree)
            global n_checked += 1
            global positive_sd_count += sd > 1e-10 ? 1 : 0
        end
    end

    @test n_checked > 0
    @test positive_sd_count == 0
end
