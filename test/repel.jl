@testitem "repel convergence success" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 20)

    conv = Float64[]
    new_cloud = repel(cloud, spacing, octree; max_iters = 100, tol = 1.0e6, convergence = conv)

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) < 100
    @test length(volume(new_cloud)) > 0
end

@testitem "repel basic behavior" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    original_vol_count = length(volume(cloud))
    original_total = length(cloud)

    conv = Float64[]
    new_cloud = repel(cloud, spacing, octree; max_iters = 10, convergence = conv)

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 10
    @test length(volume(new_cloud)) > 0

    # Total point count is preserved (no points lost)
    @test length(new_cloud) == original_total

    # All volume points inside domain
    for p in volume(new_cloud).points
        @test isinside(p, octree)
    end

    @test all(c -> c >= 0, conv)
    @test all(isfinite, conv)
end

@testitem "repel respects max_iters" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)

    conv1 = Float64[]
    cloud1 = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)
    repel(cloud1, spacing, octree; max_iters = 3, convergence = conv1)
    @test length(conv1) <= 3

    conv2 = Float64[]
    cloud2 = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)
    repel(cloud2, spacing, octree; max_iters = 10, convergence = conv2)
    @test length(conv2) <= 10
end

@testitem "repel accepts parameter combinations" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 30)

    conv = Float64[]
    new_cloud = repel(
        cloud, spacing, octree; β = 0.3, tol = 1.0e-5, max_iters = 5, convergence = conv
    )

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 5
    @test length(volume(new_cloud)) > 0
end

@testitem "repel boundary projection preserves points" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    original_total = length(cloud)

    # Strong repulsion to stress-test projection
    new_cloud = repel(cloud, spacing, octree; β = 0.1, α = 0.5, max_iters = 20)

    # No points lost — projection keeps them in domain
    @test length(new_cloud) == original_total
    @test length(volume(new_cloud)) > 0

    # All volume points inside
    for p in volume(new_cloud).points
        @test isinside(p, octree)
    end

    # Boundary has valid normals
    bnd = WhatsThePoint.boundary(new_cloud)
    normals = normal(bnd)
    for n in normals
        @test norm(n) > 0.99
        @test isfinite(norm(n))
    end
end

@testitem "repel without octree" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    conv = Float64[]
    new_cloud = repel(cloud, spacing; max_iters = 3, convergence = conv)

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 3
end

@testitem "force model compute_force values" setup = [CommonImports] begin
    # InverseDistanceForce reproduces F(u) = 1/(u²+β)² exactly
    m1 = InverseDistanceForce(0.2)
    for u in (0.0, 0.5, 1.0, 2.0)
        expected = 1 / (u^2 + 0.2)^2
        @test compute_force(m1, u) ≈ expected
    end

    # SpacingEquilibriumForce has zero at u=1, positive for u<1, negative for u>1
    m2 = SpacingEquilibriumForce(0.2)
    @test compute_force(m2, 1.0) == 0.0
    @test compute_force(m2, 0.5) > 0
    @test compute_force(m2, 2.0) < 0

    # ClippedSpacingForce matches the repulsive branch of SpacingEquilibriumForce
    # for u < u0 and is exactly zero from u0 on (compact support).
    m3 = ClippedSpacingForce(0.2)
    @test m3.u0 == 1.0
    for u in (0.0, 0.3, 0.7, 0.99)
        @test compute_force(m3, u) ≈ compute_force(m2, u)
        @test compute_force(m3, u) > 0
    end
    @test compute_force(m3, 1.0) == 0.0
    @test compute_force(m3, 1.5) == 0.0
    @test compute_force(m3, 10.0) == 0.0
    # custom support radius moves the root
    m4 = ClippedSpacingForce(0.2, 0.8)
    @test compute_force(m4, 0.79) > 0
    @test compute_force(m4, 0.8) == 0.0

    # Default β matches the original repel default (0.2)
    @test InverseDistanceForce().β == 0.2
    @test SpacingEquilibriumForce().β == 0.2
    @test ClippedSpacingForce().β == 0.2

    # StrongSpacingForce: zero at u=1, positive for u<1, negative for u>1,
    # stronger repulsive core than SpacingEquilibriumForce at small u.
    m5 = StrongSpacingForce(0.2, 3.0)
    @test m5.γ == 3.0
    @test compute_force(m5, 1.0) == 0.0
    @test compute_force(m5, 0.5) > 0
    @test compute_force(m5, 2.0) < 0
    for u in (0.0, 0.5, 1.0, 2.0)
        expected = (1 - u^2) / (u^2 + 0.2)^3
        @test compute_force(m5, u) ≈ expected
    end
    # γ=2 recovers SpacingEquilibriumForce
    m6 = StrongSpacingForce(0.2, 2.0)
    for u in (0.0, 0.5, 1.0, 2.0)
        @test compute_force(m6, u) ≈ compute_force(m2, u)
    end
    # Default γ is 3.0
    @test StrongSpacingForce().γ == 3.0
    @test StrongSpacingForce(0.5).γ == 3.0

    # One-arg constructor preserves β's machine type instead of promoting to
    # Float64 (regression: `promote(β, 3.0)` silently stored Float32 β as
    # Float64 — the old `m7.β == 0.5f0` check passed numerically anyway).
    m7 = StrongSpacingForce(0.5f0)
    @test m7 isa StrongSpacingForce{Float32}
    @test m7.β == 0.5f0 && m7.γ == 3.0f0
    m8 = StrongSpacingForce(1)
    @test m8 isa StrongSpacingForce{Int}
    @test m8.β == 1 && m8.γ == 3
    @test ClippedSpacingForce(0.5f0) isa ClippedSpacingForce{Float32}
    @test ClippedSpacingForce(0.5f0).u0 == 1.0f0
end

@testitem "repel β kwarg feeds default force_model" setup = [TestData, CommonImports] begin
    # The β kwarg must continue to affect the default ClippedSpacingForce so that
    # existing callers that pass `β=...` keep working without passing force_model.
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 30)

    Random.seed!(1)
    c_beta = repel(cloud, spacing, octree; β = 0.5, max_iters = 5)

    Random.seed!(1)
    c_model = repel(
        cloud, spacing, octree;
        force_model = ClippedSpacingForce(0.5), max_iters = 5,
    )

    pts_beta = points(c_beta)
    pts_model = points(c_model)
    @test length(pts_beta) == length(pts_model)
    for (a, b) in zip(pts_beta, pts_model)
        @test a ≈ b
    end
end

@testitem "repel accepts SpacingEquilibriumForce" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    original_total = length(cloud)

    conv = Float64[]
    new_cloud = repel(
        cloud, spacing, octree;
        force_model = SpacingEquilibriumForce(0.2),
        max_iters = 20, convergence = conv,
    )

    @test length(new_cloud) == original_total
    @test length(volume(new_cloud)) > 0
    @test all(isfinite, conv)
    for p in volume(new_cloud).points
        @test isinside(p, octree)
    end
end

@testitem "repel force models both reduce spacing error" setup = [TestData, CommonImports] begin
    using WhatsThePoint: spacing_metrics

    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary; divisor = 4)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 80)

    before = spacing_metrics(cloud, spacing; k = 10)

    c_inv = repel(
        cloud, spacing, octree;
        force_model = InverseDistanceForce(0.2), max_iters = 40,
    )
    c_eq = repel(
        cloud, spacing, octree;
        force_model = SpacingEquilibriumForce(0.2), max_iters = 40,
    )

    after_inv = spacing_metrics(c_inv, spacing; k = 10)
    after_eq = spacing_metrics(c_eq, spacing; k = 10)

    # Both force models should leave the cloud at least as well-spaced as the
    # initial discretization. This is a weak invariant — stronger claims about
    # which model wins depend on α/β tuning and iteration count.
    @test after_inv.mean_error <= before.mean_error * 1.1
    @test after_eq.mean_error <= before.mean_error * 1.1
end

@testitem "repel stall_after stops on quality plateau" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    # The force residual of a saturated repulsion-only packing plateaus instead
    # of reaching tol, so a tight-tol run burns the whole budget — stall_after
    # must end it once the d_NN/s CV stops improving.
    conv = Float64[]
    repel(
        cloud, spacing, octree;
        max_iters = 200, tol = 1.0e-12, stall_after = 5, convergence = conv,
    )
    @test 5 < length(conv) < 200

    # cv_target: a generous target must stop the relaxation almost immediately
    # (the monitor sees the d_NN/s CV of the movable points each sweep), and —
    # because the monitor measures the pre-sweep snapshot — the cloud must come
    # back untouched, not one-sweep-moved.
    conv_t = Float64[]
    c_t = repel(
        cloud, spacing, octree;
        max_iters = 200, tol = 1.0e-12, cv_target = 10.0, convergence = conv_t,
    )
    @test length(conv_t) == 1
    for (a, b) in zip(points(c_t), points(cloud))
        @test a ≈ b
    end

    # stall_after = 0 disables the quality stop: the run burns the whole budget.
    conv_off = Float64[]
    repel(
        cloud, spacing, octree;
        max_iters = 30, tol = 1.0e-12, stall_after = 0, convergence = conv_off,
    )
    @test length(conv_off) == 30
end

@testitem "near-duplicate cull mask" setup = [CommonImports] begin
    # Five collinear points; the 2.0 / 2.01 pair is a near-duplicate (gap 0.01 ≪ 0.5·s).
    pts = [
        Meshes.Point(0.0, 0.0, 0.0), Meshes.Point(1.0, 0.0, 0.0),
        Meshes.Point(2.0, 0.0, 0.0), Meshes.Point(2.01, 0.0, 0.0),
        Meshes.Point(3.0, 0.0, 0.0),
    ]
    spacings = fill(1.0, 5)

    keep = WhatsThePoint._near_duplicate_keep_mask(pts, spacings, 0.5)
    @test count(keep) == 4        # exactly one of the near-duplicate pair removed
    @test keep[3] && !keep[4]     # lower index of the pair is kept

    # ratio = 0 is a no-op (culling disabled).
    @test all(WhatsThePoint._near_duplicate_keep_mask(pts, spacings, 0.0))

    # A cluster larger than any fixed-k neighborhood: the mask must still
    # guarantee no kept pair is closer than ratio·spacing, so the whole
    # cluster collapses to its lowest-indexed point.
    cluster = [Meshes.Point(10.0 + 1.0e-3 * i, 0.0, 0.0) for i in 1:12]
    cpts = vcat(pts, cluster)
    ckeep = WhatsThePoint._near_duplicate_keep_mask(cpts, fill(1.0, length(cpts)), 0.5)
    @test count(ckeep[6:end]) == 1
    @test ckeep[6]
end

@testitem "repel deposit_ratio grows the boundary from escaped points" setup = [TestData, CommonImports] begin
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    full = PointBoundary(TestData.BOX_PATH, u"m")
    # Deliberately sparse boundary: deposition must grow the surface population
    # until it can contain the interior at the prescribed density.
    ids = 1:200:length(full)
    surf = PointSurface(
        points(full)[ids], collect(normal(full)[ids]), collect(area(full)[ids])
    )
    sparse_bnd = PointBoundary(LittleDict{Symbol, typeof(surf)}(:boundary => surf))
    n_sparse = length(sparse_bnd)

    # Spacing sized so the volume fills the box at the point budget — deposition
    # needs interior pressure against the wall, not an under-filled cloud
    # expanding inward. The sparse boundary sits at ~3.1 m, comparable to h.
    spacing = ConstantSpacing(3.0f0 * m)
    cloud = discretize(sparse_bnd, spacing; alg = SlakKosec(octree), max_points = 600)

    new_cloud = repel(cloud, spacing, octree; max_iters = 30, deposit_ratio = 0.5)

    # Deposition converts points (total conserved) and grows the boundary.
    @test length(new_cloud) == length(cloud)
    @test length(WhatsThePoint.boundary(new_cloud)) > n_sparse

    # The cv_target stop is checked before deposition: a run stopping on its
    # first iteration must return the pre-sweep configuration exactly — same
    # boundary membership, same positions. A deposit leaking through the stop
    # would grow the boundary with a point whose reverted position floats in
    # the interior.
    stopped = repel(
        cloud, spacing, octree;
        max_iters = 30, deposit_ratio = 0.5, cv_target = 10.0,
    )
    @test length(WhatsThePoint.boundary(stopped)) == n_sparse
    for (a, b) in zip(points(stopped), points(cloud))
        @test a ≈ b
    end

    # Deposited points carry valid triangle normals.
    for n in normal(WhatsThePoint.boundary(new_cloud))
        @test isfinite(norm(n)) && norm(n) > 0.99
    end

    # Off by default: boundary count unchanged.
    no_dep = repel(cloud, spacing, octree; max_iters = 5)
    @test length(WhatsThePoint.boundary(no_dep)) == n_sparse
end

@testitem "repel cull_ratio enforces the separation guarantee" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    # Spacing must be consistent with the boundary tessellation: box.stl's face
    # centers sit ~0.22 m apart, while bbox/8 ≈ 5.4 m would put the 0.5·h cull
    # radius across ~12 boundary spacings — asking the cull to decimate the
    # boundary instead of removing stray near-duplicate pairs.
    spacing = ConstantSpacing(0.25f0 * m)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 60)

    nocull = repel(cloud, spacing, octree; max_iters = 20)
    culled = repel(cloud, spacing, octree; max_iters = 20, cull_ratio = 0.5)

    # Culling never grows the cloud.
    @test length(culled) <= length(nocull)

    # No kept pair is closer than the cull threshold (0.5·spacing).
    # NB: don't name this `m` — CommonImports brings in `Unitful.m`, and
    # assigning to an imported name is an error.
    culled_metrics = metrics(culled; k = 2)
    smin = minimum(ustrip.(spacing.(points(culled))))
    @test ustrip(culled_metrics.separation) >= 0.5 * smin * (1 - 1.0e-6)
end

@testitem "repel kick_after breaks frozen standoffs" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 30)

    Random.seed!(42)
    # kick_after should not crash and should produce a valid cloud
    kicked = repel(cloud, spacing, octree; max_iters = 30, kick_after = 5)
    @test kicked isa PointCloud
    @test length(WhatsThePoint.volume(kicked)) > 0

    # Without kick: same seed, same iterations — both valid
    Random.seed!(42)
    nokick = repel(cloud, spacing, octree; max_iters = 30, kick_after = 0)
    @test nokick isa PointCloud
end

@testitem "repel supports 2D clouds" setup = [CommonImports] begin
    # The volume-only method dispatches on 𝔼{N}; the random directions in
    # _maybe_kick! and _safe_direction must match the cloud's dimension
    # (regression: both built 3D vectors and crashed any 2D repel that kicked
    # or hit a coincident pair).
    corners = Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = PointBoundary(corners)
    spacing = ConstantSpacing(0.2m)
    cloud = discretize(boundary, spacing; alg = FornbergFlyer(), max_points = 40)

    # kick_after = 1 forces a kick on the first iteration.
    Random.seed!(7)
    relaxed = repel(cloud, spacing; max_iters = 3, kick_after = 1)
    @test relaxed isa PointCloud
    @test Meshes.embeddim(first(points(relaxed))) == 2
end

@testitem "repel trace records closest pair" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 30)

    Random.seed!(1)
    traces = NamedTuple[]
    c = repel(cloud, spacing; max_iters = 5, trace = traces)
    @test length(traces) == 5
    for t in traces
        @test hasproperty(t, :idx_a)
        @test hasproperty(t, :idx_b)
        @test hasproperty(t, :r_over_s)
        @test t.r_over_s > 0
    end
end

@testitem "repel rebuild_every skips k-NN rebuild" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
    octree = TriangleOctree(import_mesh(TestData.BOX_PATH, u"m"); classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 30)

    Random.seed!(1)
    c1 = repel(cloud, spacing; max_iters = 10, rebuild_every = 1)
    Random.seed!(1)
    c3 = repel(cloud, spacing; max_iters = 10, rebuild_every = 3)
    @test c1 isa PointCloud
    @test c3 isa PointCloud
    @test length(WhatsThePoint.volume(c3)) > 0
    # rebuild_every must reject < 1
    @test_throws ArgumentError repel(cloud, spacing; rebuild_every = 0)
end
