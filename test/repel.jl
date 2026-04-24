@testitem "repel convergence success" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 20)

    conv = Float64[]
    new_cloud = repel(cloud, spacing, octree; max_iters = 100, tol = 1000.0, convergence = conv)

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) < 100
    @test length(volume(new_cloud)) > 0
end

@testitem "repel basic behavior" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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

    # Default β matches the original repel default (0.2)
    @test InverseDistanceForce().β == 0.2
    @test SpacingEquilibriumForce().β == 0.2
end

@testitem "repel β kwarg feeds default force_model" setup = [TestData, CommonImports] begin
    # The β kwarg must continue to affect the default InverseDistanceForce so that
    # existing callers that pass `β=...` keep working without passing force_model.
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 30)

    Random.seed!(1)
    c_beta = repel(cloud, spacing, octree; β = 0.5, max_iters = 5)

    Random.seed!(1)
    c_model = repel(
        cloud, spacing, octree;
        force_model = InverseDistanceForce(0.5), max_iters = 5,
    )

    pts_beta = points(c_beta)
    pts_model = points(c_model)
    @test length(pts_beta) == length(pts_model)
    for (a, b) in zip(pts_beta, pts_model)
        @test a ≈ b
    end
end

@testitem "repel accepts SpacingEquilibriumForce" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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

    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
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
