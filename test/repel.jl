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
