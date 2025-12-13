@testitem "repel with default parameters" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    original_points = deepcopy(collect(volume(cloud).points))
    original_volume_length = length(volume(cloud))

    new_cloud, conv = repel(cloud, spacing; max_iters=10)

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 10

    @test length(volume(new_cloud)) > 0

    new_points = collect(volume(new_cloud).points)
    if length(new_points) == length(original_points)
        moved = any(i -> new_points[i] != original_points[i], 1:length(new_points))
        @test moved
    else
        @test length(new_points) <= length(original_points)
    end
end

@testitem "repel with custom β parameter" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    new_cloud, conv_low = repel(cloud, spacing; β=0.1, max_iters=10)
    @test conv_low isa Vector
    @test length(volume(new_cloud)) > 0

    cloud2 = discretize(boundary, spacing; max_points=50)
    new_cloud2, conv_high = repel(cloud2, spacing; β=0.5, max_iters=10)
    @test conv_high isa Vector
    @test length(volume(new_cloud2)) > 0
end

@testitem "repel with custom max_iters" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    new_cloud, conv = repel(cloud, spacing; max_iters=3)
    @test length(conv) <= 3
    @test conv isa Vector

    cloud2 = discretize(boundary, spacing; max_points=50)
    new_cloud2, conv2 = repel(cloud2, spacing; max_iters=10)
    @test length(conv2) <= 10
    @test conv2 isa Vector
end

@testitem "repel with custom tolerance" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    new_cloud, conv_loose = repel(cloud, spacing; tol=1e-3, max_iters=10)
    @test conv_loose isa Vector

    cloud2 = discretize(boundary, spacing; max_points=50)
    new_cloud2, conv_tight = repel(cloud2, spacing; tol=1e-9, max_iters=10)
    @test conv_tight isa Vector
end

@testitem "repel point movement verification" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=30)

    original_points = collect(volume(cloud).points)
    original_count = length(original_points)

    @test original_count > 0

    new_cloud, conv = repel(cloud, spacing; max_iters=5)

    new_points = collect(volume(new_cloud).points)

    n = min(length(original_points), length(new_points))
    points_moved = any(i -> to(original_points[i]) != to(new_points[i]), 1:n)

    @test points_moved || (length(new_points) != length(original_points))

    @test length(conv) > 0
end

@testitem "repel filters points outside domain" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    original_count = length(volume(cloud))

    new_cloud, _ = repel(cloud, spacing; max_iters=10)

    for p in volume(new_cloud).points
        @test isinside(p, new_cloud)
    end

    @test length(volume(new_cloud)) <= original_count
end

@testitem "repel convergence behavior" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=30)

    new_cloud, conv = repel(cloud, spacing; max_iters=10, tol=1e-6)

    @test all(c -> c >= 0, conv)

    @test all(isfinite, conv)
end

@testitem "repel with multiple parameter combinations" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)

    cloud1 = discretize(boundary, spacing; max_points=30)
    new_cloud1, conv1 = repel(cloud1, spacing; β=0.1, tol=1e-7, max_iters=5)
    @test conv1 isa Vector

    cloud2 = discretize(boundary, spacing; max_points=30)
    new_cloud2, conv2 = repel(cloud2, spacing; β=0.4, tol=1e-4, max_iters=10)
    @test conv2 isa Vector

    cloud3 = discretize(boundary, spacing; max_points=30)
    new_cloud3, conv3 = repel(cloud3, spacing; tol=1e-5, max_iters=8)
    @test conv3 isa Vector
end
