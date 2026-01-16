@testitem "repel basic behavior" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points = 50)

    original_points = deepcopy(collect(volume(cloud).points))
    original_count = length(volume(cloud))

    new_cloud, conv = repel(cloud, spacing; max_iters = 10)

    # Return type checks
    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 10

    # Volume is non-empty
    @test length(volume(new_cloud)) > 0

    # Points moved or count changed
    new_points = collect(volume(new_cloud).points)
    if length(new_points) == length(original_points)
        moved = any(i -> to(new_points[i]) != to(original_points[i]), 1:length(new_points))
        @test moved
    else
        @test length(new_points) <= length(original_points)
    end

    # All points inside domain
    for p in volume(new_cloud).points
        @test isinside(p, new_cloud)
    end
    @test length(volume(new_cloud)) <= original_count

    # Convergence values are valid
    @test all(c -> c >= 0, conv)
    @test all(isfinite, conv)
end

@testitem "repel respects max_iters" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)

    cloud1 = discretize(boundary, spacing; max_points = 50)
    _, conv1 = repel(cloud1, spacing; max_iters = 3)
    @test length(conv1) <= 3

    cloud2 = discretize(boundary, spacing; max_points = 50)
    _, conv2 = repel(cloud2, spacing; max_iters = 10)
    @test length(conv2) <= 10
end

@testitem "repel accepts parameter combinations" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points = 30)

    # Test all parameters together in one call
    new_cloud, conv = repel(cloud, spacing; β = 0.3, tol = 1.0e-5, max_iters = 5)

    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 5
    @test length(volume(new_cloud)) > 0
end
