using WhatsThePoint
using Meshes
using Unitful: m
using LinearAlgebra

@testset "repel! with default parameters" begin
    # Create a small cloud from box.stl with limited points
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    # Store original volume points
    original_points = deepcopy(cloud.volume.points)
    original_volume_length = length(cloud.volume)

    # Apply repel with default parameters but limited iterations
    conv = repel!(cloud, spacing; max_iters=10)

    # Test that convergence history was returned
    @test conv isa Vector{<:AbstractFloat}
    @test length(conv) <= 10

    # Test that cloud still has volume points
    @test length(cloud.volume) > 0

    # Test that points actually moved (at least some should have moved)
    new_points = cloud.volume.points
    if length(new_points) == length(original_points)
        # If same number of points, check that at least some moved
        moved = false
        for i in 1:min(length(new_points), length(original_points))
            if new_points[i] != original_points[i]
                moved = true
                break
            end
        end
        @test moved
    else
        # Points were filtered (some moved outside), which is valid behavior
        @test length(new_points) <= length(original_points)
    end
end

@testset "repel! with custom β parameter" begin
    # Create a small cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    original_volume_length = length(cloud.volume)

    # Test with different β values
    conv_low = repel!(cloud, spacing; β=0.1, max_iters=10)
    @test conv_low isa Vector
    @test length(cloud.volume) > 0

    # Test with higher β
    cloud2 = discretize(boundary, spacing; max_points=50)
    conv_high = repel!(cloud2, spacing; β=0.5, max_iters=10)
    @test conv_high isa Vector
    @test length(cloud2.volume) > 0
end

@testset "repel! with custom max_iters" begin
    # Create a small cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    # Test with very few iterations
    conv = repel!(cloud, spacing; max_iters=3)
    @test length(conv) <= 3
    @test conv isa Vector

    # Test with more iterations
    cloud2 = discretize(boundary, spacing; max_points=50)
    conv2 = repel!(cloud2, spacing; max_iters=10)
    @test length(conv2) <= 10
    @test conv2 isa Vector
end

@testset "repel! with custom tolerance" begin
    # Create a small cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    # Test with looser tolerance (should converge faster)
    conv_loose = repel!(cloud, spacing; tol=1e-3, max_iters=10)
    @test conv_loose isa Vector

    # Test with tighter tolerance (may not converge in 10 iterations)
    cloud2 = discretize(boundary, spacing; max_points=50)
    conv_tight = repel!(cloud2, spacing; tol=1e-9, max_iters=10)
    @test conv_tight isa Vector
end

@testset "repel! point movement verification" begin
    # Create a small cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=30)

    # Store original points
    original_points = collect(cloud.volume.points)
    original_count = length(original_points)

    @test original_count > 0

    # Apply repulsion
    conv = repel!(cloud, spacing; max_iters=5)

    # Get new points
    new_points = collect(cloud.volume.points)

    # Verify that points changed
    # Either some points moved, or some were filtered out
    points_identical = true
    for i in 1:min(length(original_points), length(new_points))
        if to(original_points[i]) != to(new_points[i])
            points_identical = false
            break
        end
    end

    # Points should have moved OR been filtered
    @test (!points_identical) || (length(new_points) != length(original_points))

    # Convergence should show some change occurred
    @test length(conv) > 0
end

@testset "repel! filters points outside domain" begin
    # Create a small cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=50)

    original_count = length(cloud.volume)

    # Apply repulsion - points that move outside should be filtered
    repel!(cloud, spacing; max_iters=10)

    # All remaining points should be inside the domain
    for p in cloud.volume.points
        @test isinside(p, cloud)
    end

    # Volume count should be <= original (some may have been filtered)
    @test length(cloud.volume) <= original_count
end

@testset "repel! convergence behavior" begin
    # Create a small cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)
    cloud = discretize(boundary, spacing; max_points=30)

    # Apply repulsion and check convergence values
    conv = repel!(cloud, spacing; max_iters=10, tol=1e-6)

    # Convergence history should be monotonically improving (generally decreasing)
    # or at least bounded
    @test all(c -> c >= 0, conv)

    # All convergence values should be finite
    @test all(isfinite, conv)
end

@testset "repel! with multiple parameter combinations" begin
    # Test various parameter combinations work without error
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    spacing = ConstantSpacing(1.0m)

    # Combination 1: Low β, tight tolerance, few iterations
    cloud1 = discretize(boundary, spacing; max_points=30)
    conv1 = @test_nowarn repel!(cloud1, spacing; β=0.1, tol=1e-7, max_iters=5)
    @test conv1 isa Vector

    # Combination 2: High β, loose tolerance, more iterations
    cloud2 = discretize(boundary, spacing; max_points=30)
    conv2 = @test_nowarn repel!(cloud2, spacing; β=0.4, tol=1e-4, max_iters=10)
    @test conv2 isa Vector

    # Combination 3: Default β, custom tolerance and iterations
    cloud3 = discretize(boundary, spacing; max_points=30)
    conv3 = @test_nowarn repel!(cloud3, spacing; tol=1e-5, max_iters=8)
    @test conv3 isa Vector
end
