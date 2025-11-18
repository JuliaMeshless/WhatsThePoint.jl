using WhatsThePoint
using Meshes
using Random
using Unitful

N = 10

@testset "PointCloud with PointBoundary" begin
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test cloud.volume isa PointVolume
    @test WhatsThePoint.boundary(cloud)[:surface1] == b[:surface1]
end

@testset "PointCloud from file" begin
    cloud = PointCloud(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(cloud) == 24780
    @test hassurface(cloud, :surface1)
end

@testset "PointCloud from PointBoundary" begin
    cloud = PointCloud(PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl")))
    @test length(cloud) == 24780
    @test hassurface(cloud, :surface1)
end

@testset "Base Methods" begin
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test length(cloud) == N
    @test size(cloud) == (N,)

    points = rand(Point, N)
    surf = PointSurface(points)
    @test_throws ArgumentError cloud[:surface1] = surf
    cloud[:surface2] = surf
    @test cloud[:surface2] == surf

    # Test the iterate method
    @testset "iterate" begin
        points = rand(Point, N)
        b = PointBoundary(points)
        cloud = PointCloud(b)
        for (i, p) in enumerate(cloud)
            @test p.point == points[i]
        end
    end
end

@testset "generate_shadows" begin
    # Test that generate_shadows generates correct shadow positions (issue #50)
    # Use a 2D circle with known normals and radius for verification

    # Create 8 points on a unit circle (in meters)
    radius = 1.0u"m"
    circle_points = [Point(radius * cos(θ), radius * sin(θ)) for θ in 0:(π / 4):(7π / 4)]

    # Create a point cloud from the circle
    cloud = PointCloud(PointBoundary(circle_points))

    # Generate shadows with a known offset
    Δ = 0.1u"m"
    shadow = ShadowPoints(Δ)
    shadow_points = generate_shadows(cloud, shadow)

    # Verify the function returns correct type and length
    @test shadow_points isa Vector{<:Point}
    @test length(shadow_points) == length(circle_points)

    # Verify each shadow point is exactly Δ distance from its corresponding original point
    # Normals may point inward or outward (not oriented), so we just check distance
    for (i, orig_point) in enumerate(circle_points)
        sp = shadow_points[i]
        orig_coords = to(orig_point)
        shadow_coords = to(sp)

        # Calculate distance between original and shadow point
        dx = shadow_coords[1] - orig_coords[1]
        dy = shadow_coords[2] - orig_coords[2]
        distance = sqrt(dx^2 + dy^2)

        # Shadow should be exactly Δ away from original point
        @test distance ≈ Δ rtol=1e-6

        # Verify shadow point is radially aligned with original (on the same ray from origin)
        # Both points should have the same angle from origin
        orig_angle = atan(orig_coords[2], orig_coords[1])
        shadow_angle = atan(shadow_coords[2], shadow_coords[1])
        # Handle angle wrapping around ±π
        angle_diff = abs(orig_angle - shadow_angle)
        @test (angle_diff < 1e-6 || abs(angle_diff - 2π) < 1e-6 || abs(angle_diff - π) < 1e-6)
    end
end

@testset "normal and area functions" begin
    # Test that normal(cloud) and area(cloud) work correctly (issue #49)
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    @test_nowarn normal(cloud)
    @test_nowarn area(cloud)
    @test length(normal(cloud)) == N
    @test length(area(cloud)) == N
end
