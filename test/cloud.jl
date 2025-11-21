using WhatsThePoint
using Meshes
using Random
using Unitful: @u_str, m
using OrderedCollections: LittleDict

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

@testset "to functions" begin
    # Test to(cloud::PointCloud)
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    coords = to(cloud)
    @test coords isa Vector
    @test length(coords) == N
    @test all(c -> c isa Vector, coords)

    # Test to(surfaces::LittleDict)
    surf1 = PointSurface(rand(Point, 5))
    surf2 = PointSurface(rand(Point, 7))
    surfaces_dict = LittleDict(:surf1 => surf1, :surf2 => surf2)
    coords_from_dict = to(surfaces_dict)
    @test coords_from_dict isa Vector
    @test length(coords_from_dict) == 12  # 5 + 7
end

@testset "accessor functions" begin
    # Test boundary(cloud), volume(cloud), surfaces(cloud)
    points = rand(Point, N)
    b = PointBoundary(points)
    cloud = PointCloud(b)

    @test boundary(cloud) isa PointBoundary
    @test boundary(cloud) === cloud.boundary

    @test volume(cloud) isa PointVolume
    @test volume(cloud) === cloud.volume

    @test surfaces(cloud) isa LittleDict
    @test :surface1 in keys(surfaces(cloud))
end

@testset "hassurface function" begin
    points = rand(Point, N)
    b = PointBoundary(points)
    cloud = PointCloud(b)

    # Test with existing surface
    @test hassurface(cloud, :surface1) == true

    # Test with non-existing surface
    @test hassurface(cloud, :nonexistent) == false

    # Add a new surface and test
    cloud[:newsurface] = PointSurface(rand(Point, 5))
    @test hassurface(cloud, :newsurface) == true
end

@testset "Meshes interface functions" begin
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    # Test Meshes.pointify
    pts = Meshes.pointify(cloud)
    @test pts isa Vector{<:Point}
    @test length(pts) == N

    # Test Meshes.nelements
    n_elems = Meshes.nelements(cloud)
    @test n_elems isa Int
    @test n_elems == N

    # Test Meshes.boundingbox
    bbox = Meshes.boundingbox(cloud)
    @test bbox isa Box

    # Test with cloud that has volume points
    # Add some volume points by creating a new PointVolume
    vol_points = rand(Point, 5)
    cloud_with_vol = PointCloud(PointBoundary(points))
    # Manually add volume points for testing
    cloud_with_vol.volume = PointVolume(vol_points)

    pts_total = Meshes.pointify(cloud_with_vol)
    @test length(pts_total) == N + 5

    n_elems_total = Meshes.nelements(cloud_with_vol)
    @test n_elems_total == N + 5

    bbox_total = Meshes.boundingbox(cloud_with_vol)
    @test bbox_total isa Box
end

@testset "names function" begin
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    @test names(cloud) == [:surface1]

    # Add another surface
    cloud[:surface2] = PointSurface(rand(Point, 3))
    surface_names = names(cloud)
    @test :surface1 in surface_names
    @test :surface2 in surface_names
    @test length(surface_names) == 2
end

@testset "Pretty Printing" begin
    # Test boundary-only cloud
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud)
    output = String(take!(io))
    @test contains(output, "PointCloud")
    @test contains(output, "$(N) points")
    @test contains(output, "Boundary: $(N) points")
    @test contains(output, "surface1")

    # Test cloud with volume points
    vol_points = rand(Point, 5)
    cloud_with_vol = PointCloud(PointBoundary(points))
    cloud_with_vol.volume = PointVolume(vol_points)
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud_with_vol)
    output = String(take!(io))
    @test contains(output, "PointCloud")
    @test contains(output, "$(N + 5) points")
    @test contains(output, "Boundary: $(N) points")
    @test contains(output, "Volume: 5 points")

    # Test cloud with multiple surfaces
    cloud_multi = PointCloud(PointBoundary(points))
    cloud_multi[:surface2] = PointSurface(rand(Point, N))
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud_multi)
    output = String(take!(io))
    @test contains(output, "surface1")
    @test contains(output, "surface2")
end
