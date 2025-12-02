@testitem "PointCloud with PointBoundary" setup = [TestData, CommonImports] begin
    N = 10
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test cloud.volume isa PointVolume
    @test WhatsThePoint.boundary(cloud)[:surface1] == b[:surface1]
end

@testitem "PointCloud from file" setup = [TestData, CommonImports] begin
    cloud = PointCloud(TestData.BIFURCATION_PATH)
    @test length(cloud) == 24780
    @test hassurface(cloud, :surface1)
end

@testitem "PointCloud from PointBoundary file" setup = [TestData, CommonImports] begin
    cloud = PointCloud(PointBoundary(TestData.BIFURCATION_PATH))
    @test length(cloud) == 24780
    @test hassurface(cloud, :surface1)
end

@testitem "PointCloud Base Methods" setup = [TestData, CommonImports] begin
    N = 10
    b = PointBoundary(rand(Point, N))
    cloud = PointCloud(b)
    @test length(cloud) == N
    @test size(cloud) == (N,)

    points = rand(Point, N)
    surf = PointSurface(points)
    @test_throws ArgumentError cloud[:surface1] = surf
    cloud[:surface2] = surf
    @test cloud[:surface2] == surf

    @testset "iterate" begin
        points = rand(Point, N)
        b = PointBoundary(points)
        cloud = PointCloud(b)
        for (i, p) in enumerate(cloud)
            @test p.point == points[i]
        end
    end
end

@testitem "PointCloud generate_shadows" setup = [TestData, CommonImports] begin
    radius = 1.0u"m"
    circle_points = [Point(radius * cos(θ), radius * sin(θ)) for θ in 0:(π / 4):(7π / 4)]

    cloud = PointCloud(PointBoundary(circle_points))

    Δ = 0.1u"m"
    shadow = ShadowPoints(Δ)
    shadow_points = generate_shadows(cloud, shadow)

    @test shadow_points isa Vector{<:Point}
    @test length(shadow_points) == length(circle_points)

    for (i, orig_point) in enumerate(circle_points)
        sp = shadow_points[i]
        orig_coords = to(orig_point)
        shadow_coords = to(sp)

        dx = shadow_coords[1] - orig_coords[1]
        dy = shadow_coords[2] - orig_coords[2]
        distance = sqrt(dx^2 + dy^2)

        @test distance ≈ Δ rtol = 1e-6

        orig_angle = atan(orig_coords[2], orig_coords[1])
        shadow_angle = atan(shadow_coords[2], shadow_coords[1])
        angle_diff = abs(orig_angle - shadow_angle)
        @test (
            angle_diff < 1e-6 || abs(angle_diff - 2π) < 1e-6 || abs(angle_diff - π) < 1e-6
        )
    end
end

@testitem "PointCloud normal() and area()" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    @test_nowarn normal(cloud)
    @test_nowarn area(cloud)
    @test length(normal(cloud)) == N
    @test length(area(cloud)) == N
end

@testitem "PointCloud to()" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    coords = to(cloud)
    @test coords isa Vector
    @test length(coords) == N
    @test all(c -> c isa AbstractVector, coords)

    surf1 = PointSurface(rand(Point, 5))
    surf2 = PointSurface(rand(Point, 7))
    surfaces_dict = LittleDict(:surf1 => surf1, :surf2 => surf2)
    coords_from_dict = to(surfaces_dict)
    @test coords_from_dict isa Vector
    @test length(coords_from_dict) == 12
end

@testitem "PointCloud accessor functions" setup = [TestData, CommonImports] begin
    using WhatsThePoint: boundary
    N = 10
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

@testitem "PointCloud hassurface()" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    b = PointBoundary(points)
    cloud = PointCloud(b)

    @test hassurface(cloud, :surface1) == true
    @test hassurface(cloud, :nonexistent) == false

    cloud[:newsurface] = PointSurface(rand(Point, 5))
    @test hassurface(cloud, :newsurface) == true
end

@testitem "PointCloud Meshes interface" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    pts = Meshes.pointify(cloud)
    @test pts isa Vector{<:Point}
    @test length(pts) == N

    n_elems = Meshes.nelements(cloud)
    @test n_elems isa Int
    @test n_elems == N

    bbox = Meshes.boundingbox(cloud)
    @test bbox isa Box

    vol_points = rand(Point, 5)
    cloud_with_vol = PointCloud(PointBoundary(points))
    cloud_with_vol.volume = PointVolume(vol_points)

    pts_total = Meshes.pointify(cloud_with_vol)
    @test length(pts_total) == N + 5

    n_elems_total = Meshes.nelements(cloud_with_vol)
    @test n_elems_total == N + 5

    bbox_total = Meshes.boundingbox(cloud_with_vol)
    @test bbox_total isa Box
end

@testitem "PointCloud names()" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    @test names(cloud) == [:surface1]

    cloud[:surface2] = PointSurface(rand(Point, 3))
    surface_names = names(cloud)
    @test :surface1 in surface_names
    @test :surface2 in surface_names
    @test length(surface_names) == 2
end

@testitem "PointCloud Pretty Printing" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud)
    output = String(take!(io))
    @test contains(output, "PointCloud")
    @test contains(output, "$(N) points")
    @test contains(output, "Boundary: $(N) points")
    @test contains(output, "surface1")

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

    cloud_multi = PointCloud(PointBoundary(points))
    cloud_multi[:surface2] = PointSurface(rand(Point, N))
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud_multi)
    output = String(take!(io))
    @test contains(output, "surface1")
    @test contains(output, "surface2")
end
