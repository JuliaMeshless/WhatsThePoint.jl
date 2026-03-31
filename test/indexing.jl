@testitem "surface_offset" setup = [CommonImports] begin
    N = 10
    pts1 = rand(Point, N)
    pts2 = rand(Point, N ÷ 2)
    b = PointBoundary(pts1)
    b[:surface2] = PointSurface(pts2)

    @test surface_offset(b, :surface1) == 0
    @test surface_offset(b, :surface2) == N
    @test_throws ArgumentError surface_offset(b, :nonexistent)
end

@testitem "local_to_global boundary" setup = [CommonImports] begin
    N = 10
    pts1 = rand(Point, N)
    pts2 = rand(Point, N ÷ 2)
    b = PointBoundary(pts1)
    b[:surface2] = PointSurface(pts2)

    @test local_to_global(b, :surface1, 1) == 1
    @test local_to_global(b, :surface1, N) == N
    @test local_to_global(b, :surface2, 1) == N + 1
    @test local_to_global(b, :surface2, N ÷ 2) == N + N ÷ 2
end

@testitem "global_to_local boundary" setup = [CommonImports] begin
    N = 10
    pts1 = rand(Point, N)
    pts2 = rand(Point, N ÷ 2)
    b = PointBoundary(pts1)
    b[:surface2] = PointSurface(pts2)

    @test global_to_local(b, 1) == (:surface1, 1)
    @test global_to_local(b, N) == (:surface1, N)
    @test global_to_local(b, N + 1) == (:surface2, 1)
    @test global_to_local(b, N + N ÷ 2) == (:surface2, N ÷ 2)
    @test_throws BoundsError global_to_local(b, N + N ÷ 2 + 1)
end

@testitem "local_to_global and global_to_local cloud" setup = [CommonImports] begin
    using WhatsThePoint: boundary

    N = 10
    bnd_pts = rand(Point, N)
    vol_pts = rand(Point, 5)
    cloud = PointCloud(PointBoundary(bnd_pts), PointVolume(vol_pts))

    # Boundary indices pass through to boundary-level conversion
    @test local_to_global(cloud, :surface1, 1) == 1
    @test local_to_global(cloud, :surface1, N) == N

    # Volume conversion
    @test volume_to_global(cloud, 1) == N + 1
    @test volume_to_global(cloud, 5) == N + 5

    # Global to local for boundary
    name, idx = global_to_local(cloud, 1)
    @test name == :surface1
    @test idx == 1

    # Global to local for volume
    name, idx = global_to_local(cloud, N + 3)
    @test name == :volume
    @test idx == 3

    # Round-trip: local_to_global -> global_to_local
    for i in 1:N
        name, local_idx = global_to_local(cloud, local_to_global(cloud, :surface1, i))
        @test name == :surface1
        @test local_idx == i
    end
    for i in 1:5
        name, local_idx = global_to_local(cloud, volume_to_global(cloud, i))
        @test name == :volume
        @test local_idx == i
    end
end
