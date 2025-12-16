@testitem "NoTopology default" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    @test topology(cloud) isa NoTopology
    @test hastopology(cloud) == false
    @test isvalid(topology(cloud)) == true
end

@testitem "KNNTopology construction" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 20
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    k = 5
    cloud = set_topology(cloud, KNNTopology, k)

    @test hastopology(cloud) == true
    @test topology(cloud) isa KNNTopology
    @test topology(cloud).k == k
    @test isvalid(topology(cloud)) == true

    # Check neighbor structure
    nbrs = neighbors(cloud)
    @test nbrs isa Vector{Vector{Int}}
    @test length(nbrs) == N

    # Each point should have exactly k neighbors
    for n in nbrs
        @test length(n) == k
    end

    # Check single point neighbor access
    nbrs_1 = neighbors(cloud, 1)
    @test nbrs_1 isa Vector{Int}
    @test length(nbrs_1) == k

    # Neighbors should not include self
    for i in 1:N
        @test i ∉ neighbors(cloud, i)
    end
end

@testitem "RadiusTopology construction" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    # Create points in a grid pattern for predictable neighbors
    spacing = 0.1u"m"
    points = [Point(i * spacing, j * spacing) for i in 0:4 for j in 0:4]
    cloud = PointCloud(PointBoundary(points))

    # Radius that should capture adjacent points
    radius = 0.15u"m"
    cloud = set_topology(cloud, RadiusTopology, radius)

    @test hastopology(cloud) == true
    @test topology(cloud) isa RadiusTopology
    @test topology(cloud).radius == radius
    @test isvalid(topology(cloud)) == true

    # Check neighbor structure
    nbrs = neighbors(cloud)
    @test nbrs isa Vector{Vector{Int}}
    @test length(nbrs) == length(points)

    # Neighbors should not include self
    for i in 1:length(points)
        @test i ∉ neighbors(cloud, i)
    end
end

@testitem "Topology rebuild" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    k = 3
    cloud = set_topology(cloud, KNNTopology, k)

    # Rebuild modifies topology in place
    rebuild_topology!(cloud)
    @test isvalid(topology(cloud)) == true

    # Should still have correct parameters
    @test topology(cloud).k == k
    @test length(neighbors(cloud, 1)) == k
end

@testitem "NoTopology rebuild is no-op" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    # Should be a no-op, no error
    rebuild_topology!(cloud)
    @test topology(cloud) isa NoTopology
end

@testitem "NoTopology has no neighbors" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    @test_throws ArgumentError neighbors(cloud)
    @test_throws ArgumentError neighbors(cloud, 1)
end

@testitem "Topology pretty printing" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    # KNNTopology
    cloud = set_topology(cloud, KNNTopology, 3)
    io = IOBuffer()
    show(io, MIME("text/plain"), topology(cloud))
    output = String(take!(io))
    @test contains(output, "KNNTopology")
    @test contains(output, "k: 3")

    # NoTopology
    io = IOBuffer()
    show(io, NoTopology())
    output = String(take!(io))
    @test contains(output, "NoTopology")
end

@testitem "PointCloud with topology pretty printing" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    # Without topology
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud)
    output = String(take!(io))
    @test !contains(output, "Topology")

    # With topology
    cloud = set_topology(cloud, KNNTopology, 3)
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud)
    output = String(take!(io))
    @test contains(output, "Topology")
end

@testitem "Backwards compatibility - PointCloud without topology" setup = [
    TestData, CommonImports
] begin
    using WhatsThePoint: boundary
    # Ensure old code patterns still work
    N = 10
    pts = rand(Point, N)
    b = PointBoundary(pts)
    cloud = PointCloud(b)

    # All existing operations should work
    @test length(cloud) == N
    @test size(cloud) == (N,)
    @test boundary(cloud) isa PointBoundary
    @test volume(cloud) isa PointVolume
    @test points(cloud) isa Vector{<:Point}
    @test Meshes.nelements(cloud) == N
end

@testitem "Surface-level topology" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 20
    points = rand(Point, N)
    normals = [Point(0.0, 0.0, 1.0) for _ in 1:N]
    areas = zeros(N) * u"m^2"
    surf = PointSurface(points, normals, areas)

    # Default is NoTopology
    @test topology(surf) isa NoTopology
    @test hastopology(surf) == false

    # Add topology
    k = 5
    surf = set_topology(surf, KNNTopology, k)

    @test hastopology(surf) == true
    @test topology(surf) isa KNNTopology
    @test topology(surf).k == k

    # Check neighbors
    nbrs = neighbors(surf)
    @test length(nbrs) == N
    @test length(neighbors(surf, 1)) == k
end

@testitem "Volume-level topology" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 20
    points = rand(Point, N)
    vol = PointVolume(points)

    # Default is NoTopology
    @test topology(vol) isa NoTopology
    @test hastopology(vol) == false

    # Add topology
    k = 5
    vol = set_topology(vol, KNNTopology, k)

    @test hastopology(vol) == true
    @test topology(vol) isa KNNTopology
    @test topology(vol).k == k

    # Check neighbors
    nbrs = neighbors(vol)
    @test length(nbrs) == N
    @test length(neighbors(vol, 1)) == k
end

@testitem "Surface-level RadiusTopology" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    # Create grid pattern for predictable neighbors
    spacing = 0.1u"m"
    points = [Point(i * spacing, j * spacing) for i in 0:4 for j in 0:4]
    normals = [Point(0.0, 0.0, 1.0) for _ in points]
    areas = zeros(length(points)) * u"m^2"
    surf = PointSurface(points, normals, areas)

    # Add RadiusTopology
    radius = 0.15u"m"
    surf = set_topology(surf, RadiusTopology, radius)

    # Validate topology
    @test hastopology(surf) == true
    @test topology(surf) isa RadiusTopology
    @test topology(surf).radius == radius
    @test isvalid(topology(surf)) == true

    # Check neighbors structure
    nbrs = neighbors(surf)
    @test length(nbrs) == length(points)
    # Verify self-exclusion
    for i in 1:length(points)
        @test i ∉ neighbors(surf, i)
    end
end

@testitem "Volume-level RadiusTopology" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    # Create grid pattern for predictable neighbors
    spacing = 0.1u"m"
    points = [Point(i * spacing, j * spacing) for i in 0:4 for j in 0:4]
    vol = PointVolume(points)

    # Add RadiusTopology
    radius = 0.15u"m"
    vol = set_topology(vol, RadiusTopology, radius)

    # Validate topology
    @test hastopology(vol) == true
    @test topology(vol) isa RadiusTopology
    @test topology(vol).radius == radius
    @test isvalid(topology(vol)) == true

    # Check neighbors structure
    nbrs = neighbors(vol)
    @test length(nbrs) == length(points)
    # Verify self-exclusion
    for i in 1:length(points)
        @test i ∉ neighbors(vol, i)
    end
end

@testitem "RadiusTopology rebuild" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    # Create grid for predictable results
    spacing = 0.1u"m"
    points = [Point(i * spacing, j * spacing) for i in 0:3 for j in 0:3]
    cloud = PointCloud(PointBoundary(points))

    # Set RadiusTopology
    radius = 0.15u"m"
    cloud = set_topology(cloud, RadiusTopology, radius)

    # Rebuild topology in place
    rebuild_topology!(cloud)

    # Validate topology still valid
    @test isvalid(topology(cloud)) == true
    @test topology(cloud) isa RadiusTopology
    @test topology(cloud).radius == radius

    # Neighbors should still work
    nbrs = neighbors(cloud, 1)
    @test nbrs isa Vector{Int}
    @test 1 ∉ nbrs  # Self-exclusion
end

@testitem "Surface RadiusTopology rebuild" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    # Create surface with RadiusTopology
    spacing = 0.1u"m"
    points = [Point(i * spacing, j * spacing) for i in 0:3 for j in 0:3]
    normals = [Point(0.0, 0.0, 1.0) for _ in points]
    areas = zeros(length(points)) * u"m^2"
    surf = PointSurface(points, normals, areas)

    radius = 0.15u"m"
    surf = set_topology(surf, RadiusTopology, radius)

    # Rebuild topology
    rebuild_topology!(surf)

    # Validate topology preserved
    @test isvalid(topology(surf)) == true
    @test topology(surf) isa RadiusTopology
    @test topology(surf).radius == radius
    @test hastopology(surf) == true

    # Check neighbors still valid
    @test length(neighbors(surf)) == length(points)
end

@testitem "Volume RadiusTopology rebuild" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    # Create volume with RadiusTopology
    spacing = 0.1u"m"
    points = [Point(i * spacing, j * spacing) for i in 0:3 for j in 0:3]
    vol = PointVolume(points)

    radius = 0.15u"m"
    vol = set_topology(vol, RadiusTopology, radius)

    # Rebuild topology
    rebuild_topology!(vol)

    # Validate topology preserved
    @test isvalid(topology(vol)) == true
    @test topology(vol) isa RadiusTopology
    @test topology(vol).radius == radius
    @test hastopology(vol) == true

    # Check neighbors still valid
    @test length(neighbors(vol)) == length(points)
end
