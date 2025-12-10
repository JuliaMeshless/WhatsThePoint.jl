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
    set_topology!(cloud, KNNTopology, k)

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
    set_topology!(cloud, RadiusTopology, radius)

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

@testitem "Topology invalidation" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    set_topology!(cloud, KNNTopology, 3)
    @test isvalid(topology(cloud)) == true

    # Invalidate
    invalidate_topology!(cloud)
    @test isvalid(topology(cloud)) == false

    # Accessing invalid topology should throw
    @test_throws WhatsThePoint.InvalidTopologyError neighbors(cloud)
    @test_throws WhatsThePoint.InvalidTopologyError neighbors(cloud, 1)
end

@testitem "Topology rebuild" setup = [TestData, CommonImports] begin
    using WhatsThePoint: topology
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    k = 3
    set_topology!(cloud, KNNTopology, k)

    # Invalidate then rebuild
    invalidate_topology!(cloud)
    @test isvalid(topology(cloud)) == false

    rebuild_topology!(cloud)
    @test isvalid(topology(cloud)) == true

    # Should still have correct parameters
    @test topology(cloud).k == k
    @test length(neighbors(cloud, 1)) == k
end

@testitem "NoTopology cannot be rebuilt" setup = [TestData, CommonImports] begin
    N = 10
    points = rand(Point, N)
    cloud = PointCloud(PointBoundary(points))

    @test_throws ArgumentError rebuild_topology!(cloud)
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
    set_topology!(cloud, KNNTopology, 3)
    io = IOBuffer()
    show(io, MIME("text/plain"), topology(cloud))
    output = String(take!(io))
    @test contains(output, "KNNTopology")
    @test contains(output, "k: 3")
    @test contains(output, "valid")

    # Invalid topology
    invalidate_topology!(cloud)
    io = IOBuffer()
    show(io, MIME("text/plain"), topology(cloud))
    output = String(take!(io))
    @test contains(output, "INVALID")

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
    set_topology!(cloud, KNNTopology, 3)
    io = IOBuffer()
    show(io, MIME("text/plain"), cloud)
    output = String(take!(io))
    @test contains(output, "Topology")
    @test contains(output, "valid")
end

@testitem "Backwards compatibility - PointCloud without topology" setup = [TestData, CommonImports] begin
    using WhatsThePoint: boundary
    # Ensure old code patterns still work
    N = 10
    points = rand(Point, N)
    b = PointBoundary(points)
    cloud = PointCloud(b)

    # All existing operations should work
    @test length(cloud) == N
    @test size(cloud) == (N,)
    @test boundary(cloud) isa PointBoundary
    @test volume(cloud) isa PointVolume
    @test pointify(cloud) isa Vector{<:Point}
    @test Meshes.nelements(cloud) == N
end
