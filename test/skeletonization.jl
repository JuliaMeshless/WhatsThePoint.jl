@testitem "LBCSkeletonization defaults" setup = [CommonImports] begin
    alg = LBCSkeletonization()
    @test alg.k == 15
    @test alg.WL_init == 1.0
    @test alg.WL_factor == 2.0
    @test alg.max_iters == 50
    @test alg.tol == 1e-4
end

@testitem "GraphExtractionParams defaults" setup = [CommonImports] begin
    params = GraphExtractionParams()
    @test isnothing(params.voxel_size)
    @test params.min_branch_length == 3
    @test params.connectivity_k == 6
end

@testitem "build_adaptive_laplacian 2D" setup = [CommonImports] begin
    using SparseArrays
    using WhatsThePoint: build_adaptive_laplacian

    # Simple 2D line of points
    pts = [Point(i * 0.1m, 0.0m) for i in 0:10]
    surf = PointSurface(pts)

    L, vols, scales = build_adaptive_laplacian(point(surf), 3)

    @test size(L) == (11, 11)
    @test issymmetric(L)
    @test all(diag(L) .>= 0)  # Laplacian diagonal is non-negative
    @test length(vols) == 11
    @test length(scales) == 11
    @test all(scales .> 0)
end

@testitem "build_adaptive_laplacian 3D" setup = [CommonImports] begin
    using SparseArrays
    using WhatsThePoint: build_adaptive_laplacian

    # Simple 3D line of points
    pts = [Point(i * 0.1m, 0.0m, 0.0m) for i in 0:10]
    surf = PointSurface(pts)

    L, vols, scales = build_adaptive_laplacian(point(surf), 3)

    @test size(L) == (11, 11)
    @test issymmetric(L)
    @test length(vols) == 11
    @test length(scales) == 11
    # For 3D, volumes should be scale^3
    @test all(vols .>= 0)
end

@testitem "contract_lbc basic" setup = [CommonImports] begin
    using WhatsThePoint: contract_lbc

    # Create a simple 3D tube-like structure (circle of points at each z-level)
    pts = [
        Point(0.2 * cos(theta) * m, 0.2 * sin(theta) * m, z * m)
        for z in 0:0.1:1.0 for theta in range(0, 2pi; length=9)[1:8]
    ]

    surf = PointSurface(pts)

    alg = LBCSkeletonization(k=5, max_iters=5, tol=1e-2)
    contracted = contract_lbc(surf, alg)

    @test contracted isa ContractedSurface
    @test length(contracted.points) == length(pts)
    @test !isempty(contracted.convergence)
    @test length(contracted.convergence) <= 5
end

@testitem "ContractedSurface accessors" setup = [CommonImports] begin
    using WhatsThePoint: contract_lbc

    pts = [Point(i * 0.1m, 0.0m, 0.0m) for i in 0:5]
    surf = PointSurface(pts)

    alg = LBCSkeletonization(k=3, max_iters=2)
    contracted = contract_lbc(surf, alg)

    @test length(contracted) == 6
    @test points(contracted) == contracted.points
end

@testitem "SkeletonGraph creation and accessors" setup = [CommonImports] begin
    using SimpleWeightedGraphs
    using Graphs: add_edge!, degree

    # Create a simple skeleton graph manually
    g = SimpleWeightedGraph(3)
    add_edge!(g, 1, 2, 1.0)
    add_edge!(g, 2, 3, 1.5)

    node1 = SkeletonNode(Point(0.0m, 0.0m, 0.0m), [1, 2])
    node2 = SkeletonNode(Point(1.0m, 0.0m, 0.0m), [3, 4])
    node3 = SkeletonNode(Point(2.5m, 0.0m, 0.0m), [5])

    sg = SkeletonGraph(g, [node1, node2, node3])

    @test nv(sg) == 3
    @test ne(sg) == 2
    @test length(sg) == 3
    @test length(nodes(sg)) == 3
    @test graph(sg) === g

    # Test analysis functions
    @test skeleton_length(sg) == 2.5
    @test isempty(branch_points(sg))  # No nodes with degree > 2
    @test sort(end_points(sg)) == [1, 3]  # Nodes 1 and 3 have degree 1
end

@testitem "SkeletonGraph branch detection" setup = [CommonImports] begin
    using SimpleWeightedGraphs
    using Graphs: add_edge!

    # Create a Y-shaped skeleton (one branch point)
    g = SimpleWeightedGraph(4)
    add_edge!(g, 1, 2, 1.0)  # stem
    add_edge!(g, 2, 3, 1.0)  # left branch
    add_edge!(g, 2, 4, 1.0)  # right branch

    nodes_list = [
        SkeletonNode(Point(0.0m, 0.0m, 0.0m), [1]),
        SkeletonNode(Point(0.0m, 1.0m, 0.0m), [2]),  # branch point
        SkeletonNode(Point(-1.0m, 2.0m, 0.0m), [3]),
        SkeletonNode(Point(1.0m, 2.0m, 0.0m), [4]),
    ]

    sg = SkeletonGraph(g, nodes_list)

    @test branch_points(sg) == [2]  # Node 2 has degree 3
    @test sort(end_points(sg)) == [1, 3, 4]  # Nodes 1, 3, 4 have degree 1
end

@testitem "skeletonize without graph extraction" setup = [CommonImports] begin
    # Create a simple tube
    pts = [
        Point(0.1 * cos(theta) * m, 0.1 * sin(theta) * m, z * m)
        for z in 0:0.1:0.5 for theta in range(0, 2pi; length=9)[1:8]
    ]

    surf = PointSurface(pts)

    result, conv = skeletonize(surf;
        alg=LBCSkeletonization(k=5, max_iters=3),
        extract_graph=false
    )

    @test result isa ContractedSurface
    @test !isempty(conv)
end

@testitem "skeletonize with graph extraction" setup = [CommonImports] begin
    # Create a simple tube
    pts = [
        Point(0.1 * cos(theta) * m, 0.1 * sin(theta) * m, z * m)
        for z in 0:0.1:0.5 for theta in range(0, 2pi; length=9)[1:8]
    ]

    surf = PointSurface(pts)

    result, conv = skeletonize(surf;
        alg=LBCSkeletonization(k=5, max_iters=3),
        graph_params=GraphExtractionParams(min_branch_length=1)
    )

    @test result isa SkeletonGraph
    @test nv(result) > 0
    @test !isempty(conv)
end

@testitem "SkeletonGraph pretty printing" setup = [CommonImports] begin
    using SimpleWeightedGraphs
    using Graphs: add_edge!

    g = SimpleWeightedGraph(2)
    add_edge!(g, 1, 2, 1.0)

    nodes_list = [
        SkeletonNode(Point(0.0m, 0.0m, 0.0m), [1]),
        SkeletonNode(Point(1.0m, 0.0m, 0.0m), [2]),
    ]

    sg = SkeletonGraph(g, nodes_list)

    # Test that show doesn't error
    io = IOBuffer()
    show(io, MIME"text/plain"(), sg)
    output = String(take!(io))
    @test contains(output, "SkeletonGraph")
    @test contains(output, "2 nodes")
end

@testitem "ContractedSurface pretty printing" setup = [CommonImports] begin
    using WhatsThePoint: contract_lbc

    pts = [Point(i * 0.1m, 0.0m, 0.0m) for i in 0:3]
    surf = PointSurface(pts)

    alg = LBCSkeletonization(k=2, max_iters=2)
    contracted = contract_lbc(surf, alg)

    io = IOBuffer()
    show(io, MIME"text/plain"(), contracted)
    output = String(take!(io))
    @test contains(output, "ContractedSurface")
    @test contains(output, "4 contracted points")
end
