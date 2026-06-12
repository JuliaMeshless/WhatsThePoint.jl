# Fast isinside queries using TriangleOctree

@testitem "isinside with TriangleOctree" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    # Interior point
    @test isinside(SVector(0.5, 0.5, 0.5), octree) == true

    # Exterior points
    @test isinside(SVector(-0.5, 0.5, 0.5), octree) == false
    @test isinside(SVector(1.5, 0.5, 0.5), octree) == false
end

@testitem "isinside works without classification" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = false)

    @test isnothing(octree.leaf_classification)

    # Should still work (falls back to signed distance)
    @test isinside(SVector(0.5, 0.5, 0.5), octree) isa Bool
end

@testitem "isinside correctness" setup = [CommonImports, OctreeTestData] begin
    using Random
    using WhatsThePoint: _get_triangle_vertices, _get_triangle_normal,
        closest_point_on_triangle

    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    # Brute-force reference implementation
    function isinside_bruteforce(point::SVector{3, T}, octree) where {T <: Real}
        min_dist = T(Inf)
        for i in 1:Meshes.nelements(octree.mesh)
            v1, v2, v3 = _get_triangle_vertices(T, octree.mesh, i)
            cp = closest_point_on_triangle(point, v1, v2, v3)
            dist = norm(point - cp)
            if dist < abs(min_dist)
                normal = _get_triangle_normal(T, octree.mesh, i)
                sign = dot(point - cp, normal) >= 0 ? 1 : -1
                min_dist = sign * dist
            end
        end
        return min_dist < 0
    end

    # Test random points
    Random.seed!(42)
    for _ in 1:50
        point = SVector(rand(3)...) .* 2.0 .- 0.5
        @test isinside(point, octree) == isinside_bruteforce(point, octree)
    end
end

@testitem "isinside with SVector vector" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)

    points = [SVector(0.5, 0.5, 0.5), SVector(-0.5, 0.5, 0.5), SVector(0.3, 0.3, 0.3)]
    results = isinside(points, octree)
    @test results == [true, false, true]
end

@testitem "isinside bbox check uses mesh bounds not octree bounds" setup = [CommonImports] begin
    using Meshes: SimpleMesh, connect, Triangle, Point

    # Regression test for bug where points far outside mesh bbox
    # were incorrectly classified as inside because the octree's
    # cubic bounding box was used instead of the actual mesh bbox.
    #
    # This 20×7×3 cuboid has a mesh bbox of ~[0,0,0]×[20,7,3],
    # but the octree (which must be cubic) extends to ~[20.4,20.4,20.4].
    # Points at z=10 and z=20 should be exterior (z >> 3).

    vertices = Point.(
        [
            (0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 7.0, 0.0), (0.0, 7.0, 0.0),
            (0.0, 0.0, 3.0), (20.0, 0.0, 3.0), (20.0, 7.0, 3.0), (0.0, 7.0, 3.0),
        ]
    )
    triangles = [
        connect((1, 3, 2), Triangle), connect((1, 4, 3), Triangle),
        connect((5, 6, 7), Triangle), connect((5, 7, 8), Triangle),
        connect((1, 2, 6), Triangle), connect((1, 6, 5), Triangle),
        connect((3, 4, 8), Triangle), connect((3, 8, 7), Triangle),
        connect((1, 5, 8), Triangle), connect((1, 8, 4), Triangle),
        connect((2, 3, 7), Triangle), connect((2, 7, 6), Triangle),
    ]
    mesh = SimpleMesh(vertices, triangles)
    octree = TriangleOctree(mesh; classify_leaves = true)

    # Interior point
    @test isinside(SVector(5.0, 3.5, 1.5), octree) == true

    # Points far outside mesh bbox (z dimension)
    @test isinside(SVector(5.0, 3.5, 10.0), octree) == false
    @test isinside(SVector(5.0, 3.5, 20.0), octree) == false

    # Points outside in other dimensions
    @test isinside(SVector(25.0, 3.5, 1.5), octree) == false  # x > 20
    @test isinside(SVector(5.0, 10.0, 1.5), octree) == false  # y > 7
end

@testitem "isinside pseudonormal sign at corner/edge/face features" setup = [
    CommonImports, OctreeTestData,
] begin
    # The signed distance is signed by the angle-weighted pseudonormal of the
    # closest feature (Bærentzen & Aanæs 2005). Queries whose closest feature
    # is a vertex or an edge are exactly the cases a plain face-normal sign
    # test (and the retired distance-weighted sign vote) can get wrong.
    mesh = OctreeTestData.unit_cube_mesh()
    octree = TriangleOctree(mesh; classify_leaves = true)
    δ = 1.0e-3

    # face features
    @test isinside(SVector(0.5, 0.5, δ), octree)
    @test !isinside(SVector(0.5, 0.5, -δ), octree)

    # edge feature: closest point of the outside-diagonal query is on the
    # cube edge x = 0, z = 0 (the inside companion resolves to a face)
    @test isinside(SVector(δ, 0.5, δ), octree)
    @test !isinside(SVector(-δ, 0.5, -δ), octree)

    # vertex features: closest point is the corner itself
    center = SVector(0.5, 0.5, 0.5)
    for corner in (
            SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0),
            SVector(1.0, 0.0, 1.0), SVector(0.0, 1.0, 0.0),
        )
        out = normalize(corner - center)
        @test isinside(corner - δ * out, octree)
        @test !isinside(corner + δ * out, octree)
    end
end

@testitem "TriangleOctree rejects inside-out meshes (signed-volume guard)" setup = [
    CommonImports, OctreeTestData,
] begin
    using WhatsThePoint: _signed_volume

    mesh = OctreeTestData.unit_cube_mesh()
    @test _signed_volume(Float64, mesh) ≈ 1.0

    # Same cube with every winding reversed: still *consistently* oriented
    # (passes has_consistent_normals) but globally inside-out — isinside
    # would classify the complement of the domain as interior, invisible to
    # any d_NN-based downstream metric. Must be rejected at construction.
    inverted = SimpleMesh(
        collect(Meshes.vertices(mesh)),
        [
            connect(reverse(Meshes.indices(c)), Meshes.Triangle)
            for c in Meshes.topology(mesh)
        ],
    )
    @test _signed_volume(Float64, inverted) ≈ -1.0
    @test_throws ArgumentError TriangleOctree(inverted; classify_leaves = true)

    # distance-only use (no isinside semantics) stays allowed
    @test TriangleOctree(inverted; classify_leaves = false) isa TriangleOctree

    # open surfaces (signed volume ≈ 0) are not rejected
    @test TriangleOctree(OctreeTestData.simple_square_mesh()) isa TriangleOctree
end
