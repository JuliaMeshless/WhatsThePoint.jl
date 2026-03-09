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
