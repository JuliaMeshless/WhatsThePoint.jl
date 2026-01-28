# Tests for Triangle and TriangleMesh data structures

@testitem "Triangle Basic Construction with Normal" setup = [CommonImports] begin
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(0.0, 1.0, 0.0)
    n = SVector(0.0, 0.0, 1.0)

    tri = Triangle(v1, v2, v3, n)

    @test tri.v1 == v1
    @test tri.v2 == v2
    @test tri.v3 == v3
    @test tri.normal == n
    @test norm(tri.normal) ≈ 1.0
end

@testitem "Triangle Auto-compute Normal (Right-hand Rule)" setup = [CommonImports] begin
    # Triangle in XY plane, normal points +Z
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(0.0, 1.0, 0.0)

    tri = Triangle(v1, v2, v3)

    # Normal should be (v2-v1) × (v3-v1) = [1,0,0] × [0,1,0] = [0,0,1]
    @test tri.normal[1] ≈ 0.0 atol = 1.0e-10
    @test tri.normal[2] ≈ 0.0 atol = 1.0e-10
    @test tri.normal[3] ≈ 1.0 atol = 1.0e-10
    @test norm(tri.normal) ≈ 1.0
end

@testitem "Triangle Normal Direction (Counter-clockwise)" setup = [CommonImports] begin
    # View from +Z: vertices go counter-clockwise → normal points +Z
    v1 = SVector(0.0, 0.0, 5.0)
    v2 = SVector(1.0, 0.0, 5.0)
    v3 = SVector(0.5, 1.0, 5.0)

    tri = Triangle(v1, v2, v3)

    # Normal should point +Z
    @test tri.normal[3] > 0.0
    @test norm(tri.normal) ≈ 1.0
end

@testitem "Triangle Degenerate Triangle Detection" setup = [CommonImports] begin
    # Collinear points
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(2.0, 0.0, 0.0)  # On same line

    @test_throws ErrorException Triangle(v1, v2, v3)
end

@testitem "Triangle Duplicate Vertices Detection" setup = [CommonImports] begin
    v1 = SVector(1.0, 2.0, 3.0)
    v2 = SVector(1.0, 2.0, 3.0)  # Duplicate
    v3 = SVector(4.0, 5.0, 6.0)

    @test_throws ErrorException Triangle(v1, v2, v3)
end

@testitem "Triangle Invalid Normal Rejection" setup = [CommonImports] begin
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(0.0, 1.0, 0.0)
    bad_normal = SVector(0.0, 0.0, 2.0)  # Not unit length

    @test_throws ErrorException Triangle(v1, v2, v3, bad_normal)
end

@testitem "Triangle with Non-unit Edges" setup = [CommonImports] begin
    # Larger triangle - normal should still be unit
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(2.0, 0.0, 0.0)
    v3 = SVector(0.0, 2.0, 0.0)

    tri = Triangle(v1, v2, v3)

    # Even with larger triangle, normal should be unit
    @test norm(tri.normal) ≈ 1.0
end

@testitem "TriangleMesh Manual Construction" setup = [CommonImports] begin
    # Create 2 triangles forming a square
    tri1 = Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0))
    tri2 = Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), SVector(0.0, 1.0, 0.0))

    bbox_min = SVector(0.0, 0.0, -0.1)  # Small epsilon in z for valid box
    bbox_max = SVector(1.0, 1.0, 0.1)

    tri_list = Triangle{Float64}[]  # Empty list
    @test_throws ErrorException TriangleMesh(tri_list, bbox_min, bbox_max)
end

@testitem "TriangleMesh Invalid Bounding Box Rejection" setup = [CommonImports] begin
    tri = Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))

    # Max < Min
    bbox_min = SVector(1.0, 1.0, 1.0)
    bbox_max = SVector(0.0, 0.0, 0.0)

    @test_throws ErrorException TriangleMesh([tri], bbox_min, bbox_max)
end

@testitem "TriangleMesh Iteration" setup = [CommonImports] begin
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.5, 1.0, 0.0)),
        Triangle(SVector(1.0, 1.0, 0.0), SVector(2.0, 1.0, 0.0), SVector(1.5, 2.0, 0.0)),
    ]
    mesh = TriangleMesh(tri_list, SVector(0.0, 0.0, 0.0), SVector(2.0, 2.0, 1.0))

    count = 0
    for tri in mesh
        count += 1
        @test isa(tri, Triangle)
    end
    @test count == 2
end

@testitem "TriangleMesh Bounding Box Helpers" setup = [CommonImports] begin
    mesh = TriangleMesh(
        [Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))],
        SVector(-1.0, -2.0, -3.0),
        SVector(4.0, 5.0, 6.0),
    )

    size = bbox_size(mesh)
    @test size == SVector(5.0, 7.0, 9.0)

    center = bbox_center(mesh)
    @test center == SVector(1.5, 1.5, 1.5)
end

@testitem "Automatic Bounding Box - Simple Cube" setup = [CommonImports] begin
    # 8 vertices of unit cube
    tri_list = [
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0)),
        Triangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), SVector(0.0, 1.0, 0.0)),
        Triangle(SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 1.0), SVector(1.0, 1.0, 1.0)),
    ]

    mesh = TriangleMesh(tri_list)

    @test mesh.bbox_min == SVector(0.0, 0.0, 0.0)
    @test mesh.bbox_max == SVector(1.0, 1.0, 1.0)
end

@testitem "Automatic Bounding Box - Offset Triangles" setup = [CommonImports] begin
    # Triangles not at origin (planar in z)
    tri_list = [
        Triangle(SVector(5.0, 5.0, 5.0), SVector(6.0, 5.0, 5.0), SVector(6.0, 6.0, 5.0)),
        Triangle(SVector(5.0, 5.0, 5.0), SVector(6.0, 6.0, 5.0), SVector(5.0, 6.0, 5.0)),
    ]

    mesh = TriangleMesh(tri_list)

    # Check x,y are exact, z has small epsilon for planar mesh
    @test mesh.bbox_min[1] == 5.0
    @test mesh.bbox_min[2] == 5.0
    @test mesh.bbox_min[3] < 5.0  # Should be slightly less due to epsilon
    @test mesh.bbox_max[1] == 6.0
    @test mesh.bbox_max[2] == 6.0
    @test mesh.bbox_max[3] > 5.0  # Should be slightly more due to epsilon
end

@testitem "Automatic Bounding Box - Negative Coordinates" setup = [CommonImports] begin
    tri_list = [
        Triangle(
            SVector(-5.0, -3.0, -2.0),
            SVector(-1.0, -3.0, -2.0),
            SVector(-1.0, 0.0, -2.0),
        ),
    ]

    mesh = TriangleMesh(tri_list)

    # Check x,y are exact, z has small epsilon for planar mesh
    @test mesh.bbox_min[1] == -5.0
    @test mesh.bbox_min[2] == -3.0
    @test mesh.bbox_min[3] < -2.0  # Should be slightly less due to epsilon
    @test mesh.bbox_max[1] == -1.0
    @test mesh.bbox_max[2] ≈ 0.0
    @test mesh.bbox_max[3] > -2.0  # Should be slightly more due to epsilon
end

@testitem "STL Loading - box.stl" setup = [CommonImports, TestData] begin
    # Check if test data exists
    if isfile(TestData.BOX_PATH)
        mesh = TriangleMesh(TestData.BOX_PATH)

        # box.stl has 46,786 triangles and 23,395 unique points
        @test length(mesh) == 46786
        @test length(unique_points(mesh)) == 23395

        # Check all normals are unit length
        for tri in mesh
            @test norm(tri.normal) ≈ 1.0
        end

        # Check bounding box is reasonable
        size = bbox_size(mesh)
        @test all(size .> 0.0)
    else
        @test_skip "box.stl not available"
    end
end

@testitem "STL Loading - Invalid File" setup = [CommonImports] begin
    @test_throws Exception TriangleMesh("nonexistent.stl")
end
