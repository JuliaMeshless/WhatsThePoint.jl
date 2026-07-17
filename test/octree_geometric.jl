# Tests for geometric utilities (Layer 2)
# These test triangle-point distance and triangle-box intersection

@testitem "Closest Point on Triangle" setup = [CommonImports] begin
    using WhatsThePoint: closest_point_on_triangle
    # Test 1: Point directly above triangle center
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(0.0, 1.0, 0.0)
    P = SVector(0.25, 0.25, 1.0)

    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q[1] ≈ 0.25 atol = 1.0e-10
    @test Q[2] ≈ 0.25 atol = 1.0e-10
    @test Q[3] ≈ 0.0 atol = 1.0e-10

    # Test 2: Point closest to vertex v1
    P = SVector(-1.0, -1.0, 0.0)
    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q ≈ v1 atol = 1.0e-10

    # Test 3: Point closest to vertex v2
    P = SVector(2.0, -1.0, 0.0)
    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q ≈ v2 atol = 1.0e-10

    # Test 4: Point closest to vertex v3
    P = SVector(-1.0, 2.0, 0.0)
    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q ≈ v3 atol = 1.0e-10

    # Test 5: Point closest to edge v1-v2
    P = SVector(0.5, -0.5, 0.0)
    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q[1] ≈ 0.5 atol = 1.0e-10
    @test Q[2] ≈ 0.0 atol = 1.0e-10
    @test Q[3] ≈ 0.0 atol = 1.0e-10

    # Test 6: Point closest to edge v1-v3
    P = SVector(-0.5, 0.5, 0.0)
    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q[1] ≈ 0.0 atol = 1.0e-10
    @test Q[2] ≈ 0.5 atol = 1.0e-10
    @test Q[3] ≈ 0.0 atol = 1.0e-10

    # Test 7: Point inside triangle projects to itself
    P = SVector(0.3, 0.3, 0.0)
    Q = closest_point_on_triangle(P, v1, v2, v3)
    @test Q ≈ P atol = 1.0e-10
end

@testitem "Triangle-Box Intersection" setup = [CommonImports] begin
    using WhatsThePoint: triangle_box_intersection
    # Box: [0,1]³
    box_min = SVector(0.0, 0.0, 0.0)
    box_max = SVector(1.0, 1.0, 1.0)

    # Test 1: Triangle completely inside box
    v1 = SVector(0.2, 0.2, 0.2)
    v2 = SVector(0.8, 0.2, 0.2)
    v3 = SVector(0.5, 0.8, 0.2)
    @test triangle_box_intersection(v1, v2, v3, box_min, box_max) == true

    # Test 2: Triangle completely outside box
    v1 = SVector(0.2, 0.2, 2.0)
    v2 = SVector(0.8, 0.2, 2.0)
    v3 = SVector(0.5, 0.8, 2.0)
    @test triangle_box_intersection(v1, v2, v3, box_min, box_max) == false
end

@testitem "Triangle-Box Intersection triangle-normal separation" setup = [CommonImports] begin
    using WhatsThePoint: triangle_box_intersection

    box_min = SVector(0.0, 0.0, 0.0)
    box_max = SVector(1.0, 1.0, 1.0)

    # Triangle whose XYZ projections overlap the box but whose plane is separated.
    # Plane equation: x + y + z = 4. Box corner sum is at most 3, so plane is outside.
    # XYZ projections: x in [0,4], y in [0,4], z in [0,4] — all overlap [0,1].
    v1 = SVector(4.0, 0.0, 0.0)
    v2 = SVector(0.0, 4.0, 0.0)
    v3 = SVector(0.0, 0.0, 4.0)
    @test triangle_box_intersection(v1, v2, v3, box_min, box_max) == false
end
