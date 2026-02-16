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

@testitem "Distance Point to Triangle" setup = [CommonImports] begin
    using WhatsThePoint: distance_point_triangle
    # Simple triangle in xy-plane
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(0.0, 1.0, 0.0)

    # Test 1: Point on triangle
    P = SVector(0.25, 0.25, 0.0)
    d = distance_point_triangle(P, v1, v2, v3)
    @test d ≈ 0.0 atol = 1.0e-10

    # Test 2: Point above triangle
    P = SVector(0.25, 0.25, 1.0)
    d = distance_point_triangle(P, v1, v2, v3)
    @test d ≈ 1.0 atol = 1.0e-10
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


@testitem "Signed Distance Point to Triangle" setup = [CommonImports] begin
    using WhatsThePoint: distance_point_triangle
    # Triangle in xy-plane
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(0.0, 1.0, 0.0)
    normal = SVector(0.0, 0.0, 1.0)

    # Test 1: Point above triangle (on normal side)
    P_above = SVector(0.25, 0.25, 1.0)
    d = distance_point_triangle(P_above, v1, v2, v3, normal)
    @test d ≈ 1.0 atol = 1.0e-10  # Positive on normal side

    # Test 2: Point below triangle (opposite side)
    P_below = SVector(0.25, 0.25, -1.0)
    d = distance_point_triangle(P_below, v1, v2, v3, normal)
    @test d ≈ -1.0 atol = 1.0e-10  # Negative on opposite side

    # Test 3: Point on triangle plane
    P_on = SVector(0.25, 0.25, 0.0)
    d = distance_point_triangle(P_on, v1, v2, v3, normal)
    @test abs(d) < 1.0e-10  # Should be zero

    # Test 4: Point above but closer to edge
    P_edge = SVector(0.5, -0.5, 1.0)
    d = distance_point_triangle(P_edge, v1, v2, v3, normal)
    @test d > 0  # Positive on normal side
    # Distance should be sqrt(0.5^2 + 1^2) ≈ 1.118
    @test d ≈ sqrt(0.5^2 + 1.0^2) atol = 1.0e-10
end
