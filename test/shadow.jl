@testitem "ShadowPoints Constructors" setup=[TestData, CommonImports] begin
    shadow = ShadowPoints(0.1u"m")
    @test shadow isa ShadowPoints{1}
    @test shadow.Δ isa Function
    @test shadow.Δ(Point(0.0u"m", 0.0u"m")) == 0.1u"m"

    shadow0 = ShadowPoints(0.1u"m", 0)
    @test shadow0 isa ShadowPoints{0}
    @test shadow0(Point(0.0u"m", 0.0u"m")) == 0.1u"m"

    shadow1 = ShadowPoints(0.2u"m", 1)
    @test shadow1 isa ShadowPoints{1}
    @test shadow1(Point(0.0u"m", 0.0u"m")) == 0.2u"m"

    shadow2 = ShadowPoints(0.3u"m", 2)
    @test shadow2 isa ShadowPoints{2}
    @test shadow2(Point(0.0u"m", 0.0u"m")) == 0.3u"m"

    Δ_func = p -> 0.5u"m"
    shadow_func = ShadowPoints(Δ_func)
    @test shadow_func isa ShadowPoints{1}
    @test shadow_func.Δ === Δ_func

    shadow_func2 = ShadowPoints(Δ_func, 2)
    @test shadow_func2 isa ShadowPoints{2}
    @test shadow_func2.Δ === Δ_func

    test_point = Point(1.0u"m", 2.0u"m")
    @test shadow.Δ(test_point) == 0.1u"m"
    @test shadow_func.Δ(test_point) == 0.5u"m"
end

@testitem "ShadowPoints with Variable Δ Function" setup=[TestData, CommonImports] begin
    Δ_var = p -> begin
        coords = to(p)
        return sqrt(coords[1]^2 + coords[2]^2) * 0.1
    end

    shadow = ShadowPoints(Δ_var, 1)
    @test shadow.Δ isa Function

    p1 = Point(1.0u"m", 0.0u"m")
    p2 = Point(2.0u"m", 0.0u"m")
    @test shadow.Δ(p1) ≈ 0.1u"m"
    @test shadow.Δ(p2) ≈ 0.2u"m"
end

@testitem "generate_shadows with Constant Δ" setup=[TestData, CommonImports] begin
    points = [
        Point(1.0u"m", 0.0u"m"),
        Point(0.0u"m", 1.0u"m"),
        Point(-1.0u"m", 0.0u"m"),
        Point(0.0u"m", -1.0u"m"),
    ]

    normals = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(0.0, -1.0)]

    Δ = 0.5u"m"
    shadow = ShadowPoints(Δ)
    shadow_points = generate_shadows(points, normals, shadow)

    @test shadow_points isa Vector{<:Point}
    @test length(shadow_points) == length(points)

    expected = [
        Point(0.5u"m", 0.0u"m"),
        Point(0.0u"m", 0.5u"m"),
        Point(-0.5u"m", 0.0u"m"),
        Point(0.0u"m", -0.5u"m"),
    ]

    for (i, (sp, exp)) in enumerate(zip(shadow_points, expected))
        sp_coords = to(sp)
        exp_coords = to(exp)
        @test sp_coords[1] ≈ exp_coords[1] atol = 1e-10u"m"
        @test sp_coords[2] ≈ exp_coords[2] atol = 1e-10u"m"
    end
end

@testitem "generate_shadows with Function Δ" setup=[TestData, CommonImports] begin
    points = [Point(2.0u"m", 0.0u"m"), Point(4.0u"m", 0.0u"m")]

    normals = [SVector(1.0, 0.0), SVector(1.0, 0.0)]

    Δ_func = p -> begin
        coords = to(p)
        return abs(coords[1]) * 0.1
    end

    shadow = ShadowPoints(Δ_func, 2)
    shadow_points = generate_shadows(points, normals, shadow)

    @test length(shadow_points) == 2

    sp1_coords = to(shadow_points[1])
    @test sp1_coords[1] ≈ 1.8u"m" atol = 1e-10u"m"

    sp2_coords = to(shadow_points[2])
    @test sp2_coords[1] ≈ 3.6u"m" atol = 1e-10u"m"
end

@testitem "generate_shadows Different Orders" setup=[TestData, CommonImports] begin
    points = [Point(1.0u"m", 1.0u"m")]
    normals = [SVector(1.0, 0.0) / sqrt(2.0)]
    Δ = 0.1u"m"

    for order in [0, 1, 2]
        shadow = ShadowPoints(Δ, order)
        shadow_points = generate_shadows(points, normals, shadow)

        @test length(shadow_points) == 1
        @test shadow_points[1] isa Point
    end
end

@testitem "generate_shadows 3D" setup=[TestData, CommonImports] begin
    points = [Point(1.0u"m", 0.0u"m", 0.0u"m"), Point(0.0u"m", 1.0u"m", 0.0u"m")]

    normals = [SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]

    Δ = 0.25u"m"
    shadow = ShadowPoints(Δ)
    shadow_points = generate_shadows(points, normals, shadow)

    @test length(shadow_points) == 2

    sp1_coords = to(shadow_points[1])
    @test sp1_coords[1] ≈ 0.75u"m" atol = 1e-10u"m"
    @test sp1_coords[2] ≈ 0.0u"m" atol = 1e-10u"m"
    @test sp1_coords[3] ≈ 0.0u"m" atol = 1e-10u"m"

    sp2_coords = to(shadow_points[2])
    @test sp2_coords[1] ≈ 0.0u"m" atol = 1e-10u"m"
    @test sp2_coords[2] ≈ 0.75u"m" atol = 1e-10u"m"
    @test sp2_coords[3] ≈ 0.0u"m" atol = 1e-10u"m"
end

@testitem "ShadowPoints Display" setup=[TestData, CommonImports] begin
    shadow1 = ShadowPoints(0.1u"m", 1)
    @test_nowarn show(IOBuffer(), MIME"text/plain"(), shadow1)
    @test_nowarn show(IOBuffer(), shadow1)

    shadow_func = ShadowPoints(p -> 0.5u"m", 2)
    @test_nowarn show(IOBuffer(), MIME"text/plain"(), shadow_func)
end
