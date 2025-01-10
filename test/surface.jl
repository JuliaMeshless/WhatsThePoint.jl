using WhatsThePoint
using Meshes
using StructArrays
using Unitful: m

points = rand(Point, 10)
normals = [rand(3) .* m for _ in 1:10]
areas = rand(10) * m^2
shadow = ShadowPoints(2m)

@testset "PointSurface Constructors" begin
    # Test default constructor
    geoms = StructArray{SurfaceElement}((points, normals, areas))
    surf = PointSurface(geoms, nothing)

    # Test constructor with points, normals, and areas
    surf = PointSurface(points, normals, areas)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    # Test constructor with points and normals only
    surf = PointSurface(points, normals)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    # Test constructor with points only
    surf = PointSurface(points)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    # Test constructor with file path
    surf = PointSurface(joinpath(@__DIR__, "data", "bifurcation.stl"))
    @test length(surf) == 24780
    @test surf.geoms isa StructVector

    # Test shadow assignment
    surf_with_shadow = surf(shadow)
    @test surf_with_shadow.shadow == shadow
end

@testset "SurfaceElement Constructors" begin
    elem = SurfaceElement(points[1], normals[1], areas[1])
    @test elem isa SurfaceElement
    @test typeof(elem) <: Geometry
    @test elem.point == points[1]
    @test elem.normal == normals[1]
    @test elem.area == areas[1]
end

@testset "Properties" begin
    surf = PointSurface(points, normals, areas)
    @test to(surf) == to.(surf.geoms.point)
    @test point(surf) == surf.geoms.point
    @test normal(surf) == surf.geoms.normal
    @test area(surf) == surf.geoms.area
    @test parent(surf) == surf.geoms
end

@testset "Base Methods" begin
    surf = PointSurface(points, normals, areas)
    @test length(surf) == 10
    @test firstindex(surf) == 1
    @test lastindex(surf) == 10
    @test getindex(surf, 1) == surf.geoms[1]
    @test iterate(surf, 1) == iterate(surf.geoms, 1)
    @test view(surf, 1:5) == view(surf.geoms, 1:5)
    @test view(surf, 1:2:5) == view(surf.geoms, 1:2:5)
end

@testset "Overloads" begin
    surf = PointSurface(points, normals, areas)
    @test collect(Meshes.elements(surf)) == collect(surf.geoms)
    @test Meshes.nelements(surf) == length(surf)
end

@testset "Surface Operations" begin
    boundary = PointBoundary(joinpath(@__DIR__, "data", "bifurcation.stl"))
    split_surface!(boundary, 80)
    @test length(surfaces(boundary)) == 4
end
