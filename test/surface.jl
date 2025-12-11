@testitem "PointSurface Constructors" setup=[TestData, CommonImports] begin
    points = rand(Point, 10)
    normals = [Vec(rand(3)...) for _ in 1:10]
    areas = rand(10) * m^2
    shadow = ShadowPoints(2m, 2)

    geoms = StructArray{SurfaceElement}((points, normals, areas))
    surf = PointSurface(geoms, nothing, NoTopology())

    surf = PointSurface(points, normals, areas)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    pointset = PointSet(points)
    surf = PointSurface(pointset, normals, areas)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    surf = PointSurface(pointset, normals, areas; shadow=shadow)
    @test length(surf) == 10
    @test surf.shadow == shadow

    surf = PointSurface(points, normals)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    surf = PointSurface(points)
    @test length(surf) == 10
    @test surf.geoms isa StructVector

    surf = PointSurface(TestData.BIFURCATION_PATH)
    @test length(surf) == 24780
    @test surf.geoms isa StructVector

    surf_with_shadow = surf(shadow)
    @test surf_with_shadow.shadow == shadow
end

@testitem "SurfaceElement Constructors" setup=[TestData, CommonImports] begin
    points = rand(Point, 10)
    normals = [Vec(rand(3)...) for _ in 1:10]
    areas = rand(10) * m^2

    elem = SurfaceElement(points[1], normals[1], areas[1])
    @test elem isa SurfaceElement
    @test typeof(elem) <: Geometry
    @test elem.point == points[1]
    @test elem.normal == normals[1]
    @test elem.area == areas[1]
end

@testitem "PointSurface Properties" setup=[TestData, CommonImports] begin
    points = rand(Point, 10)
    normals = [Vec(rand(3)...) for _ in 1:10]
    areas = rand(10) * m^2

    surf = PointSurface(points, normals, areas)
    @test to(surf) == to.(surf.geoms.point)
    @test point(surf) == surf.geoms.point
    @test normal(surf) == surf.geoms.normal
    @test area(surf) == surf.geoms.area
    @test parent(surf) == surf.geoms
end

@testitem "PointSurface Base Methods" setup=[TestData, CommonImports] begin
    points = rand(Point, 10)
    normals = [Vec(rand(3)...) for _ in 1:10]
    areas = rand(10) * m^2

    surf = PointSurface(points, normals, areas)
    @test length(surf) == 10
    @test firstindex(surf) == 1
    @test lastindex(surf) == 10
    @test getindex(surf, 1) == surf.geoms[1]
    @test collect(surf) == collect(surf.geoms)
    @test view(surf, 1:5) == view(surf.geoms, 1:5)
    @test view(surf, 1:2:5) == view(surf.geoms, 1:2:5)
end

@testitem "PointSurface Meshes.jl Interface" setup=[TestData, CommonImports] begin
    points = rand(Point, 10)
    normals = [Vec(rand(3)...) for _ in 1:10]
    areas = rand(10) * m^2

    surf = PointSurface(points, normals, areas)

    pts = Meshes.pointify(surf)
    @test pts == point(surf)
    @test length(pts) == 10

    @test collect(Meshes.elements(surf)) == collect(surf.geoms)

    @test Meshes.nelements(surf) == length(surf)

    c = Meshes.centroid(surf)
    @test c isa Point

    bbox = Meshes.boundingbox(surf)
    @test bbox isa Box
end

@testitem "PointSurface Shadow Generation" setup=[TestData, CommonImports] begin
    surf_file = PointSurface(TestData.BIFURCATION_PATH)
    shadow = ShadowPoints(2m)

    shadow_points = generate_shadows(surf_file, shadow)
    @test length(shadow_points) == length(surf_file)
    @test all(p -> p isa Point, shadow_points)
end

@testitem "PointSurface Surface Operations" setup=[TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BIFURCATION_PATH)
    split_surface!(boundary, 80°)
    @test length(namedsurfaces(boundary)) == 4
end

@testitem "PointSurface Pretty Printing" setup=[TestData, CommonImports] begin
    points = rand(Point, 10)
    normals = [Vec(rand(3)...) for _ in 1:10]
    areas = rand(10) * m^2
    shadow = ShadowPoints(2m, 2)

    surf = PointSurface(points, normals, areas)
    io = IOBuffer()
    show(io, MIME("text/plain"), surf)
    output = String(take!(io))
    @test contains(output, "PointSurface")
    @test contains(output, "Number of points: 10")
    @test contains(output, "Area:")
    @test contains(output, "Shadow:")

    surf_with_shadow = PointSurface(points, normals, areas; shadow=shadow)
    io = IOBuffer()
    show(io, MIME("text/plain"), surf_with_shadow)
    output = String(take!(io))
    @test contains(output, "Shadow:")
    @test contains(output, "2")
end
