@testitem "sample_surface Poisson-disk guarantees on cube" setup = [
    CommonImports, OctreeTestData,
] begin
    Random.seed!(2024)

    mesh = OctreeTestData.unit_cube_mesh()
    spacing = ConstantSpacing(0.15m)
    surf = sample_surface(mesh, spacing)

    pts = points(surf)
    @test length(pts) > 50

    # Separation guarantee: pairwise distance ≥ factor·h (default factor 0.75).
    raw = [Float64.(ustrip.(Meshes.to(p))) for p in pts]
    min_sep = minimum(
        norm(raw[i] - raw[j]) for i in eachindex(raw) for j in 1:(i - 1)
    )
    @test min_sep >= 0.75 * 0.15 - 1.0e-9

    # Samples lie on the cube surface: one coordinate at 0 or 1.
    on_face(c) = any(isapprox(c[d], 0.0; atol = 1.0e-12) || isapprox(c[d], 1.0; atol = 1.0e-12) for d in 1:3)
    @test all(on_face, raw)

    # Normals are unit, axis-aligned (parent-triangle normals).
    @test all(normal(surf)) do n
        isapprox(norm(n), 1.0; atol = 1.0e-9) && count(c -> abs(abs(c) - 1) < 1.0e-9, n) == 1
    end

    # Point areas preserve the total mesh surface area.
    @test sum(area(surf)) ≈ 6.0m^2 atol = 1.0e-9m^2

    # Explicit factor: strict d_NN ≥ h sampling.
    Random.seed!(2025)
    surf1 = sample_surface(mesh, spacing; factor = 1.0)
    raw1 = [Float64.(ustrip.(Meshes.to(p))) for p in points(surf1)]
    min_sep1 = minimum(
        norm(raw1[i] - raw1[j]) for i in eachindex(raw1) for j in 1:(i - 1)
    )
    @test min_sep1 >= 0.15 - 1.0e-9
end

@testitem "PointBoundary(mesh, spacing) Poisson-disk constructor" setup = [
    CommonImports, OctreeTestData,
] begin
    Random.seed!(99)

    mesh = OctreeTestData.unit_cube_mesh()
    spacing = ConstantSpacing(0.15m)
    bnd = PointBoundary(mesh, spacing)

    @test bnd isa PointBoundary
    @test length(points(bnd)) > 50

    # Integrates with discretize as a drop-in boundary.
    alg = Octree(mesh; spacing, alpha = 1.0, placement = :bridson)
    cloud = discretize(bnd, spacing; alg, max_points = 2000)
    @test length(WhatsThePoint.volume(cloud)) > 0
end

@testitem "sample_surface argument validation" setup = [CommonImports, OctreeTestData] begin
    mesh = OctreeTestData.unit_cube_mesh()
    spacing = ConstantSpacing(0.15m)

    @test_throws ArgumentError sample_surface(mesh, spacing; factor = 0.0)
    @test_throws ArgumentError sample_surface(mesh, spacing; stall_limit = 0)
end
