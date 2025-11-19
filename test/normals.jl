using WhatsThePoint
using Meshes
using StaticArrays
using LinearAlgebra

@testitem "2D" begin
    circle2D = Point.([(cos(θ), sin(θ)) for θ in 0:(π / 4):(7π / 4)])
    n = SVector(0.0, 1.0)
    e = SVector(1.0, 0.0)
    s = SVector(0.0, -1.0)
    w = SVector(-1.0, 0.0)
    ne = ones(SVector{2}) * sqrt(2) / 2
    sw = -ones(SVector{2}) * sqrt(2) / 2
    nw = SVector(-sqrt(2) / 2, sqrt(2) / 2)
    se = SVector(sqrt(2) / 2, -sqrt(2) / 2)
    correct_normals = (e, ne, n, nw, w, sw, s, se)
    test_normals = [(n, -n) for n in correct_normals]

    @test any((WhatsThePoint._compute_normal(circle2D[2:4]),) .≈ (n, -n))
    @test any((WhatsThePoint._compute_normal(circle2D[1:3]),) .≈ (ne, -ne))

    k = 3
    search_method = KNearestSearch(circle2D, k)
    computed_normals = compute_normals(search_method, circle2D)

    @test all([any((computed_normals[i],) .≈ n) for (i, n) in enumerate(test_normals)])
    orient_normals!(computed_normals, circle2D; k=k)
    @test all(computed_normals .≈ correct_normals)
end

@testitem "3D" begin
    tri3d = Point.([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    n = ones(SVector{3}) * sqrt(3) / 3
    computed_normal = WhatsThePoint._compute_normal(tri3d)
    @test any(Ref(computed_normal) .≈ (n, -n))

    # create sphere using fibonacci lattice
    N = 100
    ids = 0.0:(N .+ 0.5)
    ϕ = @. acos(1 - 2 * ids / N)
    θ = @. π * (1 + sqrt(5)) * ids
    x = @. cos(θ) * sin(ϕ)
    y = @. sin(θ) * sin(ϕ)
    z = @. cos(ϕ)
    sphere3d = Point.(x, y, z)
    correct_normals = SVector.(x, y, z)
    test_normals = [[n, -n] for n in correct_normals]

    #tree = KDTree(coordinates.(sphere3d))
    k = 5
    search_method = KNearestSearch(sphere3d, k)
    computed_normals = compute_normals(search_method, sphere3d)

    # test if normals are within 5 deg of correct given it may be unoriented
    @test all([
        any(WhatsThePoint._angle.((computed_normals[i],), n) .< 10 * π / 180) for
        (i, n) in enumerate(test_normals)
    ])
    orient_normals!(computed_normals, sphere3d; k=5)
    # test if normals are within 5 deg of correct after correcting orientation
    @test all(WhatsThePoint._angle.(computed_normals, correct_normals) .< 10 * π / 180)
end

@testitem "update_normals!" begin
    # Test that update_normals! correctly recomputes normals (issue #47)
    # Use the same 2D circle from the 2D test
    circle2D = Point.([(cos(θ), sin(θ)) for θ in 0:(π / 4):(7π / 4)])

    # Create surface with computed normals
    k = 3
    surf = PointSurface(circle2D; k=k)

    # Save the correctly computed normals
    original_normals = copy(normal(surf))

    # Randomize the normals
    normals_ref = normal(surf)
    for i in eachindex(normals_ref)
        normals_ref[i] = normalize(randn(SVector{2,Float64}))
    end

    # Verify normals were actually randomized
    @test !(normal(surf) ≈ original_normals)

    # Call update_normals! to recompute them
    update_normals!(surf; k=k)
    orient_normals!(surf)

    # Test that the recomputed normals match the original
    @test normal(surf) ≈ original_normals
end
