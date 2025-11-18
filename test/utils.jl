using WhatsThePoint
using Meshes
using StaticArrays
using Unitful: m, rad, °
using Test

@testset "findmin_turbo" begin
    @test findmin_turbo([3.0, 1.0, 4.0, 1.0, 5.0]) == (1.0, 2)
    @test findmin_turbo([5.0, 4.0, 3.0, 2.0, 1.0]) == (1.0, 5)
    @test findmin_turbo([1.0]) == (1.0, 1)
    @test findmin_turbo([10, 20, 5, 15]) == (5, 3)
    @test findmin_turbo([-1.0, -5.0, 3.0]) == (-5.0, 2)
end

@testset "_angle SVector{2}" begin
    u = SVector(1.0, 0.0)
    v = SVector(0.0, 1.0)
    @test _angle(u, v) ≈ 90° atol = 1e-10rad

    u = SVector(1.0, 0.0)
    v = SVector(-1.0, 0.0)
    @test _angle(u, v) ≈ 180° atol = 1e-10rad

    u = SVector(1.0, 0.0)
    v = SVector(1.0, 0.0)
    @test _angle(u, v) ≈ 0° atol = 1e-10rad

    u = SVector(1.0, 0.0)
    v = SVector(0.0, -1.0)
    @test _angle(u, v) ≈ -90° atol = 1e-10rad

    u = SVector(1.0, 1.0)
    v = SVector(-1.0, 1.0)
    @test _angle(u, v) ≈ 90° atol = 1e-10rad
end

@testset "_angle SVector{3}" begin
    u = SVector(1.0, 0.0, 0.0)
    v = SVector(0.0, 1.0, 0.0)
    @test _angle(u, v) ≈ 90° atol = 1e-10rad

    u = SVector(1.0, 0.0, 0.0)
    v = SVector(-1.0, 0.0, 0.0)
    @test _angle(u, v) ≈ 180° atol = 1e-10rad

    u = SVector(1.0, 0.0, 0.0)
    v = SVector(1.0, 0.0, 0.0)
    @test _angle(u, v) ≈ 0° atol = 1e-10rad

    u = SVector(1.0, 1.0, 1.0)
    v = SVector(-1.0, -1.0, -1.0)
    @test _angle(u, v) ≈ 180° atol = 1e-10rad

    u = SVector(0.0, 0.0, 1.0)
    v = SVector(0.0, 1.0, 0.0)
    @test _angle(u, v) ≈ 90° atol = 1e-10rad
end

@testset "_angle Vec" begin
    u = Vec(1.0, 0.0)
    v = Vec(0.0, 1.0)
    @test _angle(u, v) ≈ 90° atol = 1e-10rad

    u = Vec(1.0, 0.0)
    v = Vec(-1.0, 0.0)
    @test _angle(u, v) ≈ 180° atol = 1e-10rad

    u = Vec(1.0, 0.0, 0.0)
    v = Vec(0.0, 1.0, 0.0)
    @test _angle(u, v) ≈ 90° atol = 1e-10rad

    u = Vec(1.0, 1.0, 1.0)
    v = Vec(1.0, 1.0, 1.0)
    @test _angle(u, v) ≈ 0° atol = 1e-10rad
end

@testset "ranges_from_permutation" begin
    permutations = [[1, 2, 3], [4, 5], [6]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    @test ranges == [1:3, 4:5, 6:6]

    permutations = [[1], [2], [3]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    @test ranges == [1:1, 2:2, 3:3]

    permutations = [[1, 2, 3, 4, 5]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    @test ranges == [1:5]

    permutations = [[1, 2], [3, 4], [5, 6]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    @test ranges == [1:2, 3:4, 5:6]
end

@testset "many_permute! Array" begin
    arr = [1, 2, 3, 4, 5, 6]
    permutations = [[3, 2, 1], [2, 1], [1]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(arr, permutations, ranges)
    @test arr == [3, 2, 1, 5, 4, 6]

    arr = [10, 20, 30, 40]
    permutations = [[2, 1], [2, 1]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(arr, permutations, ranges)
    @test arr == [20, 10, 40, 30]

    arr = [1.0, 2.0, 3.0]
    permutations = [[1, 2, 3]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(arr, permutations, ranges)
    @test arr == [1.0, 2.0, 3.0]
end

@testset "many_permute! PointSet" begin
    points = Point.([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
    permutations = [[2, 1], [2, 1]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(points, permutations, ranges)
    expected = Point.([(2.0, 2.0), (1.0, 1.0), (4.0, 4.0), (3.0, 3.0)])
    @test all(points .== expected)

    points = Point.([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)])
    permutations = [[3, 2, 1]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(points, permutations, ranges)
    expected = Point.([(2.0, 2.0, 2.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)])
    @test all(points .== expected)
end
