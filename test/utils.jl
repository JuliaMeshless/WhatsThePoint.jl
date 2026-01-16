@testitem "findmin_turbo" setup = [TestData, CommonImports] begin
    @test WhatsThePoint.findmin_turbo([3.0, 1.0, 4.0, 1.0, 5.0]) == (1.0, 2)
    @test WhatsThePoint.findmin_turbo([5.0, 4.0, 3.0, 2.0, 1.0]) == (1.0, 5)
    @test WhatsThePoint.findmin_turbo([1.0]) == (1.0, 1)
    @test WhatsThePoint.findmin_turbo([10, 20, 5, 15]) == (5, 3)
    @test WhatsThePoint.findmin_turbo([-1.0, -5.0, 3.0]) == (-5.0, 2)
end

@testitem "_angle SVector{2}" setup = [TestData, CommonImports] begin
    u = SVector(1.0, 0.0)
    v = SVector(0.0, 1.0)
    @test WhatsThePoint._angle(u, v) ≈ 90° atol = 1.0e-10

    u = SVector(1.0, 0.0)
    v = SVector(-1.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 180° atol = 1.0e-10

    u = SVector(1.0, 0.0)
    v = SVector(1.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 0° atol = 1.0e-10

    u = SVector(1.0, 0.0)
    v = SVector(0.0, -1.0)
    @test WhatsThePoint._angle(u, v) ≈ -90° atol = 1.0e-10

    u = SVector(1.0, 1.0)
    v = SVector(-1.0, 1.0)
    @test WhatsThePoint._angle(u, v) ≈ 90° atol = 1.0e-10
end

@testitem "_angle SVector{3}" setup = [TestData, CommonImports] begin
    u = SVector(1.0, 0.0, 0.0)
    v = SVector(0.0, 1.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 90° atol = 1.0e-10

    u = SVector(1.0, 0.0, 0.0)
    v = SVector(-1.0, 0.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 180° atol = 1.0e-10

    u = SVector(1.0, 0.0, 0.0)
    v = SVector(1.0, 0.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 0° atol = 1.0e-10

    u = SVector(1.0, 1.0, 1.0)
    v = SVector(-1.0, -1.0, -1.0)
    @test WhatsThePoint._angle(u, v) ≈ 180° atol = 1.0e-10

    u = SVector(0.0, 0.0, 1.0)
    v = SVector(0.0, 1.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 90° atol = 1.0e-10
end

@testitem "_angle Vec" setup = [TestData, CommonImports] begin
    u = Vec(1.0, 0.0)
    v = Vec(0.0, 1.0)
    @test WhatsThePoint._angle(u, v) ≈ 90° atol = 1.0e-10

    u = Vec(1.0, 0.0)
    v = Vec(-1.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 180° atol = 1.0e-10

    u = Vec(1.0, 0.0, 0.0)
    v = Vec(0.0, 1.0, 0.0)
    @test WhatsThePoint._angle(u, v) ≈ 90° atol = 1.0e-10

    u = Vec(1.0, 1.0, 1.0)
    v = Vec(1.0, 1.0, 1.0)
    @test WhatsThePoint._angle(u, v) ≈ 0° atol = 1.0e-10
end

@testitem "ranges_from_permutation" setup = [TestData, CommonImports] begin
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

@testitem "many_permute! Array" setup = [TestData, CommonImports] begin
    arr = [1, 2, 3, 4, 5, 6]
    permutations = [[3, 2, 1], [5, 4], [6]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(arr, permutations, ranges)
    @test arr == [3, 2, 1, 5, 4, 6]

    arr = [10, 20, 30, 40]
    permutations = [[2, 1], [4, 3]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(arr, permutations, ranges)
    @test arr == [20, 10, 40, 30]

    arr = [1.0, 2.0, 3.0]
    permutations = [[1, 2, 3]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(arr, permutations, ranges)
    @test arr == [1.0, 2.0, 3.0]
end

@testitem "many_permute! Vector{Point}" setup = [TestData, CommonImports] begin
    pts = Point.([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
    permutations = [[2, 1], [4, 3]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(pts, permutations, ranges)
    expected = Point.([(2.0, 2.0), (1.0, 1.0), (4.0, 4.0), (3.0, 3.0)])
    @test all(pts .== expected)

    pts = Point.([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)])
    permutations = [[3, 2, 1]]
    ranges = WhatsThePoint.ranges_from_permutation(permutations)
    WhatsThePoint.many_permute!(pts, permutations, ranges)
    expected = Point.([(3.0, 3.0, 3.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0)])
    @test all(pts .== expected)
end
