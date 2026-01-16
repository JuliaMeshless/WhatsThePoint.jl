@testitem "PointVolume Constructors" setup = [TestData, CommonImports] begin
    N = 10

    @testset "Empty constructor" begin
        vol = PointVolume{🌐, Cartesian{NoDatum}}()
        @test vol isa PointVolume
        @test isempty(vol)
        @test length(vol) == 0
    end

    @testset "Vector constructor" begin
        pts = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(pts)
        @test vol isa PointVolume
        @test length(vol) == N
        @test !isempty(vol)
        @test parent(vol) == pts
    end
end

@testitem "PointVolume Base Methods" setup = [TestData, CommonImports] begin
    N = 10

    @testset "length and size" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        @test length(vol) == N
        @test size(vol) == (N,)
    end

    @testset "getindex" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        @test vol[1] == points[1]
        @test vol[end] == points[end]
        @test vol[1:3] == points[1:3]
    end

    @testset "iterate" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        collected = collect(vol)
        @test collected == points

        for (i, p) in enumerate(vol)
            @test p == points[i]
        end
    end

    @testset "isempty" begin
        vol_empty = PointVolume{🌐, Cartesian{NoDatum}}()
        @test isempty(vol_empty)

        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        @test !isempty(vol)
    end

    @testset "parent" begin
        pts = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(pts)
        @test parent(vol) == pts
        @test parent(vol) isa AbstractVector{<:Point}
    end

    @testset "filter" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:10]
        vol = PointVolume(points)
        filtered_vol = filter(p -> to(p)[1] > 5.0m, vol)
        @test length(filtered_vol) == 5
        for p in filtered_vol
            @test to(p)[1] > 5.0m
        end
        # Original should be unchanged (immutable)
        @test length(vol) == 10
    end
end

@testitem "PointVolume Coordinate and Geometry Methods" setup = [TestData, CommonImports] begin
    N = 10

    @testset "to" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        coords = to(vol)
        @test coords isa Vector
        @test length(coords) == N
        @test coords == to.(points)
    end

    @testset "centroid" begin
        pts = [Point(1.0, 1.0), Point(3.0, 3.0), Point(5.0, 5.0)]
        vol = PointVolume(pts)
        c = centroid(vol)
        @test c isa Point
        expected = centroid(pts)
        @test to(c) == to(expected)
    end

    @testset "boundingbox" begin
        pts = [Point(1.0, 1.0), Point(3.0, 3.0), Point(5.0, 5.0)]
        vol = PointVolume(pts)
        bbox = boundingbox(vol)
        @test bbox isa Box
        expected = boundingbox(pts)
        @test bbox == expected
    end
end

@testitem "PointVolume points()" setup = [TestData, CommonImports] begin
    N = 10

    @testset "points returns points" begin
        pts = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(pts)
        result = points(vol)
        @test result isa AbstractVector{<:Point}
        @test result == pts
    end

    @testset "points empty volume" begin
        vol = PointVolume{🌐, Cartesian{NoDatum}}()
        result = points(vol)
        @test isempty(result)
    end
end

@testitem "PointVolume Pretty Printing" setup = [TestData, CommonImports] begin
    N = 10
    points = [Point(Float64(i), Float64(i)) for i in 1:N]
    vol = PointVolume(points)

    io = IOBuffer()
    show(io, MIME("text/plain"), vol)
    output = String(take!(io))
    @test occursin("PointVolume", output)
    @test occursin("Number of points: $N", output)
    @test occursin("Topology: NoTopology", output)

    io = IOBuffer()
    show(io, vol)
    output = String(take!(io))
    @test occursin("PointVolume", output)
end
