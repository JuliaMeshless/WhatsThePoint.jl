@testitem "PointVolume Constructors" setup=[TestData, CommonImports] begin
    N = 10

    @testset "Empty constructor" begin
        vol = PointVolume{🌐,Cartesian{NoDatum}}()
        @test vol isa PointVolume
        @test isempty(vol)
        @test length(vol) == 0
    end

    @testset "PointSet constructor" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(PointSet(points))
        @test vol isa PointVolume
        @test length(vol) == N
        @test !isempty(vol)
    end

    @testset "Vector constructor" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        @test vol isa PointVolume
        @test length(vol) == N
        @test parent(vol) == PointSet(points)
    end
end

@testitem "PointVolume Base Methods" setup=[TestData, CommonImports] begin
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
        vol_empty = PointVolume{🌐,Cartesian{NoDatum}}()
        @test isempty(vol_empty)

        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        @test !isempty(vol)
    end

    @testset "parent" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        @test parent(vol) == PointSet(points)
        @test parent(vol) isa PointSet
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

@testitem "PointVolume Coordinate and Geometry Methods" setup=[TestData, CommonImports] begin
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
        points = [Point(1.0, 1.0), Point(3.0, 3.0), Point(5.0, 5.0)]
        vol = PointVolume(points)
        c = centroid(vol)
        @test c isa Point
        expected = centroid(PointSet(points))
        @test to(c) == to(expected)
    end

    @testset "boundingbox" begin
        points = [Point(1.0, 1.0), Point(3.0, 3.0), Point(5.0, 5.0)]
        vol = PointVolume(points)
        bbox = boundingbox(vol)
        @test bbox isa Box
        expected = boundingbox(PointSet(points))
        @test bbox == expected
    end
end

@testitem "PointVolume Meshes.pointify" setup=[TestData, CommonImports] begin
    N = 10

    @testset "pointify returns points" begin
        points = [Point(Float64(i), Float64(i)) for i in 1:N]
        vol = PointVolume(points)
        pointified = Meshes.pointify(vol)
        @test pointified isa Vector{<:Point}
        @test pointified == points
    end

    @testset "pointify empty volume" begin
        vol = PointVolume{🌐,Cartesian{NoDatum}}()
        pointified = Meshes.pointify(vol)
        @test isempty(pointified)
    end
end

@testitem "PointVolume Pretty Printing" setup=[TestData, CommonImports] begin
    N = 10
    points = [Point(Float64(i), Float64(i)) for i in 1:N]
    vol = PointVolume(points)

    io = IOBuffer()
    show(io, MIME("text/plain"), vol)
    output = String(take!(io))
    @test occursin("PointVolume", output)
    @test occursin("Number of points: $N", output)

    io = IOBuffer()
    show(io, vol)
    output = String(take!(io))
    @test occursin("PointVolume", output)
end
