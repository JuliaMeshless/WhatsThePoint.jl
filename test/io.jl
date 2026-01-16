@testitem "import_surface" setup = [TestData, CommonImports] begin
    filepath = TestData.BOX_PATH
    points, normals, areas, mesh = import_surface(filepath)

    @test points isa Vector{<:Point}
    @test length(points) > 0
    @test normals isa Vector
    @test length(normals) == length(points)
    @test areas isa Vector
    @test length(areas) == length(points)
    @test mesh isa Meshes.Mesh

    @test all(n -> isapprox(norm(n), 1.0; rtol = 1.0e-6), normals)

    @test all(a -> Unitful.ustrip(a) > 0, areas)
end

@testitem "export_cloud" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    cloud = PointCloud(boundary)

    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_export")
        export_cloud(filename, cloud)

        @test isfile(filename * ".vtu")
    end
end

@testitem "FileIO.save and load" setup = [TestData, CommonImports] begin
    points = rand(Point, 10)
    boundary = PointBoundary(points)
    cloud = PointCloud(boundary)

    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_cloud.jld2")
        FileIO.save(filename, cloud)

        @test isfile(filename)

        loaded = FileIO.load(filename)
        @test haskey(loaded, "cloud")
        @test loaded["cloud"] isa PointCloud
        @test length(loaded["cloud"]) == length(cloud)
    end
end

@testitem "IO Round-trip" setup = [TestData, CommonImports] begin
    mktempdir() do tmpdir
        boundary = PointBoundary(TestData.BOX_PATH)

        cloud = PointCloud(boundary)
        original_length = length(cloud)

        vtk_filename = joinpath(tmpdir, "roundtrip_test")
        export_cloud(vtk_filename, cloud)
        @test isfile(vtk_filename * ".vtu")

        jld2_filename = joinpath(tmpdir, "roundtrip_test.jld2")
        FileIO.save(jld2_filename, cloud)
        @test isfile(jld2_filename)

        loaded = FileIO.load(jld2_filename)
        @test haskey(loaded, "cloud")
        loaded_cloud = loaded["cloud"]
        @test loaded_cloud isa PointCloud
        @test length(loaded_cloud) == original_length
        @test hassurface(loaded_cloud, :surface1)
    end
end
