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

@testitem "save VTK cloud" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    cloud = PointCloud(boundary)

    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_export")
        save(filename, cloud; format = :vtk)

        @test isfile(filename * ".vtu")
    end
end

@testitem "export_vtk with solution fields" setup = [TestData, CommonImports] begin
    using StaticArrays: SVector
    using Unitful: Pa

    boundary = PointBoundary(TestData.BOX_PATH)
    cloud = PointCloud(boundary)
    n = length(cloud)

    mktempdir() do tmpdir
        # bare geometry export (point_type + surface_id + normals)
        geo = joinpath(tmpdir, "geo")
        export_vtk(geo, cloud)
        @test isfile(geo * ".vtu")

        # scalar (unitful) + vector solution fields, ordered like points(cloud)
        sol = joinpath(tmpdir, "sol")
        p = [(101325.0 + i) * Pa for i in 1:n]
        u = [SVector(0.1i, 0.0, -0.1i) for i in 1:n]
        export_vtk(sol, cloud; fields = ("p" => p, "U" => u))
        @test isfile(sol * ".vtu")

        # a single bare pair is accepted (no tuple needed)
        one = joinpath(tmpdir, "one")
        export_vtk(one, cloud; fields = "p" => p)
        @test isfile(one * ".vtu")

        # mismatched field length is rejected loudly
        @test_throws DimensionMismatch export_vtk(
            joinpath(tmpdir, "bad"), cloud; fields = ("bad" => [1.0, 2.0, 3.0],),
        )
    end
end

@testitem "save VTK boundary" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)

    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_boundary")
        save(filename, boundary; format = :vtk)

        @test isfile(filename * ".vtu")
    end
end

@testitem "save VTK surface" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    surf = boundary[:surface1]

    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_surface")
        save(filename, surf; format = :vtk)

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
        save(vtk_filename, cloud; format = :vtk)
        @test isfile(vtk_filename * ".vtu")

        jld2_filename = joinpath(tmpdir, "roundtrip_test.jld2")
        save(jld2_filename, cloud)
        @test isfile(jld2_filename)

        loaded = FileIO.load(jld2_filename)
        @test haskey(loaded, "cloud")
        loaded_cloud = loaded["cloud"]
        @test loaded_cloud isa PointCloud
        @test length(loaded_cloud) == original_length
        @test hassurface(loaded_cloud, :surface1)
    end
end
