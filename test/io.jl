@testitem "import_surface" setup = [TestData, CommonImports] begin
    filepath = TestData.BOX_PATH
    points, normals, areas, mesh = import_surface(filepath, u"m")

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
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
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

    boundary = PointBoundary(TestData.BOX_PATH, u"m")
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

        # the surface-id legend is opt-in: silent by default, printed with
        # verbose = true
        function capture_stdout(f)
            old_stdout = stdout
            rd, wr = redirect_stdout()
            f()
            redirect_stdout(old_stdout)
            close(wr)
            out = read(rd, String)
            close(rd)
            return out
        end
        @test capture_stdout(() -> export_vtk(joinpath(tmpdir, "quiet"), cloud)) == ""
        loud = capture_stdout(
            () -> export_vtk(joinpath(tmpdir, "loud"), cloud; verbose = true)
        )
        @test occursin("surface_id 1", loud)
    end
end

@testitem "save VTK boundary" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")

    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_boundary")
        save(filename, boundary; format = :vtk)

        @test isfile(filename * ".vtu")
    end
end

@testitem "save VTK surface" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH, u"m")
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
        boundary = PointBoundary(TestData.BOX_PATH, u"m")

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

@testitem "geometry_info reports raw bounding boxes" setup = [CommonImports, TestData] begin
    info = geometry_info(TestData.BOX_PATH; verbose = false)
    @test length(info) == 1
    @test info[1].file == TestData.BOX_PATH
    @test info[1].extent == info[1].max .- info[1].min
    ref = boundingbox(Meshes.vertices(GeoIO.load(TestData.BOX_PATH).geometry))
    @test info[1].min == Tuple(ustrip.(to(minimum(ref))))
    @test info[1].max == Tuple(ustrip.(to(maximum(ref))))

    info2 = geometry_info(TestData.BOX_PATH, TestData.BIFURCATION_PATH; verbose = false)
    @test length(info2) == 2
    @test info2[2].file == TestData.BIFURCATION_PATH

    old_stdout = stdout
    rd, wr = redirect_stdout()
    geometry_info(TestData.BOX_PATH)
    redirect_stdout(old_stdout)
    close(wr)
    single = read(rd, String)
    close(rd)
    @test occursin("box.stl", single)
    @test occursin("extent:", single)
    @test !occursin("Union", single)

    old_stdout = stdout
    rd, wr = redirect_stdout()
    geometry_info(TestData.BOX_PATH, TestData.BIFURCATION_PATH)
    redirect_stdout(old_stdout)
    close(wr)
    multi = read(rd, String)
    close(rd)
    @test occursin("box.stl", multi)
    @test occursin("bifurcation.stl", multi)
    @test occursin("Union", multi)
end

@testitem "import_mesh reinterprets units without conversion" setup = [CommonImports, TestData] begin
    r = import_mesh(TestData.BOX_PATH, u"mm")
    @test r isa SimpleMesh
    @test unit(to(first(Meshes.vertices(r)))[1]) == u"mm"

    # raw numbers unchanged — reinterpretation, not conversion
    info = geometry_info(TestData.BOX_PATH; verbose = false)
    box = boundingbox(Meshes.vertices(r))
    @test Tuple(ustrip.(to(minimum(box)))) == info[1].min
    @test Tuple(ustrip.(to(maximum(box)))) == info[1].max

    @test PointBoundary(r) isa PointBoundary
    @test TriangleOctree(r) isa TriangleOctree

    @test_throws ArgumentError import_mesh(TestData.BOX_PATH, u"s")
    # unit is now required on every path-taking entry point
    @test_throws ArgumentError PointBoundary(TestData.BOX_PATH)
end

@testitem "unit reinterpretation preserves topology and mactype" setup = [CommonImports, OctreeTestData] begin
    cube = OctreeTestData.unit_cube_mesh()
    r = WhatsThePoint._reinterpret_unit(cube, u"mm")
    @test Meshes.topology(r) === Meshes.topology(cube)
    @test ustrip.(to(last(Meshes.vertices(r)))) == ustrip.(to(last(Meshes.vertices(cube))))
    @test to(maximum(boundingbox(Meshes.vertices(r))))[1] == 1.0u"mm"
    b = PointBoundary(r)
    @test unit(first(area(first(surfaces(b))))) == u"mm^2"

    mesh32 = OctreeTestData.unit_cube_mesh(Float32)
    r32 = WhatsThePoint._reinterpret_unit(mesh32, u"mm")
    @test CoordRefSystems.mactype(Meshes.crs(r32)) === Float32
    @test ustrip(to(first(Meshes.vertices(r32)))[1]) isa Float32
    @test CoordRefSystems.mactype(Meshes.crs(first(points(PointBoundary(r32))))) === Float32
end
