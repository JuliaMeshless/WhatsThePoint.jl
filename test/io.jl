using WhatsThePoint
using Meshes
using FileIO
using LinearAlgebra
using Unitful
using Unitful: m, °, @u_str

@testitem "import_surface" begin
    # Test importing from STL file
    filepath = joinpath(@__DIR__, "data", "box.stl")
    points, normals, areas, mesh = import_surface(filepath)

    @test points isa Vector{<:Point}
    @test length(points) > 0
    @test normals isa Vector
    @test length(normals) == length(points)
    @test areas isa Vector
    @test length(areas) == length(points)
    @test mesh isa Meshes.Mesh

    # Verify normals are normalized
    for n in normals
        @test norm(n) ≈ 1.0 rtol=1e-6
    end

    # Verify areas are positive
    for a in areas
        @test Unitful.ustrip(a) > 0.0
    end
end

@testitem "export_cloud" begin
    # Create a simple point cloud
    boundary = PointBoundary(joinpath(@__DIR__, "data", "box.stl"))
    cloud = PointCloud(boundary)

    # Export to temporary file
    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_export")
        export_cloud(filename, cloud)

        # Verify VTK file was created (WriteVTK appends .vtu extension)
        @test isfile(filename * ".vtu")
    end
end

@testitem "FileIO.save" begin
    # Create a simple point cloud
    points = rand(Point, 10)
    boundary = PointBoundary(points)
    cloud = PointCloud(boundary)

    # Save to temporary file
    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "test_cloud.jld2")
        FileIO.save(filename, cloud)

        # Verify file was created
        @test isfile(filename)

        # Verify we can load it back
        loaded = FileIO.load(filename)
        @test haskey(loaded, "cloud")
        @test loaded["cloud"] isa PointCloud
        @test length(loaded["cloud"]) == length(cloud)
    end
end

@testitem "Round-trip: import -> create -> export -> verify" begin
    mktempdir() do tmpdir
        # Import surface from STL
        stl_path = joinpath(@__DIR__, "data", "box.stl")
        boundary = PointBoundary(stl_path)

        # Create point cloud
        cloud = PointCloud(boundary)
        original_length = length(cloud)

        # Export to VTK
        vtk_filename = joinpath(tmpdir, "roundtrip_test")
        export_cloud(vtk_filename, cloud)
        @test isfile(vtk_filename * ".vtu")

        # Save to JLD2
        jld2_filename = joinpath(tmpdir, "roundtrip_test.jld2")
        FileIO.save(jld2_filename, cloud)
        @test isfile(jld2_filename)

        # Load back and verify
        loaded = FileIO.load(jld2_filename)
        @test haskey(loaded, "cloud")
        loaded_cloud = loaded["cloud"]
        @test loaded_cloud isa PointCloud
        @test length(loaded_cloud) == original_length
        @test hassurface(loaded_cloud, :surface1)
    end
end
