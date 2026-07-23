@testsnippet CommonImports begin
    using WhatsThePoint
    using WhatsThePoint: volume, surfaces, topology
    import Meshes
    using Meshes: Euclidean, 🌐, Vec, SimpleMesh, connect, Box, PointSet, to
    using Random
    using FileIO
    using GeoIO
    using Unitful
    using Unitful: m, °, @u_str
    using StaticArrays
    using LinearAlgebra
    using StructArrays
    using OrderedCollections: LittleDict
    using CoordRefSystems

    _relative_spacing(domain; divisor = 8) = ConstantSpacing(
        norm(Meshes.boundingbox(domain).max - Meshes.boundingbox(domain).min) / divisor
    )
end

@testmodule TestData begin
    const TEST_DIR = @__DIR__
    const BIFURCATION_PATH = joinpath(TEST_DIR, "data", "bifurcation.stl")
    const BOX_PATH = joinpath(TEST_DIR, "data", "box.stl")
end

@testmodule STLHelpers begin
    import Meshes
    using Unitful: ustrip

    """Write a triangle `SimpleMesh` as a minimal ASCII STL (loads back as Float64,
    unlike the binary Float32 fixtures — handy for mactype-promotion tests)."""
    function write_ascii_stl(path, mesh)
        open(path, "w") do io
            println(io, "solid gen")
            for elem in Meshes.elements(mesh)
                vs = Meshes.vertices(elem)
                n = Meshes.normal(elem)
                println(io, " facet normal ", join(ustrip.(n), " "))
                println(io, "  outer loop")
                for v in vs
                    println(io, "   vertex ", join(ustrip.(Meshes.to(v)), " "))
                end
                println(io, "  endloop")
                println(io, " endfacet")
            end
            println(io, "endsolid gen")
        end
        return path
    end
end

@testmodule OctreeTestData begin
    using Meshes: Point, SimpleMesh, connect
    import Meshes

    """Unit cube mesh: 8 vertices, 12 triangles (2 per face, CCW from outside).
    `T` sets the coordinate machine type (default Float64)."""
    function unit_cube_mesh(::Type{T} = Float64) where {T}
        pts = [
            Point(T(0), T(0), T(0)), Point(T(1), T(0), T(0)),
            Point(T(1), T(1), T(0)), Point(T(0), T(1), T(0)),
            Point(T(0), T(0), T(1)), Point(T(1), T(0), T(1)),
            Point(T(1), T(1), T(1)), Point(T(0), T(1), T(1)),
        ]
        connec = [
            connect((1, 3, 2), Meshes.Triangle), connect((1, 4, 3), Meshes.Triangle),
            connect((5, 6, 7), Meshes.Triangle), connect((5, 7, 8), Meshes.Triangle),
            connect((1, 2, 6), Meshes.Triangle), connect((1, 6, 5), Meshes.Triangle),
            connect((3, 4, 8), Meshes.Triangle), connect((3, 8, 7), Meshes.Triangle),
            connect((1, 5, 8), Meshes.Triangle), connect((1, 8, 4), Meshes.Triangle),
            connect((2, 3, 7), Meshes.Triangle), connect((2, 7, 6), Meshes.Triangle),
        ]
        return SimpleMesh(pts, connec)
    end

    """Simple square mesh: 4 vertices, 2 triangles in xy-plane."""
    function simple_square_mesh()
        pts = [
            Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0),
        ]
        connec = [
            connect((1, 2, 3), Meshes.Triangle),
            connect((1, 3, 4), Meshes.Triangle),
        ]
        return SimpleMesh(pts, connec)
    end
end
