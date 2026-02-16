@testsnippet CommonImports begin
    using WhatsThePoint
    using WhatsThePoint: volume, surfaces, topology
    import Meshes
    using Meshes: Euclidean, 🌐, Vec, SimpleMesh, connect, Box, PointSet, Geometry, to
    using Random
    using FileIO
    using Unitful
    using Unitful: m, °, @u_str
    using StaticArrays
    using LinearAlgebra
    using StructArrays
    using OrderedCollections: LittleDict
    using CoordRefSystems
end

@testmodule TestData begin
    const TEST_DIR = @__DIR__
    const BIFURCATION_PATH = joinpath(TEST_DIR, "data", "bifurcation.stl")
    const BOX_PATH = joinpath(TEST_DIR, "data", "box.stl")
end
