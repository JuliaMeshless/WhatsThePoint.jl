module WhatsThePoint

using Meshes
using CoordRefSystems
using LinearAlgebra
using SparseArrays
using StaticArrays
using StructArrays
using NearestNeighbors
using ThreadsX
using ChunkSplitters
using OhMyThreads: tmap, tmap!, tmapreduce
using OrderedCollections: LittleDict
using Statistics
using Random
using FileIO
using ProgressMeter
using GeoIO
using WriteVTK
using Makie: Makie
using Makie:
    Figure,
    Axis,
    Axis3,
    scatter,
    meshscatter,
    meshscatter!,
    meshscatter!,
    arrows,
    arrows!,
    DataAspect
using Graphs, SimpleWeightedGraphs

using Unitful

import Meshes: Manifold, Domain
import Meshes: centroid, boundingbox, discretize, to
import Meshes: elements, nelements, lentype, normal, area, pointify
# re-export from Meshes.jl
export Point, coords, isinside, centroid, boundingbox, pointify
export KNearestSearch, BallSearch, search, searchdists

const spinner_icons = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
const Angle{T} = Union{Quantity{T,NoDims,typeof(u"rad")},Quantity{T,NoDims,typeof(u"°")}}

include("utils.jl")
export metrics

include("points.jl")
export emptyspace

include("shadow.jl")
export ShadowPoints

include("surface.jl")
export AbstractSurface, PointSurface, SurfaceElement
export point, normal, area

include("volume.jl")
export PointVolume

include("boundary.jl")
export PointBoundary, surfaces, names, normals, areas, hassurface

include("cloud.jl")
export PointCloud, boundary, volume

include("normals.jl")
export compute_normals, orient_normals!, update_normals!, compute_edge, compute_edges

include("neighbors.jl")

include("surface_operations.jl")
export generate_shadows, add_surface!, combine_surfaces!, split_surface!, rename!

include("isinside.jl")
export isinside

include("discretization/spacings.jl")
export AbstractSpacing, ConstantSpacing, LogLike, Power

include("discretization/discretization.jl")
export AbstractNodeGenerationAlgorithm, SlakKosec, VanDerSandeFornberg, FornbergFlyer
export discretize!, discretize

include("repel.jl")
export repel!

include("metrics.jl")

include("io.jl")
export import_surface, export_cloud, visualize, visualize_normals, save

include("visualize.jl")

######################################################
using PrecompileTools

@setup_workload begin
    using Unitful: m, °
    @compile_workload begin
        b = PointBoundary(joinpath(@__DIR__, "precompile_tools_dummy.stl"))
        split_surface!(b, 75°)
        cloud = discretize(b, ConstantSpacing(1m); alg=VanDerSandeFornberg())
        visualize(cloud; markersize=0.01)
    end
end

end # module
