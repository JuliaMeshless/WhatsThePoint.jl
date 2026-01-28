module WhatsThePoint

using Meshes
using CoordRefSystems
using LinearAlgebra
using StaticArrays
using StructArrays
using NearestNeighbors
using ChunkSplitters
using OhMyThreads: tmap, tmap!, tmapreduce
using OrderedCollections: LittleDict
using Statistics
using Random
using FileIO
using ProgressMeter
using GeoIO
using WriteVTK
using Graphs, SimpleWeightedGraphs
using Distances: Distances, Euclidean, evaluate

using Unitful

import Meshes: Manifold, Domain
import Meshes: centroid, boundingbox, discretize, to, crs
import Meshes: elements, nelements, lentype, normal, area
# re-export from Meshes.jl
export Point, coords, isinside, centroid, boundingbox, points
export KNearestSearch, BallSearch, MetricBall, search, searchdists

const spinner_icons = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
const Angle{T} = Union{Quantity{T, NoDims, typeof(u"rad")}, Quantity{T, NoDims, typeof(u"°")}}

include("utils.jl")
export metrics

include("geometry.jl")

# Octree spatial indexing
include("octree/traits.jl")
include("octree/spatial_octree.jl")
include("octree/geometric_utils.jl")
include("octree/triangle_mesh.jl")
include("octree/triangle_octree.jl")
export AbstractSpatialTree, AbstractOctree, AbstractQuadtree
export SpatialOctree
export InsertionStrategy, PointInsertion, GeometricInsertion
export SubdivisionCriterion, MaxElementsCriterion, SizeCriterion, AndCriterion
export should_subdivide, find_leaf, box_center, box_size, box_bounds, bounding_box
export subdivide!, balance_octree!, find_neighbor
export is_leaf, has_children, num_elements, all_leaves, all_boxes, needs_balancing
export Triangle, TriangleMesh, bbox_size, bbox_center, unique_points
export TriangleOctree, num_leaves, num_triangles
export closest_point_on_triangle, distance_point_triangle
export triangle_box_intersection, boxes_intersected_by_triangle

include("points.jl")
export emptyspace

include("shadow.jl")
export ShadowPoints

# Topology must come before surface/volume/cloud since they use topology types
include("topology.jl")
export AbstractTopology, NoTopology, KNNTopology, RadiusTopology
export neighbors, hastopology, set_topology, rebuild_topology!

include("surface.jl")
export AbstractSurface, PointSurface, SurfaceElement
export point, normal, area

include("volume.jl")
export PointVolume

include("boundary.jl")
export PointBoundary, surfaces, namedsurfaces, names, normals, areas, hassurface

include("cloud.jl")
export PointCloud, boundary, volume, topology

include("normals.jl")
export compute_normals, orient_normals!, update_normals!, compute_edge, compute_edges

include("neighbors.jl")

include("surface_operations.jl")
export generate_shadows, combine_surfaces!, split_surface!, rename!

include("isinside.jl")
export isinside

include("discretization/spacings.jl")
export AbstractSpacing, ConstantSpacing, LogLike, Power

include("discretization/discretization.jl")
export AbstractNodeGenerationAlgorithm, SlakKosec, VanDerSandeFornberg, FornbergFlyer
export discretize

include("repel.jl")
export repel

include("metrics.jl")

include("io.jl")
export import_surface, export_cloud, visualize, save

# visualize function is defined in WhatsThePointMakieExt when Makie is loaded
function visualize end

# Backward compatibility for deprecated Meshes.jl pointify
Base.@deprecate pointify(x) points(x)

######################################################
using PrecompileTools

@setup_workload begin
    using Unitful: m, °
    @compile_workload begin
        b = PointBoundary(joinpath(@__DIR__, "precompile_tools_dummy.stl"))
        split_surface!(b, 75°)
        cloud = discretize(b, ConstantSpacing(1m); alg = VanDerSandeFornberg())
    end
end

end # module
