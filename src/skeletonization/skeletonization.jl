# Centerline extraction using Antiga et al. (2003) algorithm
# Based on: "Centerline Computation and Geometric Analysis of Branching
# Tubular Surfaces With Application to Blood Vessel Modeling"

# ============================================================================
# Algorithm Type
# ============================================================================

"""
    abstract type AbstractCenterlineAlgorithm end

Base type for centerline extraction algorithms.
"""
abstract type AbstractCenterlineAlgorithm end

"""
    VoronoiCenterlines <: AbstractCenterlineAlgorithm

Voronoi diagram-based centerline extraction (Antiga et al. 2003).

This is the gold standard algorithm for blood vessel centerline computation.
It provides:
- Mathematically guaranteed medial axis centerlines
- Maximal inscribed sphere radius at every centerline point
- Robust handling of bifurcations

# Reference
Antiga, L., Ene-Iordache, B., & Remuzzi, A. (2003). Centerline computation
and geometric analysis of branching tubular surfaces with application to
blood vessel modeling. WSCG 2003.
"""
struct VoronoiCenterlines <: AbstractCenterlineAlgorithm end

# ============================================================================
# Include sub-modules
# ============================================================================

include("delaunay.jl")
include("voronoi.jl")
include("fast_marching.jl")
include("centerline.jl")

# ============================================================================
# Result Types
# ============================================================================

"""
    CenterlineResult{T}

Result of centerline computation containing one or more centerlines.

# Fields
- `centerlines::Vector{Centerline{T}}` - Computed centerlines from source to each target
- `voronoi::EmbeddedVoronoi{T}` - Embedded Voronoi diagram (for debugging/visualization)
- `source::SVector{3,Float64}` - Source point coordinates
- `targets::Vector{SVector{3,Float64}}` - Target point coordinates
"""
struct CenterlineResult{T<:AbstractFloat}
    centerlines::Vector{Centerline{T}}
    voronoi::EmbeddedVoronoi{T}
    source::SVector{3,Float64}
    targets::Vector{SVector{3,Float64}}
end

# Accessors
centerlines(cr::CenterlineResult) = cr.centerlines
Base.length(cr::CenterlineResult) = length(cr.centerlines)
Base.getindex(cr::CenterlineResult, i) = cr.centerlines[i]
Base.iterate(cr::CenterlineResult) = iterate(cr.centerlines)
Base.iterate(cr::CenterlineResult, state) = iterate(cr.centerlines, state)

# ============================================================================
# Surface Point Extraction
# ============================================================================

"""
    get_surface_points(surf::PointSurface{𝔼{3}}) -> Vector{Point}

Extract surface points from a PointSurface.
"""
function get_surface_points(surf::PointSurface{𝔼{3}})
    return collect(point(surf))
end

"""
    get_surface_points(filepath::String) -> (Vector{Point}, Mesh)

Extract mesh vertices from a surface mesh file.

Returns both the vertices and the original mesh (for inside testing).

Note: This extracts mesh VERTICES, not face centroids (which is what
`import_surface` returns for point cloud generation).
"""
function get_surface_points(filepath::String)
    geo = GeoIO.load(filepath)
    mesh = geo.geometry

    # Extract unique vertices from the mesh
    verts = collect(Meshes.vertices(mesh))

    return verts, mesh
end

# ============================================================================
# Main API
# ============================================================================

"""
    compute_centerlines(surface, source, targets; alg=VoronoiCenterlines()) -> CenterlineResult

Compute centerlines from a source point to one or more target points using
the Antiga et al. (2003) Voronoi-based algorithm.

Each centerline point has an associated **maximal inscribed sphere radius**,
which corresponds to the vessel lumen radius at that location.

# Arguments
- `surface` - Input surface, either:
  - `PointSurface{𝔼{3}}` - Point cloud with normals
  - `String` - Path to surface mesh file (STL, PLY, etc.)
- `source` - Source point (inlet) as:
  - `Point{𝔼{3}}` - Meshes.jl Point
  - `Tuple{Real,Real,Real}` - (x, y, z) coordinates
  - `SVector{3}` - StaticArrays vector
- `targets` - Target points (outlets) as Vector of above types
- `alg` - Algorithm to use (default: `VoronoiCenterlines()`)

# Returns
- `CenterlineResult` containing:
  - `centerlines` - Vector of `Centerline` objects
  - `voronoi` - Embedded Voronoi diagram (for visualization)
  - `source` - Source coordinates
  - `targets` - Target coordinates

# Example
```julia
# From PointSurface
surf = PointSurface("vessel.stl")
result = compute_centerlines(surf, (0.0, 0.0, 0.0), [(10.0, 0.0, 0.0)])

# From mesh file directly
result = compute_centerlines("vessel.stl", (0.0, 0.0, 0.0), [(10.0, 0.0, 0.0), (8.0, 5.0, 0.0)])

# Access centerline data
for cl in result.centerlines
    println("Arc length: \$(arc_length(cl))")
    println("Stenosis grade: \$(stenosis_grade(cl))%")

    for pt in cl
        println("  position: \$(position(pt)), radius: \$(radius(pt))")
    end
end
```

# Algorithm Overview

1. Extract surface points from input
2. Compute Delaunay tetrahedralization of surface points
3. Extract embedded Voronoi diagram (internal circumcenters only)
4. Run Fast Marching from source with F(x) = 1/R(x)
5. Backtrack from each target to source

The key insight is that Voronoi vertices (tetrahedron circumcenters) approximate
the medial axis, and their circumradii give maximal inscribed sphere radii.
"""
function compute_centerlines(
    surf::PointSurface{𝔼{3}},
    source,
    targets;
    alg::AbstractCenterlineAlgorithm=VoronoiCenterlines()
)
    return _compute_centerlines_voronoi(surf, source, targets)
end

function compute_centerlines(
    mesh_file::String,
    source,
    targets;
    alg::AbstractCenterlineAlgorithm=VoronoiCenterlines()
)
    return _compute_centerlines_voronoi_mesh(mesh_file, source, targets)
end

# ============================================================================
# Implementation
# ============================================================================

function _compute_centerlines_voronoi(
    surf::PointSurface{𝔼{3}},
    source,
    targets
)
    # Convert source/targets to standard format
    source_vec = _to_svector(source)
    target_vecs = [_to_svector(t) for t in targets]

    # 1. Extract surface points
    points = get_surface_points(surf)

    if length(points) < 4
        error("Need at least 4 surface points for centerline computation, got $(length(points))")
    end

    @info "Centerline computation: $(length(points)) surface points"

    # 2. Delaunay tetrahedralization
    @info "Computing Delaunay tetrahedralization..."
    delaunay = tetrahedralize_points(points)
    @info "  $(num_tetrahedra(delaunay)) tetrahedra"

    # 3. Extract embedded Voronoi diagram
    @info "Extracting embedded Voronoi diagram..."
    boundary = PointBoundary(surf)
    voronoi = extract_embedded_voronoi(delaunay, boundary)
    @info "  $(length(voronoi)) internal Voronoi vertices"

    if length(voronoi) == 0
        error("No internal Voronoi vertices found - check surface orientation and closure")
    end

    # 4-6. Fast march and backtrack
    return _compute_centerlines_impl(voronoi, source_vec, target_vecs)
end

function _compute_centerlines_voronoi_mesh(
    mesh_file::String,
    source,
    targets
)
    # Convert source/targets to standard format
    source_vec = _to_svector(source)
    target_vecs = [_to_svector(t) for t in targets]

    # 1. Extract mesh vertices
    @info "Loading mesh from $mesh_file..."
    points, mesh = get_surface_points(mesh_file)

    if length(points) < 4
        error("Need at least 4 mesh vertices for centerline computation, got $(length(points))")
    end

    @info "Centerline computation: $(length(points)) mesh vertices"

    # 2. Delaunay tetrahedralization
    @info "Computing Delaunay tetrahedralization..."
    delaunay = tetrahedralize_points(points)
    @info "  $(num_tetrahedra(delaunay)) tetrahedra"

    # 3. Extract embedded Voronoi diagram
    @info "Extracting embedded Voronoi diagram..."
    voronoi = extract_embedded_voronoi(delaunay, mesh)
    @info "  $(length(voronoi)) internal Voronoi vertices"

    if length(voronoi) == 0
        error("No internal Voronoi vertices found - check mesh orientation and closure")
    end

    # 4-6. Fast march and backtrack
    return _compute_centerlines_impl(voronoi, source_vec, target_vecs)
end

function _compute_centerlines_impl(
    voronoi::EmbeddedVoronoi{T},
    source::SVector{3,Float64},
    targets::Vector{SVector{3,Float64}}
) where {T}
    # 4. Find nearest Voronoi vertices to source/targets
    source_idx = find_nearest_vertex(voronoi, source)
    target_idxs = [find_nearest_vertex(voronoi, t) for t in targets]

    @info "Source vertex: $source_idx (radius = $(voronoi.vertices[source_idx].radius))"
    for (i, tidx) in enumerate(target_idxs)
        @info "Target $i vertex: $tidx (radius = $(voronoi.vertices[tidx].radius))"
    end

    # 5. Fast March from source
    @info "Running Fast Marching..."
    state = FastMarchingState(voronoi)
    fast_march!(state, voronoi, source_idx)
    @info "  Reached $(count_reached(state)) of $(length(voronoi)) vertices"

    # 6. Backtrack centerlines to each target
    @info "Backtracking centerlines..."
    centerlines = Centerline{T}[]

    for (i, target_idx) in enumerate(target_idxs)
        if !is_reachable(state, target_idx)
            @warn "Target $i (vertex $target_idx) is not reachable from source"
            continue
        end

        cl = backtrack_centerline(state, voronoi, target_idx)
        push!(centerlines, cl)
        @info "  Centerline $i: $(length(cl)) points, arc length = $(arc_length(cl))"
    end

    return CenterlineResult(centerlines, voronoi, source, targets)
end

# ============================================================================
# Helper Functions
# ============================================================================

function _to_svector(p::Point{𝔼{3}})
    coords = to(p)
    return SVector{3,Float64}(ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3]))
end

function _to_svector(t::Tuple{<:Real,<:Real,<:Real})
    return SVector{3,Float64}(Float64(t[1]), Float64(t[2]), Float64(t[3]))
end

function _to_svector(v::SVector{3,<:Real})
    return SVector{3,Float64}(v)
end

function _to_svector(v::AbstractVector{<:Real})
    if length(v) != 3
        error("Expected 3-element vector, got $(length(v)) elements")
    end
    return SVector{3,Float64}(v[1], v[2], v[3])
end

# ============================================================================
# Pretty Printing
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", cr::CenterlineResult)
    println(io, "CenterlineResult")
    println(io, "  $(length(cr.centerlines)) centerlines")
    println(io, "  $(length(cr.voronoi)) Voronoi vertices")
    println(io, "  source: $(cr.source)")
    for (i, cl) in enumerate(cr.centerlines)
        println(io, "  centerline $i: $(length(cl)) pts, length=$(arc_length(cl)), r=$(min_radius(cl))-$(max_radius(cl))")
    end
end
