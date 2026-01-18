"""
Triangle mesh data structures for spatial indexing and octree construction.

This module provides lightweight, unit-free triangle representations optimized
for geometric queries and spatial indexing. All coordinates are pure Float64
with no Unitful dependencies in the core structures.
"""

using StaticArrays
using LinearAlgebra

"""
    struct Triangle{T<:Real}

A triangle defined by three vertices and a precomputed normal vector.
All coordinates are plain numbers (no units).

# Fields
- `v1::SVector{3,T}` - First vertex
- `v2::SVector{3,T}` - Second vertex  
- `v3::SVector{3,T}` - Third vertex
- `normal::SVector{3,T}` - Unit normal vector (right-hand rule: (v2-v1) × (v3-v1))

# Type Parameter
- `T<:Real` - Coordinate type (typically Float64)

# Examples
```julia
# Automatic normal computation
v1 = SVector(0.0, 0.0, 0.0)
v2 = SVector(1.0, 0.0, 0.0)
v3 = SVector(0.0, 1.0, 0.0)
tri = Triangle(v1, v2, v3)  # Normal computed as [0, 0, 1]

# Explicit normal (must be unit length)
n = SVector(0.0, 0.0, 1.0)
tri = Triangle(v1, v2, v3, n)
```
"""
struct Triangle{T<:Real}
    v1::SVector{3,T}
    v2::SVector{3,T}
    v3::SVector{3,T}
    normal::SVector{3,T}

    # Inner constructor with validation
    function Triangle(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T},
        normal::SVector{3,T}) where T<:Real
        # Validate normal is unit length
        n_mag = norm(normal)
        if !isapprox(n_mag, one(T), atol=1e-6)
            error("Triangle normal must be unit length, got norm=$n_mag")
        end
        return new{T}(v1, v2, v3, normal)
    end
end

# Outer constructor that computes normal automatically
"""
    Triangle(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T

Construct a triangle with automatic normal computation using right-hand rule.
Normal direction: (v2-v1) × (v3-v1), normalized to unit length.

Throws an error if vertices are collinear or duplicated (degenerate triangle).
"""
function Triangle(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T<:Real
    e1 = v2 - v1
    e2 = v3 - v1
    n = cross(e1, e2)

    # Check for degenerate triangle
    n_mag = norm(n)
    if n_mag < eps(T) * 100
        error("Degenerate triangle: vertices are collinear or duplicate")
    end

    normal = n / n_mag  # Normalize
    return Triangle(v1, v2, v3, normal)
end

"""
    struct TriangleMesh{T<:Real}

A collection of triangles with precomputed bounding box for spatial queries.
Lightweight wrapper around triangle data - no topology, no adjacency.

# Fields
- `triangles::Vector{Triangle{T}}` - Array of triangles
- `bbox_min::SVector{3,T}` - Minimum corner of bounding box
- `bbox_max::SVector{3,T}` - Maximum corner of bounding box

# Type Parameter
- `T<:Real` - Coordinate type (typically Float64)

# Examples
```julia
# Manual construction
triangles = [Triangle(v1, v2, v3), Triangle(v4, v5, v6)]
mesh = TriangleMesh(triangles)  # Automatic bbox computation

# Load from STL file (optional, requires GeoIO)
mesh = TriangleMesh("bunny.stl")
```
"""
struct TriangleMesh{T<:Real}
    triangles::Vector{Triangle{T}}
    bbox_min::SVector{3,T}
    bbox_max::SVector{3,T}

    # Inner constructor with validation
    function TriangleMesh(triangles::Vector{Triangle{T}},
        bbox_min::SVector{3,T},
        bbox_max::SVector{3,T}) where T<:Real
        isempty(triangles) && error("TriangleMesh cannot be empty")

        # Validate bounding box
        if any(bbox_max .<= bbox_min)
            error("Invalid bounding box: max must be > min in all dimensions")
        end

        return new{T}(triangles, bbox_min, bbox_max)
    end
end

"""
    TriangleMesh(triangles::Vector{Triangle{T}}) where T

Construct TriangleMesh with automatic bounding box computation from all vertices.
"""
function TriangleMesh(triangles::Vector{Triangle{T}}) where T<:Real
    isempty(triangles) && error("Cannot create TriangleMesh from empty triangle list")

    # Compute bounding box from all vertices
    all_coords = [tri.v1 for tri in triangles]
    append!(all_coords, [tri.v2 for tri in triangles])
    append!(all_coords, [tri.v3 for tri in triangles])

    # Find min/max in each dimension
    bbox_min = SVector{3,T}(
        minimum(v[1] for v in all_coords),
        minimum(v[2] for v in all_coords),
        minimum(v[3] for v in all_coords)
    )

    bbox_max = SVector{3,T}(
        maximum(v[1] for v in all_coords),
        maximum(v[2] for v in all_coords),
        maximum(v[3] for v in all_coords)
    )

    # Handle planar meshes by adding small epsilon in degenerate dimensions
    eps_val = max(eps(T) * 100, 1e-10)
    bbox_size = bbox_max .- bbox_min

    # If any dimension has zero size (planar mesh), add small thickness
    bbox_min = SVector{3,T}(
        bbox_size[1] == 0 ? bbox_min[1] - eps_val : bbox_min[1],
        bbox_size[2] == 0 ? bbox_min[2] - eps_val : bbox_min[2],
        bbox_size[3] == 0 ? bbox_min[3] - eps_val : bbox_min[3]
    )

    bbox_max = SVector{3,T}(
        bbox_size[1] == 0 ? bbox_max[1] + eps_val : bbox_max[1],
        bbox_size[2] == 0 ? bbox_max[2] + eps_val : bbox_max[2],
        bbox_size[3] == 0 ? bbox_max[3] + eps_val : bbox_max[3]
    )

    return TriangleMesh(triangles, bbox_min, bbox_max)
end

# Convenience accessors
Base.length(mesh::TriangleMesh) = length(mesh.triangles)
Base.getindex(mesh::TriangleMesh, i::Int) = mesh.triangles[i]
Base.iterate(mesh::TriangleMesh, state=1) =
    state > length(mesh.triangles) ? nothing : (mesh.triangles[state], state + 1)

"""
    bbox_size(mesh::TriangleMesh)

Return the size of the bounding box as SVector{3,T}.
"""
bbox_size(mesh::TriangleMesh) = mesh.bbox_max - mesh.bbox_min

"""
    unique_points(mesh::TriangleMesh)

Extract unique vertices from all triangles in the mesh.
Returns a vector of unique SVector{3,T} points.
"""
function unique_points(mesh::TriangleMesh{T}) where T
    points = Set{SVector{3,T}}()
    for tri in mesh.triangles
        push!(points, tri.v1)
        push!(points, tri.v2)
        push!(points, tri.v3)
    end
    return collect(points)
end

"""
    bbox_center(mesh::TriangleMesh)

Return the center of the bounding box as SVector{3,T}.
"""
bbox_center(mesh::TriangleMesh) = (mesh.bbox_min + mesh.bbox_max) / 2

# ============================================================================
# OPTIONAL: STL File Loading (requires GeoIO and Meshes)
# This is the ONLY place where Meshes.jl/Unitful are touched!
# ============================================================================

# Try to load GeoIO/Meshes for STL support (optional)
const HAS_GEOIO = try
    @eval using GeoIO
    @eval using Meshes
    true
catch
    false
end

if HAS_GEOIO
    """
        TriangleMesh(filepath::String)

    Load triangle mesh from STL file. **Strips units and converts to plain Float64 coordinates.**

    **CRITICAL**: This is the ONLY function that touches Meshes.jl/Unitful!
    Core Triangle and TriangleMesh structures remain completely unit-free.

    # Arguments
    - `filepath::String` - Path to STL file

    # Returns
    - `TriangleMesh{Float64}` with **pure numeric coordinates** (no units)

    # Example
    ```julia
    mesh = TriangleMesh("test/data/box.stl")
    println("Loaded \$(length(mesh)) triangles")
    # mesh.triangles[1].v1 is SVector{3,Float64}, NOT with units!
    ```
    """
    function TriangleMesh(filepath::String)
        # Load using existing GeoIO infrastructure
        geo = GeoIO.load(filepath)
        mesh_data = geo.geometry

        # Extract triangles and STRIP UNITS IMMEDIATELY
        triangles = Triangle{Float64}[]

        for elem in Meshes.elements(mesh_data)
            # Get vertices (Meshes.jl Points with units) - ONLY in this scope!
            verts = Meshes.vertices(elem)

            if length(verts) != 3
                @warn "Skipping non-triangular element with $(length(verts)) vertices"
                continue
            end

            # Convert to pure tuples and strip units
            v1_tuple = Meshes.to(verts[1])
            v2_tuple = Meshes.to(verts[2])
            v3_tuple = Meshes.to(verts[3])

            # Convert to pure SVector{3,Float64} - NO UNITS!
            v1 = SVector{3,Float64}(ustrip.(v1_tuple))
            v2 = SVector{3,Float64}(ustrip.(v2_tuple))
            v3 = SVector{3,Float64}(ustrip.(v3_tuple))
            # After this point, units are gone forever!

            # Create triangle (auto-computes normal)
            try
                tri = Triangle(v1, v2, v3)
                push!(triangles, tri)
            catch e
                if isa(e, ErrorException) && occursin("degenerate", lowercase(e.msg))
                    @warn "Skipping degenerate triangle at vertices: $v1, $v2, $v3"
                    continue
                else
                    rethrow(e)
                end
            end
        end

        if isempty(triangles)
            error("No valid triangles found in $filepath")
        end

        # Use automatic bounding box constructor
        return TriangleMesh(triangles)
    end
else
    # Fallback if GeoIO not available
    function TriangleMesh(filepath::String)
        error("GeoIO.jl not available. Install it with: using Pkg; Pkg.add(\"GeoIO\")")
    end
end
