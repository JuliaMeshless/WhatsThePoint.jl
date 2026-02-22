#=============================================================================
TYPES AND DATA STRUCTURES
=============================================================================#

"""
    TriangleOctree{M<:Manifold, C<:CRS, T<:Real}

An octree-based spatial index for efficient triangle mesh queries.

# Fields
- `tree::SpatialOctree{Int,T}`: Underlying octree storing triangle indices
- `mesh::SimpleMesh{M,C}`: Original Meshes.jl mesh (single source of truth for triangle data)
- `leaf_classification::Union{Nothing, Vector{Int8}}`: Classification of empty leaves
  - `0`: Exterior (outside mesh)
  - `1`: Boundary (near surface but empty)
  - `2`: Interior (inside mesh)

# Performance
For a mesh with M triangles:
- Construction: O(M log M) - distribute triangles to boxes
- Query: O(log M + k) where k ≈ 10-50 triangles per leaf
- Memory: O(N) for octree nodes (mesh data accessed on-the-fly)

# Example
```julia
# Option 1: From STL file (simplest)
octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)

# Option 2: From SimpleMesh
mesh = GeoIO.load("model.stl").geometry
octree = TriangleOctree(mesh; h_min=0.01, classify_leaves=true)

# Fast queries (100-1000x speedup over brute force)
point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)
```

# Workflow with PointBoundary
Since PointBoundary discards mesh topology, build both from the same mesh:
```julia
mesh = GeoIO.load("model.stl").geometry
boundary = PointBoundary(mesh)
octree = TriangleOctree(mesh; h_min=0.01, classify_leaves=true)
cloud = discretize(boundary, spacing; alg=OctreeRandom(octree))
```
"""
struct TriangleOctree{M<:Manifold,C<:CRS,T<:Real}
    tree::SpatialOctree{Int,T}
    mesh::SimpleMesh{M,C}
    leaf_classification::Union{Nothing,Vector{Int8}}
end

# Leaf classification tags
const LEAF_UNKNOWN::Int8 = -1
const LEAF_EXTERIOR::Int8 = 0
const LEAF_BOUNDARY::Int8 = 1
const LEAF_INTERIOR::Int8 = 2

# Numeric constants for bounding box and classification
const _BBOX_EXPANSION = 1.02        # 2% buffer for numerical safety
const _DEGENERATE_EPS = 1.0e-10       # Min bbox extent for degenerate dims
const _CLASSIFICATION_INSET = 1.0e-9  # Corner inset to avoid on-surface probes
const _CLASSIFY_TOLERANCE_REL = 1.0e-6  # Relative tolerance for classification (scaled by h)
const _CLASSIFY_TOLERANCE_ABS = 1.0e-8  # Absolute tolerance floor for classification

"""
    NearestTriangleState{T<:Real}

Mutable state for nearest-triangle search, replacing 3 separate `Ref` allocations
with a single struct (1 heap allocation instead of 3, enables potential SROA).
"""
mutable struct NearestTriangleState{T<:Real}
    best_dist_sq::T
    closest_idx::Int
    closest_pt::SVector{3,T}
end

NearestTriangleState{T}(point::SVector{3,T}) where {T<:Real} =
    NearestTriangleState{T}(typemax(T), 0, point)

#=============================================================================
UTILITIES
=============================================================================#

"""
    _normalize_normal(::Type{T}, n_vec) where {T} -> SVector{3,T}

Extract and normalize a Meshes.jl Vec normal to a unit SVector{3,T}.
"""
@inline function _normalize_normal(::Type{T}, n_vec) where {T}
    n = SVector{3,T}(ustrip(n_vec[1]), ustrip(n_vec[2]), ustrip(n_vec[3]))
    n_mag = norm(n)
    if n_mag < eps(T) * 100
        error("Degenerate triangle: zero normal")
    end
    return n / n_mag
end

"""
    _extract_vertex(::Type{T}, vert) where {T} -> SVector{3,T}

Extract coordinates from a Meshes.jl vertex to an SVector{3,T}.
"""
@inline function _extract_vertex(::Type{T}, vert) where {T}
    coords = Meshes.to(vert)
    return SVector{3,T}(ustrip(coords[1]), ustrip(coords[2]), ustrip(coords[3]))
end

"""
    _get_triangle_vertices(::Type{T}, mesh::SimpleMesh, tri_idx::Int) where {T} -> (SVector{3,T}, SVector{3,T}, SVector{3,T})

Extract triangle vertices from mesh as SVectors. Accesses mesh data on-the-fly.
"""
@inline function _get_triangle_vertices(::Type{T}, mesh::SimpleMesh, tri_idx::Int) where {T}
    elem = mesh[tri_idx]
    verts = Meshes.vertices(elem)
    v1 = _extract_vertex(T, verts[1])
    v2 = _extract_vertex(T, verts[2])
    v3 = _extract_vertex(T, verts[3])
    return v1, v2, v3
end

"""
    _get_triangle_normal(::Type{T}, mesh::SimpleMesh, tri_idx::Int) where {T} -> SVector{3,T}

Extract and normalize triangle normal from mesh. Accesses mesh data on-the-fly.
"""
@inline function _get_triangle_normal(::Type{T}, mesh::SimpleMesh, tri_idx::Int) where {T}
    elem = mesh[tri_idx]
    return _normalize_normal(T, Meshes.normal(elem))
end

"""
    _compute_bbox(::Type{T}, mesh::SimpleMesh) where {T} -> (bbox_min, bbox_max)

Compute bounding box from mesh triangle data.

Performance: Single-pass min/max accumulation over all triangles.
"""
function _compute_bbox(::Type{T}, mesh::SimpleMesh) where {T}
    n = Meshes.nelements(mesh)

    # Initialize with first triangle's vertices
    v1, v2, v3 = _get_triangle_vertices(T, mesh, 1)
    min_x = min(v1[1], v2[1], v3[1])
    min_y = min(v1[2], v2[2], v3[2])
    min_z = min(v1[3], v2[3], v3[3])
    max_x = max(v1[1], v2[1], v3[1])
    max_y = max(v1[2], v2[2], v3[2])
    max_z = max(v1[3], v2[3], v3[3])

    # Single-pass accumulation over all triangles
    for i in 2:n
        v1, v2, v3 = _get_triangle_vertices(T, mesh, i)

        min_x = min(min_x, v1[1], v2[1], v3[1])
        min_y = min(min_y, v1[2], v2[2], v3[2])
        min_z = min(min_z, v1[3], v2[3], v3[3])
        max_x = max(max_x, v1[1], v2[1], v3[1])
        max_y = max(max_y, v1[2], v2[2], v3[2])
        max_z = max(max_z, v1[3], v2[3], v3[3])
    end

    # Handle degenerate cases (zero extent in any dimension)
    eps_val = max(eps(T) * 100, T(_DEGENERATE_EPS))
    if min_x == max_x
        min_x -= eps_val
        max_x += eps_val
    end
    if min_y == max_y
        min_y -= eps_val
        max_y += eps_val
    end
    if min_z == max_z
        min_z -= eps_val
        max_z += eps_val
    end

    return SVector{3,T}(min_x, min_y, min_z), SVector{3,T}(max_x, max_y, max_z)
end

"""
    _edge_key(v1::SVector{3,T}, v2::SVector{3,T}) where {T} -> Tuple

Create a canonical edge key from two vertices (order-independent).
Uses component-wise comparison for consistent ordering.
"""
@inline function _edge_key(v1::SVector{3,T}, v2::SVector{3,T}) where {T}
    return v1 < v2 ? (v1, v2) : (v2, v1)
end

"""
    has_consistent_normals(mesh::SimpleMesh) -> Bool

Check if triangle faces are consistently oriented (manifold orientation test).

This function verifies that all shared edges between adjacent triangles are
traversed in OPPOSITE directions, which is the geometric requirement for a
properly oriented manifold surface. This test is independent of surface curvature.

# Algorithm
For each shared edge between two triangles:
- Triangle A has edge v1→v2
- Triangle B (adjacent) must have edge v2→v1 (opposite direction)
- If both traverse in the same direction → faces are incorrectly oriented

# Returns
- `true` if all triangles are correctly oriented (manifold surface)
- `false` if any triangles have flipped faces (orientation errors)

# Performance
O(n) complexity using edge hash map instead of O(n²) pairwise comparison.

# Examples
```julia
octree = TriangleOctree("model.stl"; h_min=0.01)
if !has_consistent_normals(octree.mesh)
    @warn "Mesh has flipped triangles - will cause incorrect isinside() results"
end
```

# Note
This test uses GEOMETRIC edge orientation, not algebraic normal dot products.
It will correctly validate meshes with high curvature (sharp edges, creases).
"""
function has_consistent_normals(::Type{T}, mesh::SimpleMesh) where {T}
    n = Meshes.nelements(mesh)
    n <= 1 && return true

    # Build edge → (triangle index, edge vertices) map in O(n)
    # When we encounter an edge again, check if it's traversed in opposite direction
    edge_map = Dict{
        Tuple{SVector{3,T},SVector{3,T}},
        Tuple{Int,SVector{3,T},SVector{3,T}},
    }()
    sizehint!(edge_map, 3 * n)  # Each triangle has 3 edges

    for i in 1:n
        v1, v2, v3 = _get_triangle_vertices(T, mesh, i)

        # Process each edge of triangle i
        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            key = _edge_key(va, vb)
            existing = get(edge_map, key, nothing)

            if existing !== nothing
                # Edge shared with another triangle - check orientation
                (_, other_va, other_vb) = existing

                # For correct manifold orientation, the edge should be traversed
                # in OPPOSITE directions: if this triangle goes va→vb,
                # the other should go vb→va
                same_direction = (va ≈ other_va && vb ≈ other_vb)

                if same_direction
                    # Both triangles traverse edge in SAME direction → FLIPPED FACE!
                    return false
                end
                # If opposite direction → correct orientation, continue
            else
                # First time seeing this edge - store triangle index and edge vertices
                edge_map[key] = (i, va, vb)
            end
        end
    end

    return true
end

has_consistent_normals(mesh::SimpleMesh) = has_consistent_normals(Float64, mesh)

#=============================================================================
CONSTRUCTION (BUILD-TIME ONLY)

Signed-distance classification is used for empty leaf labeling.
Classification is performed on the final balanced tree for correctness.
=============================================================================#

"""
    _create_root_octree(::Type{T}, mesh::SimpleMesh, n_triangles::Int) where {T} -> SpatialOctree{Int,T}

Create root octree covering mesh bounding box with all triangles in root.
"""
function _create_root_octree(::Type{T}, mesh::SimpleMesh, n_triangles::Int) where {T}
    bbox_min, bbox_max = _compute_bbox(T, mesh)

    # Expand bounding box by 2% to avoid numerical precision issues
    # and ensure geometry is fully contained with buffer zone
    bbox_sz = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) * T(0.5)
    expansion_factor = T(_BBOX_EXPANSION)
    expanded_half_size = (bbox_sz * T(0.5)) * expansion_factor
    bbox_min = bbox_center - expanded_half_size
    bbox_max = bbox_center + expanded_half_size

    root_size = maximum(bbox_max - bbox_min)

    estimated_boxes = max(1000, n_triangles * 2)
    tree = SpatialOctree{Int,T}(bbox_min, root_size; initial_capacity=estimated_boxes)

    root_elements = tree.element_lists[1]
    for tri_idx in 1:n_triangles
        push!(root_elements, tri_idx)
    end

    return tree
end

"""
    TriangleOctree(mesh::SimpleMesh{M,C};
                   h_min,
                   max_triangles_per_box::Int=50,
                   classify_leaves::Bool=true,
                   verify_orientation::Bool=true) -> TriangleOctree{M,C,Float64}

Build an octree spatial index for a SimpleMesh.

# Arguments
- `mesh::SimpleMesh{M,C}`: Meshes.jl SimpleMesh to index
- `h_min`: Minimum octree box size (stopping criterion)
- `max_triangles_per_box::Int=50`: Maximum triangles per leaf before subdivision
- `classify_leaves::Bool=true`: Whether to classify empty leaves as interior/exterior
- `verify_orientation::Bool=true`: Check if triangle normals are consistently oriented

# Algorithm
1. (Optional) Verify triangle normal consistency
2. Create root octree covering mesh bounding box
3. Distribute triangles to boxes using SAT intersection tests
4. Recursively subdivide boxes exceeding triangle threshold
5. Balance octree to ensure 2:1 refinement constraint
6. (Optional) Classify empty leaves via signed-distance tests

# Returns
`TriangleOctree{M,C,T}` ready for fast spatial queries (T defaults to Float64)

# Example
```julia
mesh = GeoIO.load("box.stl").geometry
octree = TriangleOctree(mesh; h_min=0.01, max_triangles_per_box=50)
println("Built octree with ", num_leaves(octree), " leaves")
```
"""
function TriangleOctree(
    mesh::SimpleMesh{M,C};
    h_min,
    max_triangles_per_box::Int=50,
    classify_leaves::Bool=true,
    verify_orientation::Bool=true,
) where {M<:Manifold,C<:CRS}
    T = Float64
    h_min_val = T(ustrip(h_min))
    h_min_val > 0 || throw(ArgumentError("h_min must be positive, got $h_min_val"))
    n_triangles = Meshes.nelements(mesh)

    if verify_orientation && !has_consistent_normals(T, mesh)
        throw(
            ArgumentError(
                "Triangle mesh has orientation errors (flipped faces). " *
                "Some triangles have their faces oriented incorrectly (shared edges " *
                "traversed in the same direction instead of opposite directions). " *
                "This will cause incorrect isinside() and signed distance calculations. " *
                "The mesh needs to be repaired before use with the octree. " *
                "To skip this check, pass `verify_orientation=false`.",
            )
        )
    end

    tree = _create_root_octree(T, mesh, n_triangles)

    size_criterion = SizeCriterion(h_min_val)
    criterion = AndCriterion((MaxElementsCriterion(max_triangles_per_box), size_criterion))

    _subdivide_triangle_octree!(tree, mesh, 1, criterion)

    # Balance octree to ensure 2:1 refinement constraint
    # This may subdivide leaves and create new empty leaves.
    balance_octree!(tree, size_criterion)

    # Classify FINAL leaves after balancing for correctness.
    # (More expensive than propagation, but robust.)
    classification = if classify_leaves
        _classify_leaves(tree, mesh)
    else
        nothing
    end

    return TriangleOctree(tree, mesh, classification)
end

"""
    TriangleOctree(filepath::String; h_min, kwargs...) -> TriangleOctree

Build an octree spatial index from an STL file.

# Arguments
- `filepath::String`: Path to STL file
- `h_min`: Minimum octree box size
- `kwargs...`: Additional arguments passed to TriangleOctree(mesh; ...)

# Example
```julia
octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)
```
"""
function TriangleOctree(filepath::String; h_min, kwargs...)
    geo = GeoIO.load(filepath)
    return TriangleOctree(geo.geometry; h_min, kwargs...)
end

"""
    _subdivide_triangle_octree!(tree, mesh, box_idx, criterion)

Recursively subdivide a box in the triangle octree.

For each box that needs subdivision:
1. Create 8 child boxes
2. For each triangle in parent, find which children it intersects
3. Distribute triangle indices to children
4. Recursively subdivide children if they meet criteria
"""
function _subdivide_triangle_octree!(
    tree::SpatialOctree{Int,T},
    mesh::SimpleMesh,
    box_idx::Int,
    criterion::SubdivisionCriterion,
) where {T<:Real}
    if !should_subdivide(criterion, tree, box_idx)
        return
    end

    parent_triangles = tree.element_lists[box_idx]
    isempty(parent_triangles) && return

    subdivide!(tree, box_idx)

    children = tree.children[box_idx]

    for tri_idx in parent_triangles
        v1, v2, v3 = _get_triangle_vertices(T, mesh, tri_idx)

        for child_idx in children
            child_idx == 0 && continue

            child_min, child_max = box_bounds(tree, child_idx)

            if triangle_box_intersection(v1, v2, v3, child_min, child_max)
                push!(tree.element_lists[child_idx], tri_idx)
            end
        end
    end

    for child_idx in children
        child_idx == 0 && continue
        _subdivide_triangle_octree!(tree, mesh, child_idx, criterion)
    end
    return
end

@inline function _point_box_distance_sq(
    point::SVector{3,T},
    bbox_min::SVector{3,T},
    bbox_max::SVector{3,T},
) where {T<:Real}
    dx = point[1] < bbox_min[1] ? (bbox_min[1] - point[1]) :
         (point[1] > bbox_max[1] ? (point[1] - bbox_max[1]) : zero(T))
    dy = point[2] < bbox_min[2] ? (bbox_min[2] - point[2]) :
         (point[2] > bbox_max[2] ? (point[2] - bbox_max[2]) : zero(T))
    dz = point[3] < bbox_min[3] ? (bbox_min[3] - point[3]) :
         (point[3] > bbox_max[3] ? (point[3] - bbox_max[3]) : zero(T))
    return dx * dx + dy * dy + dz * dz
end

@inline function _leaf_class_from_signed_distance(sd, tol)
    if abs(sd) <= tol
        return LEAF_BOUNDARY
    end
    return sd < 0 ? LEAF_INTERIOR : LEAF_EXTERIOR
end

@inline function _update_closest_triangle!(
    point::SVector{3,T},
    mesh::SimpleMesh,
    tri_idx::Int,
    state::NearestTriangleState{T},
) where {T<:Real}
    v1, v2, v3 = _get_triangle_vertices(T, mesh, tri_idx)
    cp = closest_point_on_triangle(point, v1, v2, v3)
    dvec = point - cp
    d2 = dot(dvec, dvec)

    if d2 < state.best_dist_sq
        state.best_dist_sq = d2
        state.closest_idx = tri_idx
        state.closest_pt = cp
    end

    return nothing
end

function _nearest_triangle_octree!(
    point::SVector{3,T},
    tree::SpatialOctree{Int,T},
    mesh::SimpleMesh,
    box_idx::Int,
    state::NearestTriangleState{T},
) where {T<:Real}
    bbox_min, bbox_max = box_bounds(tree, box_idx)
    _point_box_distance_sq(point, bbox_min, bbox_max) > state.best_dist_sq && return

    if is_leaf(tree, box_idx)
        @inbounds for tri_idx in tree.element_lists[box_idx]
            _update_closest_triangle!(point, mesh, tri_idx, state)
        end
        return
    end

    # Collect children with distances into stack-allocated buffer (max 8 children).
    children = tree.children[box_idx]
    dists = MVector{8,T}(ntuple(_ -> typemax(T), Val(8)))
    idxs = MVector{8,Int}(ntuple(_ -> 0, Val(8)))
    n_valid = 0
    @inbounds for child_idx in children
        child_idx == 0 && continue
        cmin, cmax = box_bounds(tree, child_idx)
        d2 = _point_box_distance_sq(point, cmin, cmax)
        if d2 <= state.best_dist_sq
            n_valid += 1
            # Insertion sort into sorted position
            pos = n_valid
            while pos > 1 && d2 < dists[pos-1]
                dists[pos] = dists[pos-1]
                idxs[pos] = idxs[pos-1]
                pos -= 1
            end
            dists[pos] = d2
            idxs[pos] = child_idx
        end
    end

    return @inbounds for i in 1:n_valid
        _nearest_triangle_octree!(point, tree, mesh, idxs[i], state)
    end
end


"""
    _compute_signed_distance_octree(point, mesh, tree) -> T

Compute signed distance to the closest triangle using octree branch-and-bound.

This avoids global O(M) triangle scans by searching only relevant octree regions.
"""
function _compute_signed_distance_octree(
    point::SVector{3,T},
    mesh::SimpleMesh,
    tree::SpatialOctree{Int,T},
) where {T<:Real}
    state = NearestTriangleState{T}(point)

    _nearest_triangle_octree!(point, tree, mesh, 1, state)

    if state.closest_idx == 0
        return typemax(T)
    end

    n = _get_triangle_normal(T, mesh, state.closest_idx)
    sign = dot(point - state.closest_pt, n) < 0 ? -one(T) : one(T)
    return sign * sqrt(state.best_dist_sq)
end

"""
    _classify_empty_leaf_conservative(tree, mesh, leaf_idx) -> Int8

Conservative classification for empty leaves.

- `2` (interior): center and all 8 corners classify inside
- `0` (exterior): center and all 8 corners classify outside
- `1` (boundary): mixed results

This avoids falsely labeling a partially-outside leaf as interior.

Implementation uses octree-accelerated signed distance, not ray casting.
"""
function _classify_empty_leaf_conservative(
    tree::SpatialOctree{Int,T},
    mesh::SimpleMesh,
    leaf_idx::Int,
) where {T<:Real}
    bbox_min, bbox_max = box_bounds(tree, leaf_idx)
    h = box_size(tree, leaf_idx)
    δ = max(T(_CLASSIFICATION_INSET), h * T(_CLASSIFY_TOLERANCE_REL))
    tol = max(T(_CLASSIFY_TOLERANCE_ABS), h * T(_CLASSIFY_TOLERANCE_REL))

    center = box_center(tree, leaf_idx)
    sd_center = _compute_signed_distance_octree(center, mesh, tree)

    # Early exit: if center is far from surface, use Lipschitz bound
    half_diag = (sqrt(T(3)) * h) / T(2)
    if abs(sd_center) > half_diag + tol
        return _leaf_class_from_signed_distance(sd_center, tol)
    end

    # Classify center (reuse already-computed signed distance)
    center_class = _leaf_class_from_signed_distance(sd_center, tol)

    # Early exit: if center is on boundary, entire box is boundary
    center_class == LEAF_BOUNDARY && return LEAF_BOUNDARY

    # Inset corners slightly to avoid exact-on-surface degeneracy
    x0, x1 = bbox_min[1] + δ, bbox_max[1] - δ
    y0, y1 = bbox_min[2] + δ, bbox_max[2] - δ
    z0, z1 = bbox_min[3] + δ, bbox_max[3] - δ

    # Check 8 corners for unanimous agreement with center classification
    corners = (
        SVector{3,T}(x0, y0, z0),
        SVector{3,T}(x1, y0, z0),
        SVector{3,T}(x0, y1, z0),
        SVector{3,T}(x1, y1, z0),
        SVector{3,T}(x0, y0, z1),
        SVector{3,T}(x1, y0, z1),
        SVector{3,T}(x0, y1, z1),
        SVector{3,T}(x1, y1, z1),
    )

    @inbounds for corner in corners
        corner_class = _leaf_class_from_signed_distance(
            _compute_signed_distance_octree(corner, mesh, tree),
            tol,
        )
        # Any disagreement with center → mixed classification → boundary
        corner_class != center_class && return LEAF_BOUNDARY
    end

    # All 9 points (center + 8 corners) agree on classification
    return center_class
end

"""
    _classify_leaves(tree::SpatialOctree{Int,T}, mesh::SimpleMesh) -> Vector{Int8}

Classify octree leaves as exterior (0), boundary (1), or interior (2).

Uses octree-accelerated signed distance for robust classification.

# Returns
- `classification::Vector{Int8}`: Classification for each box
    - `-1`: Internal node (not a leaf, unclassified)
    - `0`: Exterior leaf
    - `1`: Boundary leaf
    - `2`: Interior leaf
"""
function _classify_leaves(tree::SpatialOctree{Int,T}, mesh::SimpleMesh) where {T<:Real}
    n_boxes = tree.num_boxes[]
    classification = fill(LEAF_UNKNOWN, n_boxes)
    leaves = all_leaves(tree)

    # Classify each leaf in parallel (each classification is independent read-only work)
    results = tmap(leaves) do leaf_idx
        if !isempty(tree.element_lists[leaf_idx])
            LEAF_BOUNDARY
        else
            _classify_empty_leaf_conservative(tree, mesh, leaf_idx)
        end
    end

    for (leaf_idx, cls) in zip(leaves, results)
        classification[leaf_idx] = cls
    end

    return classification
end


# length() supports Julia iteration protocol; num_triangles() is the descriptive API
Base.length(octree::TriangleOctree) = Meshes.nelements(octree.mesh)
num_leaves(octree::TriangleOctree) = length(all_leaves(octree.tree))
num_triangles(octree::TriangleOctree) = Meshes.nelements(octree.mesh)

#=============================================================================
QUERIES (RUN-TIME, PERFORMANCE CRITICAL)

isinside() is called thousands of times during discretization.
Common path uses cached classification or local signed distance.

Performance:
- Exterior/interior leaves: O(1) - single array lookup
- Boundary leaves: O(k) where k ≈ 10-50 triangles in leaf
=============================================================================#

"""
    isinside(point::SVector{3,T}, octree::TriangleOctree) -> Bool

Fast interior/exterior test using octree spatial index.

Returns `true` if point is inside the closed surface defined by the mesh.

# Performance
- Complexity: O(log M + k) where M = number of triangles, k ≈ 10-50
- Speedup: 100-1000× faster than brute-force O(M) approach

# Example
```julia
using WhatsThePoint, StaticArrays

octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)

point = SVector(0.5, 0.5, 0.5)
is_inside = isinside(point, octree)
```
"""
function isinside(point::SVector{3,T}, octree::TriangleOctree) where {T<:Real}
    # Fast rejection: if point is outside bounding box, it's definitely outside
    bbox_min, bbox_max = box_bounds(octree.tree, 1)
    if any(point .< bbox_min) || any(point .> bbox_max)
        return false
    end

    # Without leaf classification, fall back to octree-accelerated signed distance
    if isnothing(octree.leaf_classification)
        return _compute_signed_distance_octree(point, octree.mesh, octree.tree) < 0
    end

    leaf_idx = find_leaf(octree.tree, point)
    classification = octree.leaf_classification[leaf_idx]

    # Use leaf classification for clear cases
    classification == LEAF_EXTERIOR && return false
    classification == LEAF_INTERIOR && return true

    # Empty boundary leaf (conservative mixed classification):
    # use signed-distance fallback for this query point.
    return _compute_signed_distance_octree(point, octree.mesh, octree.tree) < 0
end

"""
    isinside(points::Vector{SVector{3,T}}, octree::TriangleOctree) -> Vector{Bool}

Batch interior test for multiple points.

# Example
```julia
test_points = [SVector(randn(3)...) for _ in 1:1000]
results = isinside(test_points, octree)  # Returns Vector{Bool}
```
"""
function isinside(points::Vector{SVector{3,T}}, octree::TriangleOctree) where {T<:Real}
    return tmap(p -> isinside(p, octree), points)
end

"""
    isinside(point::Point{𝔼{3}}, octree::TriangleOctree{𝔼{3},C,T}) -> Bool

Bridge from Meshes.jl Point to SVector-based isinside.

Extracts coordinates from Point, converts to SVector{3,T}, and delegates
to the SVector method.
"""
function isinside(point::Point{𝔼{3}}, octree::TriangleOctree{𝔼{3},C,T}) where {C,T}
    sv = _extract_vertex(T, point)
    return isinside(sv, octree)
end

"""
    isinside(points::AbstractVector{<:Point{𝔼{3}}}, octree::TriangleOctree{𝔼{3}}) -> Vector{Bool}

Batch bridge from Meshes.jl Points to SVector-based isinside.
"""
function isinside(
    points::AbstractVector{<:Point{𝔼{3}}},
    octree::TriangleOctree{𝔼{3}},
)
    return tmap(p -> isinside(p, octree), points)
end
