# Layer 1: Spatial Tree (dimension-generic)
#
# `SpatialTree{N,E,T}` is an adaptive 2^N-tree over N-dimensional Cartesian space
# (N=3 octree, N=2 quadtree). Integer (cell, level) coordinates give O(1)
# neighbour arithmetic and 2:1 balance enforcement. It is element- and CRS-blind:
# the whole structure lives in stripped `SVector{N,T}` numeric space.
#
# `const SpatialOctree{E,T} = SpatialTree{3,E,T}` keeps the 3D call sites (and the
# public name) unchanged.

"""
    SubdivisionCriterion

Abstract type for tree subdivision decision logic. Subtypes implement
`should_subdivide(criterion, tree, box_idx)` (subdivide if content/size
demands it) and `can_subdivide(criterion, tree, box_idx)` (physical limits
only — used during balancing).
"""
abstract type SubdivisionCriterion end

#=============================================================================
Core Data Structure
=============================================================================#

"""
    SpatialTree{N,E,T}

Adaptive 2^N-tree over N-d Cartesian space (N=3 octree, N=2 quadtree). Uses a
`(cell, level)` integer coordinate system:
- `cell` are integer coordinates at refinement `level`
- box center = `origin + (2*cell + 1) * (root_size / level) / 2`
- O(1) neighbour calculation from coordinates

Children of a subdivided box are the contiguous block
`first_child .+ (0:2^N-1)` (`first_child == 0` marks a leaf).

INVARIANT: `subdivide!` always allocates the `2^N` children contiguously, and
children are never re-mutated (there is no coarsening). A future merge/coarsen
feature MUST preserve this or replace `first_child` with an explicit child list.

# Type Parameters
- `N`: spatial dimension
- `E`: element type stored per box (e.g. `Int` for triangle IDs)
- `T`: coordinate numeric type (`Float64`, `Float32`)
"""
struct SpatialTree{N, E, T <: Real}
    parent::Vector{Int}
    first_child::Vector{Int}          # 0 = leaf; children are first_child .+ (0:2^N-1)
    cell::Vector{SVector{N, Int}}     # integer coords at `level`
    level::Vector{Int}                # refinement level (root = 1, doubles per subdivide)
    origin::SVector{N, T}
    root_size::T
    element_lists::Vector{Vector{E}}
    num_boxes::Base.RefValue{Int}
end

"3D octree — the historical name and the default 3D specialization."
const SpatialOctree{E, T} = SpatialTree{3, E, T}

@inline n_children(::SpatialTree{N}) where {N} = 1 << N   # 2^N — single source of truth

#=============================================================================
Constructor
=============================================================================#

"""
    SpatialTree{N,E,T}(origin::SVector{N,T}, size::T; initial_capacity=1000)

Create an empty tree with a single root box (cubic, edge length `size`).
"""
function SpatialTree{N, E, T}(
        origin::SVector{N, T},
        size::T;
        initial_capacity = 1000,
    ) where {N, E, T}
    parent = zeros(Int, initial_capacity)
    first_child = zeros(Int, initial_capacity)
    cell = [zero(SVector{N, Int}) for _ in 1:initial_capacity]
    level = ones(Int, initial_capacity)               # root level = 1
    element_lists = [E[] for _ in 1:initial_capacity]

    num_boxes = Ref(1)
    return SpatialTree{N, E, T}(
        parent, first_child, cell, level, origin, size, element_lists, num_boxes
    )
end

#=============================================================================
Child access (replaces the old `children` field)
=============================================================================#

"Range of child indices of `box` (empty range if `box` is a leaf)."
@inline function children(t::SpatialTree, box::Int)
    fc = t.first_child[box]
    return fc == 0 ? (1:0) : (fc:(fc + n_children(t) - 1))
end

#=============================================================================
Box Geometry
=============================================================================#

"""
    box_center(tree, box_idx) -> SVector{N}

`center = origin + (2*cell + 1) * box_size / 2`, `box_size = root_size / level`.
"""
@inline function box_center(t::SpatialTree{N, E, T}, box::Int) where {N, E, T}
    c = t.cell[box]
    h = t.root_size / t.level[box]
    return t.origin + SVector{N, T}(ntuple(d -> (2 * c[d] + 1) * h / 2, N))
end

"""
    box_size(tree, box_idx) -> Real

Edge length of the box: `root_size / level`.
"""
@inline box_size(t::SpatialTree, box::Int) = t.root_size / t.level[box]

"""
    box_bounds(tree, box_idx) -> (min_corner, max_corner)
"""
@inline function box_bounds(t::SpatialTree{N, E, T}, box::Int) where {N, E, T}
    c = t.cell[box]
    h = t.root_size / t.level[box]
    lo = t.origin + SVector{N, T}(ntuple(d -> c[d] * h, N))
    return (lo, lo + SVector{N, T}(ntuple(_ -> h, N)))
end

#=============================================================================
Tree Structure Queries
=============================================================================#

"""
    is_leaf(tree, box_idx) -> Bool
"""
@inline is_leaf(t::SpatialTree, box::Int) = t.first_child[box] == 0

"""
    has_children(tree, box_idx) -> Bool
"""
@inline has_children(t::SpatialTree, box::Int) = !is_leaf(t, box)

"""
    num_elements(tree) -> Int
"""
function num_elements(t::SpatialTree)
    return sum(i -> length(t.element_lists[i]), 1:t.num_boxes[]; init = 0)
end

"""
    bounding_box(tree) -> (min_corner, max_corner)

Root box bounds.
"""
function bounding_box(t::SpatialTree{N, E, T}) where {N, E, T}
    max_corner = t.origin + SVector{N, T}(ntuple(_ -> t.root_size, N))
    return (t.origin, max_corner)
end

#=============================================================================
Tree Modification
=============================================================================#

"""
    add_box!(tree, cell::SVector{N,Int}, level, parent_idx) -> Int

Append a new leaf box at `(cell, level)`; grows arrays as needed. Returns index.
"""
function add_box!(t::SpatialTree{N, E, T}, cell::SVector{N, Int}, level::Int, parent::Int) where {N, E, T}
    t.num_boxes[] += 1
    box = t.num_boxes[]

    if box > length(t.parent)
        old = length(t.parent)
        newcap = 2 * old
        resize!(t.parent, newcap)
        resize!(t.first_child, newcap)
        @inbounds for i in (old + 1):newcap
            t.first_child[i] = 0
        end
        resize!(t.level, newcap)
        append!(t.cell, [zero(SVector{N, Int}) for _ in 1:(newcap - old)])
        append!(t.element_lists, [E[] for _ in 1:(newcap - old)])
    end

    t.parent[box] = parent
    t.first_child[box] = 0
    t.cell[box] = cell
    t.level[box] = level
    return box
end

"""
    subdivide!(tree, box_idx) -> UnitRange{Int}

Subdivide a leaf into `2^N` children (contiguous block). Returns the child range.
Child `m` (0-based) sits at `2*cell + bits(m)`, where bit `d-1` of `m` is the
offset along axis `d` — generalizing the standard octant ordering.
"""
function subdivide!(t::SpatialTree{N, E, T}, box::Int) where {N, E, T}
    @assert is_leaf(t, box) "Cannot subdivide non-leaf box"
    c = t.cell[box]
    child_level = 2 * t.level[box]
    first = t.num_boxes[] + 1
    for m in 0:(n_children(t) - 1)
        childcell = SVector{N, Int}(ntuple(d -> 2 * c[d] + ((m >> (d - 1)) & 1), N))
        add_box!(t, childcell, child_level, box)
    end
    t.first_child[box] = first
    return first:(first + n_children(t) - 1)
end

#=============================================================================
Point Queries
=============================================================================#

"""
    find_leaf(tree, point::SVector{N}) -> Int

Leaf box containing `point`, O(log n) from root.
"""
@inline function find_leaf(t::SpatialTree{N, E, T}, point::SVector{N, T}) where {N, E, T}
    current = 1
    while t.first_child[current] != 0
        c = box_center(t, current)
        offset = 0
        @inbounds for d in 1:N
            offset += (point[d] >= c[d] ? 1 : 0) << (d - 1)
        end
        current = t.first_child[current] + offset
    end
    return current
end

#=============================================================================
Neighbour Finding
=============================================================================#

"Axis-aligned neighbour offset for a direction code in `1:2N` (odd = negative face)."
@inline function _neighbor_offset(direction::Int, ::Val{N}) where {N}
    axis = (direction + 1) >> 1          # ceil(direction/2)
    s = isodd(direction) ? -1 : 1        # odd = negative face, even = positive
    return SVector{N, Int}(ntuple(d -> d == axis ? s : 0, N))
end

"""
    find_neighbor(tree, box_idx, direction) -> Vector{Int}

Neighbour(s) across the face `direction` (`1:2N`). Handles the 2:1 level
difference: returns `[same-level]`, or the finer boxes covering the face, or
`Int[]` at a boundary.
"""
function find_neighbor(t::SpatialTree{N}, box::Int, direction::Int) where {N}
    cell = t.cell[box]
    lvl = t.level[box]
    ncell = cell + _neighbor_offset(direction, Val(N))
    @inbounds for d in 1:N
        (ncell[d] < 0 || ncell[d] >= lvl) && return Int[]   # outside root
    end
    return find_boxes_at_coords(t, ncell, lvl)
end

"""
    find_boxes_at_coords(tree, target_cell::SVector{N,Int}, target_level) -> Vector{Int}

Box(es) covering `(target_cell, target_level)`: exact match, the coarser box
covering it, or `Int[]` if absent.
"""
function find_boxes_at_coords(t::SpatialTree{N}, target_cell::SVector{N, Int}, target_level::Int) where {N}
    current = 1
    while true
        cell = t.cell[current]
        lvl = t.level[current]

        if lvl == target_level
            return cell == target_cell ? [current] : Int[]
        elseif lvl < target_level
            is_leaf(t, current) && return [current]      # coarser leaf covers target
            scale = target_level ÷ lvl
            scaled = target_cell .÷ scale
            scaled == cell || return Int[]               # target not in this subtree
            offset = 0
            @inbounds for d in 1:N
                b = (target_cell[d] - cell[d] * scale) ÷ (scale >> 1)
                b > 1 && (b = 1)
                offset += b << (d - 1)
            end
            current = t.first_child[current] + offset
        else
            return [current]                             # current finer than target
        end
    end
    return
end

"3D convenience overload preserving the historical `(i,j,k,N)` signature."
find_boxes_at_coords(t::SpatialTree{3}, i::Int, j::Int, k::Int, N::Int) =
    find_boxes_at_coords(t, SVector{3, Int}(i, j, k), N)

#=============================================================================
Iteration Utilities
=============================================================================#

"""
    all_leaves(tree) -> Vector{Int}
"""
function all_leaves(t::SpatialTree)
    leaves = Int[]
    for box in 1:t.num_boxes[]
        is_leaf(t, box) && push!(leaves, box)
    end
    return leaves
end

"""
    all_boxes(tree) -> Vector{Int}
"""
all_boxes(t::SpatialTree) = collect(1:t.num_boxes[])

@inline function _boxes_overlap(a_min, a_max, b_min, b_max)
    @inbounds for d in eachindex(a_min)
        a_min[d] > b_max[d] && return false
        a_max[d] < b_min[d] && return false
    end
    return true
end

"""
    any_leaf_overlapping(tree, bbox_min, bbox_max, predicate) -> Bool

`true` if any leaf overlapping `[bbox_min, bbox_max]` satisfies `predicate`.
Prunes non-overlapping subtrees for O(log L) expected cost.
"""
function any_leaf_overlapping(t::SpatialTree, bbox_min, bbox_max, predicate)
    return _any_leaf_overlapping(t, 1, bbox_min, bbox_max, predicate)
end

function _any_leaf_overlapping(t::SpatialTree, box::Int, bmin, bmax, predicate)
    node_min, node_max = box_bounds(t, box)
    _boxes_overlap(bmin, bmax, node_min, node_max) || return false
    if is_leaf(t, box)
        return predicate(box)
    end
    for ch in children(t, box)
        _any_leaf_overlapping(t, ch, bmin, bmax, predicate) && return true
    end
    return false
end

#=============================================================================
Balancing (2:1 Constraint)
=============================================================================#

"""
    needs_balancing(tree, box_idx) -> Bool

`true` if subdividing this leaf would leave a 2-level jump with any face
neighbour (i.e. a neighbour that already has grandchildren).
"""
function needs_balancing(t::SpatialTree{N}, box::Int) where {N}
    is_leaf(t, box) || return false
    for direction in 1:(2N)
        for nb in find_neighbor(t, box, direction)
            if has_children(t, nb)
                for ch in children(t, nb)
                    has_children(t, ch) && return true
                end
            end
        end
    end
    return false
end

"""
    balance_octree!(tree, criterion::SubdivisionCriterion; redistribute! = nothing)

Enforce the 2:1 balance constraint across the tree (dimension-agnostic).
Uses `can_subdivide` (physical limits only), not `should_subdivide`.

`subdivide!` does not move a leaf's `element_lists` entries into the new
children — queries only scan leaves, so elements of a balance-subdivided box
would vanish from the queryable tree. Callers whose queries read element
lists (e.g. `TriangleOctree`'s nearest-triangle search) must pass
`redistribute!(tree, box_idx)`, invoked right after each forced subdivision
to push the parent's elements into the intersecting children.
"""
function balance_octree!(t::SpatialTree, criterion::SubdivisionCriterion; redistribute! = nothing)
    max_iterations = 100
    iteration = 0
    while iteration < max_iterations
        iteration += 1
        n = t.num_boxes[]
        subdivided_any = false
        for box in 1:n
            is_leaf(t, box) || continue
            if needs_balancing(t, box) && can_subdivide(criterion, t, box)
                subdivide!(t, box)
                subdivided_any = true
                redistribute! !== nothing && redistribute!(t, box)
            end
        end
        subdivided_any || break
    end
    if iteration >= max_iterations
        @warn "balance_octree! hit iteration limit - tree may not be fully balanced"
    end
    return nothing
end

#=============================================================================
LEAF CLASSIFICATION (GENERIC over a geometry query closure)
=============================================================================#

const LEAF_UNKNOWN::Int8 = -1
const LEAF_EXTERIOR::Int8 = 0
const LEAF_BOUNDARY::Int8 = 1
const LEAF_INTERIOR::Int8 = 2

const _CLASSIFICATION_INSET = 1.0e-9
const _CLASSIFY_TOLERANCE_REL = 1.0e-6
const _CLASSIFY_TOLERANCE_ABS = 1.0e-8

"""
    _leaf_class_from_signed_distance(sd, tol) -> Int8
"""
@inline function _leaf_class_from_signed_distance(sd, tol)
    if abs(sd) <= tol
        return LEAF_BOUNDARY
    end
    return sd < 0 ? LEAF_INTERIOR : LEAF_EXTERIOR
end

"""
    classify_leaves!(tree, geometry_query; tolerance_relative, tolerance_absolute) -> Vector{Int8}

Conservative `(2^N + 1)`-point classification (center + corners) per leaf, using
`geometry_query(point::SVector{N,T}, tol::T) -> Int8`.
"""
function classify_leaves!(
        t::SpatialTree{N, E, T},
        geometry_query::Function;
        tolerance_relative::Real = _CLASSIFY_TOLERANCE_REL,
        tolerance_absolute::Real = _CLASSIFY_TOLERANCE_ABS,
    ) where {N, E, T <: Real}
    n_boxes = t.num_boxes[]
    classification = fill(LEAF_UNKNOWN, n_boxes)
    leaves = all_leaves(t)

    classes = tmap(leaves) do leaf_idx
        _classify_leaf_conservative(
            t, leaf_idx, geometry_query, T(tolerance_relative), T(tolerance_absolute)
        )
    end
    for (i, leaf_idx) in enumerate(leaves)
        classification[leaf_idx] = classes[i]
    end
    return classification
end

"""
    _box_corners(lo, hi) -> NTuple{2^N, SVector{N}}

The `2^N` corners of a box, in bit order (bit `d-1` selects `hi[d]`).
"""
@inline function _box_corners(lo::SVector{N, T}, hi::SVector{N, T}) where {N, T}
    return ntuple(1 << N) do m
        mm = m - 1
        SVector{N, T}(ntuple(d -> ((mm >> (d - 1)) & 1) == 1 ? hi[d] : lo[d], N))
    end
end

"""
    _classify_leaf_conservative(tree, leaf_idx, geometry_query, tol_rel, tol_abs) -> Int8
"""
function _classify_leaf_conservative(
        t::SpatialTree{N, E, T},
        leaf_idx::Int,
        geometry_query::Function,
        tol_rel::T,
        tol_abs::T,
    ) where {N, E, T <: Real}
    bbox_min, bbox_max = box_bounds(t, leaf_idx)
    h = box_size(t, leaf_idx)
    tol = max(tol_abs, h * tol_rel)

    center = box_center(t, leaf_idx)
    corners = _box_corners(bbox_min, bbox_max)

    classes = map(p -> geometry_query(p, tol), (center, corners...))

    any(==(LEAF_BOUNDARY), classes) && return LEAF_BOUNDARY
    all(==(LEAF_INTERIOR), classes) && return LEAF_INTERIOR
    all(==(LEAF_EXTERIOR), classes) && return LEAF_EXTERIOR
    return LEAF_BOUNDARY
end
