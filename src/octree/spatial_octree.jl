# Layer 1: Generic Spatial Octree
#
# Concrete implementation of AbstractOctree using integer coordinate system
# for efficient neighbor finding and 2:1 balance enforcement.

using StaticArrays

#=============================================================================
Core Data Structure
=============================================================================#

"""
    SpatialOctree{T,C<:Real} <: AbstractOctree{T,C}

Concrete octree implementation using integer coordinate system for efficient neighbor finding.

Uses (i,j,k,N) coordinate system where:
- (i,j,k) are integer coordinates at refinement level N
- Box center = origin + (2*[i,j,k] + 1) * (root_size / N) / 2
- Enables O(1) neighbor calculation from coordinates

# Type Parameters
- `T`: Type of elements stored (e.g., Int for triangle IDs)
- `C`: Coordinate numeric type (e.g., Float64, Float32)

# Fields
- `parent::Vector{Int}`: Parent box index for each node (0 = root has no parent)
- `children::Matrix{Int}`: 8 children per node [node, child_idx] (0 = no child)
- `coords::Matrix{Int}`: (i,j,k,N) coordinates [node, :] where N is refinement level
- `origin::SVector{3,C}`: Spatial origin of root box
- `root_size::C`: Size of root box (assumed cubic)
- `element_lists::Vector{Vector{T}}`: Elements in each box
- `num_boxes::Ref{Int}`: Current number of boxes (mutable counter)

# Interface Implementation

Implements `AbstractSpatialTree` interface:
- `find_leaf(tree, point)` - O(log n) point location
- `bounding_box(tree)` - Root box bounds
- `num_elements(tree)` - Total stored elements
- `find_neighbor(tree, box, dir)` - 6-directional neighbor finding
- `balance_octree!(tree)` - 2:1 refinement constraint

# Example
```julia
using StaticArrays

origin = SVector(0.0, 0.0, 0.0)
octree = SpatialOctree{Int,Float64}(origin, 10.0)

# Subdivide root
subdivide!(octree, 1)

# Find leaf containing point
point = SVector(2.0, 2.0, 2.0)
leaf = find_leaf(octree, point)
```
"""
struct SpatialOctree{T, C <: Real} <: AbstractOctree{T, C}
    parent::Vector{Int}
    children::Vector{SVector{8, Int}}    # 8 child indices per box (0 = no child)
    coords::Vector{SVector{4, Int}}      # (i,j,k,N) coordinates per box
    origin::SVector{3, C}
    root_size::C
    element_lists::Vector{Vector{T}}
    num_boxes::Ref{Int}          # Mutable counter
end

#=============================================================================
Constructor
=============================================================================#

"""
    SpatialOctree{T,C}(origin::SVector{3,C}, size::C; initial_capacity=1000)

Create empty octree with root node.

# Arguments
- `origin`: Minimum corner of root box
- `size`: Edge length of root box (assumed cubic)
- `initial_capacity`: Initial array capacity (will grow as needed)

# Example
```julia
origin = SVector(0.0, 0.0, 0.0)
octree = SpatialOctree{Int,Float64}(origin, 10.0)
```
"""
function SpatialOctree{T, C}(
        origin::SVector{3, C},
        size::C;
        initial_capacity = 1000,
    ) where {T, C}
    parent = zeros(Int, initial_capacity)
    children = [SVector{8, Int}(zeros(Int, 8)) for _ in 1:initial_capacity]
    coords = [SVector{4, Int}(zeros(Int, 4)) for _ in 1:initial_capacity]
    element_lists = [T[] for _ in 1:initial_capacity]

    # Initialize root at (0,0,0,1)
    coords[1] = SVector{4, Int}(0, 0, 0, 1)
    num_boxes = Ref(1)

    return SpatialOctree{T, C}(parent, children, coords, origin, size, element_lists, num_boxes)
end

#=============================================================================
Box Geometry
=============================================================================#

"""
    box_center(octree::SpatialOctree, box_idx::Int) -> SVector{3}

Compute spatial center of box using (i,j,k,N) coordinates.

# Formula
```
center = origin + (2*[i,j,k] + 1) * box_size / 2
```
where `box_size = root_size / N`
"""
function box_center(octree::SpatialOctree{T, C}, box_idx::Int) where {T, C}
    i, j, k, N = octree.coords[box_idx]
    h = octree.root_size / N  # Box size at this level

    x = octree.origin[1] + (2 * i + 1) * h / 2
    y = octree.origin[2] + (2 * j + 1) * h / 2
    z = octree.origin[3] + (2 * k + 1) * h / 2

    return SVector{3, C}(x, y, z)
end

"""
    box_size(octree::SpatialOctree, box_idx::Int) -> Real

Get edge length of box.

Box size at refinement level N is: `root_size / N`
"""
function box_size(octree::SpatialOctree, box_idx::Int)
    N = octree.coords[box_idx][4]
    return octree.root_size / N
end

"""
    box_bounds(octree::SpatialOctree, box_idx::Int) -> (SVector{3}, SVector{3})

Get (min_corner, max_corner) of box.

# Returns
Tuple of (min_corner, max_corner) as SVector{3,C}
"""
function box_bounds(octree::SpatialOctree{T, C}, box_idx::Int) where {T, C}
    i, j, k, N = octree.coords[box_idx]
    h = octree.root_size / N

    min_corner = octree.origin + SVector{3, C}(i * h, j * h, k * h)
    max_corner = min_corner + SVector{3, C}(h, h, h)

    return (min_corner, max_corner)
end

#=============================================================================
Tree Structure Queries
=============================================================================#

"""
    is_leaf(octree::SpatialOctree, box_idx::Int) -> Bool

Check if box has no children.
"""
function is_leaf(octree::SpatialOctree, box_idx::Int)
    return octree.children[box_idx][1] == 0
end

"""
    has_children(octree::SpatialOctree, box_idx::Int) -> Bool

Check if box has been subdivided.
"""
function has_children(octree::SpatialOctree, box_idx::Int)
    return !is_leaf(octree, box_idx)
end

"""
    num_elements(octree::SpatialOctree) -> Int

Total number of elements stored in tree.
"""
function num_elements(octree::SpatialOctree)
    total = 0
    for box_idx in 1:octree.num_boxes[]
        total += length(octree.element_lists[box_idx])
    end
    return total
end

"""
    bounding_box(octree::SpatialOctree) -> (SVector{3}, SVector{3})

Get overall bounding box of tree (root box).
"""
function bounding_box(octree::SpatialOctree{T, C}) where {T, C}
    max_corner =
        octree.origin + SVector{3, C}(octree.root_size, octree.root_size, octree.root_size)
    return (octree.origin, max_corner)
end

#=============================================================================
Tree Modification
=============================================================================#

"""
    add_box!(octree::SpatialOctree, i::Int, j::Int, k::Int, N::Int, parent_idx::Int) -> Int

Add new box to octree. Returns box index.

Automatically grows arrays if capacity exceeded.

# Arguments
- `i, j, k`: Integer coordinates at level N
- `N`: Refinement level (N=1 is root, N=2 is first subdivision, etc.)
- `parent_idx`: Index of parent box

# Returns
Index of newly created box
"""
function add_box!(
        octree::SpatialOctree{T},
        i::Int,
        j::Int,
        k::Int,
        N::Int,
        parent_idx::Int,
    ) where {T}
    octree.num_boxes[] += 1
    box_idx = octree.num_boxes[]

    # Grow arrays if needed
    # Invariant: all arrays have same length
    if box_idx > length(octree.parent)
        old_capacity = length(octree.parent)
        new_capacity = 2 * old_capacity

        resize!(octree.parent, new_capacity)
        append!(
            octree.children,
            [SVector{8, Int}(zeros(Int, 8)) for _ in 1:(new_capacity - old_capacity)],
        )
        append!(
            octree.coords,
            [SVector{4, Int}(zeros(Int, 4)) for _ in 1:(new_capacity - old_capacity)],
        )
        append!(octree.element_lists, [T[] for _ in 1:(new_capacity - old_capacity)])
    end

    octree.parent[box_idx] = parent_idx
    octree.coords[box_idx] = SVector{4, Int}(i, j, k, N)

    return box_idx
end

"""
    subdivide!(octree::SpatialOctree, box_idx::Int) -> Vector{Int}

Subdivide box into 8 children. Returns child indices [1:8].

# Child Ordering (Standard Octree Convention)
1: (0,0,0) - bottom-left-front   (x-, y-, z-)
2: (1,0,0) - bottom-right-front  (x+, y-, z-)
3: (0,1,0) - top-left-front      (x-, y+, z-)
4: (1,1,0) - top-right-front     (x+, y+, z-)
5: (0,0,1) - bottom-left-back    (x-, y-, z+)
6: (1,0,1) - bottom-right-back   (x+, y-, z+)
7: (0,1,1) - top-left-back       (x-, y+, z+)
8: (1,1,1) - top-right-back      (x+, y+, z+)

# Arguments
- `box_idx`: Index of box to subdivide (must be leaf)

# Returns
Vector of 8 child indices in standard order
"""
function subdivide!(octree::SpatialOctree, box_idx::Int)
    @assert is_leaf(octree, box_idx) "Cannot subdivide non-leaf box"

    i_p, j_p, k_p, N_p = octree.coords[box_idx]
    N_child = 2 * N_p  # Double refinement level

    child_indices = Int[]

    # Standard octree ordering: iterate through (di, dj, dk) ∈ {0,1}³
    for dk in 0:1, dj in 0:1, di in 0:1
        i_c = 2 * i_p + di
        j_c = 2 * j_p + dj
        k_c = 2 * k_p + dk

        child_idx = add_box!(octree, i_c, j_c, k_c, N_child, box_idx)
        push!(child_indices, child_idx)
    end

    # Update parent's children array
    octree.children[box_idx] = SVector{8, Int}(child_indices...)

    return child_indices
end

#=============================================================================
Point Queries
=============================================================================#

"""
    find_leaf(octree::SpatialOctree, point::SVector{3}) -> Int

Find leaf box containing point. Returns box index.

Traverses tree from root to leaf in O(log n) time.

# Arguments
- `point`: Query point in same coordinate system as octree

# Returns
Index of leaf box containing point

# Throws
AssertionError if point is outside octree bounds
"""
function find_leaf(octree::SpatialOctree{T, C}, point::SVector{3, C}) where {T, C}
    current = 1  # Start at root

    while has_children(octree, current)
        # Determine which octant contains point
        center = box_center(octree, current)

        # Binary decision for each dimension
        di = point[1] >= center[1] ? 1 : 0
        dj = point[2] >= center[2] ? 1 : 0
        dk = point[3] >= center[3] ? 1 : 0

        # Child index (1-8) based on octant
        # Ordering: di + 2*dj + 4*dk + 1
        child_offset = di + 2 * dj + 4 * dk + 1

        next = octree.children[current][child_offset]
        @assert next != 0 "Point outside octree bounds or tree structure corrupted"

        current = next
    end

    return current
end

#=============================================================================
Neighbor Finding
=============================================================================#

"""
    neighbor_direction(direction::Int) -> (Int, Int, Int)

Convert direction code to (di, dj, dk) offset.

# Direction Codes
- 1: -x (left)
- 2: +x (right)
- 3: -y (bottom)
- 4: +y (top)
- 5: -z (front)
- 6: +z (back)
"""
function neighbor_direction(direction::Int)
    @assert 1 <= direction <= 6 "Direction must be in 1:6"

    directions = (
        (-1, 0, 0),  # 1: -x
        (1, 0, 0),  # 2: +x
        (0, -1, 0),  # 3: -y
        (0, 1, 0),  # 4: +y
        (0, 0, -1),  # 5: -z
        (0, 0, 1),  # 6: +z
    )

    return directions[direction]
end

"""
    find_neighbor(octree::SpatialOctree, box_idx::Int, direction::Int) -> Vector{Int}

Find neighbor(s) in given direction. Returns vector of neighbor indices.

Handles 2:1 refinement level difference:
- If neighbor exists at same level: returns [neighbor_idx]
- If neighbor is subdivided (finer): returns children on shared face (up to 4)
- If neighbor doesn't exist (boundary): returns empty vector

# Arguments
- `box_idx`: Box to find neighbor of
- `direction`: Direction code (1-6, see `neighbor_direction`)

# Returns
Vector of neighbor box indices (may be empty if at boundary)
"""
function find_neighbor(octree::SpatialOctree, box_idx::Int, direction::Int)
    i, j, k, N = octree.coords[box_idx]
    di, dj, dk = neighbor_direction(direction)

    # Target neighbor coordinates
    i_n = i + di
    j_n = j + dj
    k_n = k + dk

    # Check if outside domain (root box spans 0:N-1 in each dimension)
    if i_n < 0 || i_n >= N || j_n < 0 || j_n >= N || k_n < 0 || k_n >= N
        return Int[]
    end

    # Search for box at these coordinates
    # May return single box or multiple if neighbor is subdivided
    return find_boxes_at_coords(octree, i_n, j_n, k_n, N)
end

"""
    find_boxes_at_coords(octree::SpatialOctree, i_target::Int, j_target::Int, k_target::Int, N_target::Int) -> Vector{Int}

Find box(es) at given (i,j,k,N) coordinates.

- If exact match found at level N_target, returns [box_idx]
- If location is covered by coarser box, returns [coarser_box_idx]
- If location is subdivided finer, returns all descendants at that location

# Returns
Vector of box indices covering the target location
"""
function find_boxes_at_coords(
        octree::SpatialOctree,
        i_target::Int,
        j_target::Int,
        k_target::Int,
        N_target::Int,
    )
    # Start at root
    current = 1

    while true
        i, j, k, N = octree.coords[current]

        if N == N_target
            # Check if this is the target
            if i == i_target && j == j_target && k == k_target
                return [current]
            else
                return Int[]  # Not found
            end
        elseif N < N_target
            # Need to go deeper
            if !has_children(octree, current)
                # This leaf covers the target region (coarser resolution)
                return [current]
            end

            # Determine which child contains target
            # Scale target coords to current level
            scale_factor = N_target ÷ N
            i_scaled = i_target ÷ scale_factor
            j_scaled = j_target ÷ scale_factor
            k_scaled = k_target ÷ scale_factor

            # Check if scaled coords match current box
            if i_scaled != i || j_scaled != j || k_scaled != k
                return Int[]  # Target not in this subtree
            end

            # Find which child to descend into
            di = (i_target - i * scale_factor) ÷ (scale_factor ÷ 2)
            dj = (j_target - j * scale_factor) ÷ (scale_factor ÷ 2)
            dk = (k_target - k * scale_factor) ÷ (scale_factor ÷ 2)

            # Clamp to {0,1}
            di = min(di, 1)
            dj = min(dj, 1)
            dk = min(dk, 1)

            # Child index based on (di, dj, dk)
            child_offset = di + 2 * dj + 4 * dk + 1
            current = octree.children[current][child_offset]

            if current == 0
                return Int[]  # Path doesn't exist
            end
        else
            # N > N_target: current box is finer than target
            # This case shouldn't happen in normal traversal from root
            # Return current box as approximation
            return [current]
        end
    end
    return
end

#=============================================================================
Iteration Utilities
=============================================================================#

"""
    all_leaves(octree::SpatialOctree) -> Vector{Int}

Return indices of all leaf boxes.
"""
function all_leaves(octree::SpatialOctree)
    leaves = Int[]
    for box_idx in 1:octree.num_boxes[]
        if is_leaf(octree, box_idx)
            push!(leaves, box_idx)
        end
    end
    return leaves
end

"""
    all_boxes(octree::SpatialOctree) -> Vector{Int}

Return indices of all boxes (leaves and internal nodes).
"""
function all_boxes(octree::SpatialOctree)
    return collect(1:octree.num_boxes[])
end

#=============================================================================
Balancing (2:1 Constraint)
=============================================================================#

"""
    needs_balancing(octree::SpatialOctree, box_idx::Int) -> Bool

Check if subdividing this box would violate 2:1 balance with any neighbor.

Returns true if any neighbor has grandchildren (2-level refinement difference).
"""
function needs_balancing(octree::SpatialOctree, box_idx::Int)
    if !is_leaf(octree, box_idx)
        return false  # Only check leaves
    end

    # Check all 6 neighbors
    for direction in 1:6
        neighbors = find_neighbor(octree, box_idx, direction)

        for neighbor_idx in neighbors
            if has_children(octree, neighbor_idx)
                # Check if neighbor's children also have children
                # This would create a 2-level difference if we don't subdivide
                for child_idx in octree.children[neighbor_idx]
                    if child_idx > 0 && has_children(octree, child_idx)
                        # Found a grandchild neighbor - balance violation!
                        return true
                    end
                end
            end
        end
    end

    return false
end

"""
    balance_octree!(octree::SpatialOctree, criterion::SubdivisionCriterion)

Enforce 2:1 balance constraint across entire octree.

Iteratively subdivides boxes that violate the 2:1 constraint until
all adjacent boxes differ by at most one refinement level.

# Arguments
- `criterion`: Subdivision criterion (only size constraints are enforced)

# Algorithm
1. Collect all leaves
2. Check each leaf for balance violations
3. Subdivide violating neighbors (only respecting physical limits like h_min)
4. Repeat until no violations

# Note
- Uses `can_subdivide` (not `should_subdivide`) to ignore element count criteria
- Balancing is a geometric constraint, not an optimization decision
- Maximum iterations limit prevents infinite loops. If hit, tree may not be fully balanced.
"""
function balance_octree!(octree::SpatialOctree, criterion::SubdivisionCriterion)
    max_iterations = 100  # Safety limit
    iteration = 0

    while iteration < max_iterations
        iteration += 1
        leaves_to_check = all_leaves(octree)
        subdivided_any = false

        for leaf_idx in leaves_to_check
            if !is_leaf(octree, leaf_idx)
                continue  # Already subdivided in this iteration
            end

            if needs_balancing(octree, leaf_idx)
                # Check if we CAN subdivide (only physical limits, not element count)
                if can_subdivide(criterion, octree, leaf_idx)
                    subdivide!(octree, leaf_idx)
                    subdivided_any = true
                end
            end
        end

        # If no boxes subdivided, we're done
        if !subdivided_any
            break
        end
    end

    if iteration >= max_iterations
        @warn "balance_octree! hit iteration limit - tree may not be fully balanced"
    end

    return nothing
end
