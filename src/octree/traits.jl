# Layer 0: Trait System for Spatial Trees
#
# This file defines abstract types and traits that enable extensible
# spatial tree implementations. Designed for future extraction to
# standalone SpatialTrees.jl package.

using StaticArrays

#=============================================================================
Abstract Types
=============================================================================#

"""
    AbstractSpatialTree{N,T,C}

Abstract type for N-dimensional spatial trees.

# Type Parameters
- `N::Int`: Spatial dimensionality (2 for quadtree, 3 for octree)
- `T`: Element type stored in tree (e.g., Int for indices)
- `C<:Real`: Coordinate numeric type (Float64, Float32, etc.)

# Interface Requirements

Trees must implement:
- `find_leaf(tree, point)` - Locate leaf containing point
- `bounding_box(tree)` - Get overall bounds
- `num_elements(tree)` - Total elements stored

Optional:
- `find_neighbors(tree, box_idx, direction)` - Neighbor queries
- `balance!(tree)` - Enforce refinement constraints
"""
abstract type AbstractSpatialTree{N, T, C <: Real} end

"""
    AbstractOctree{T,C} = AbstractSpatialTree{3,T,C}

Convenience alias for 3D spatial trees (octrees).
"""
const AbstractOctree{T, C} = AbstractSpatialTree{3, T, C}

"""
    AbstractQuadtree{T,C} = AbstractSpatialTree{2,T,C}

Convenience alias for 2D spatial trees (quadtrees).
"""
const AbstractQuadtree{T, C} = AbstractSpatialTree{2, T, C}

#=============================================================================
Element Insertion Traits
=============================================================================#

"""
    InsertionStrategy

Holy trait for element insertion behavior.

# Strategies
- `PointInsertion()`: Element belongs to exactly one box (e.g., points)
- `GeometricInsertion()`: Element can span multiple boxes (e.g., triangles, lines)

# Example
```julia
# Define insertion strategy for a type
insertion_strategy(::Type{Int}) = PointInsertion()
insertion_strategy(::Type{Triangle}) = GeometricInsertion()
```
"""
abstract type InsertionStrategy end

"""
    PointInsertion <: InsertionStrategy

Elements belong to exactly one box. Used for point-like objects.
"""
struct PointInsertion <: InsertionStrategy end

"""
    GeometricInsertion <: InsertionStrategy

Elements can span multiple boxes. Used for geometric objects like triangles, edges.
"""
struct GeometricInsertion <: InsertionStrategy end

"""
    insertion_strategy(::Type{T}) -> InsertionStrategy

Define insertion strategy for element type T.

Default is `PointInsertion()`. Override for geometric types.

# Examples
```julia
# Points go in single box
insertion_strategy(::Type{Int}) = PointInsertion()

# Triangles can span boxes
insertion_strategy(::Type{Triangle}) = GeometricInsertion()
```
"""
insertion_strategy(::Type{T}) where {T} = PointInsertion()  # Default

# Integer IDs always use point insertion
insertion_strategy(::Type{Int}) = PointInsertion()

#=============================================================================
Tree Construction Traits
=============================================================================#

"""
    SubdivisionCriterion

Abstract type for subdivision decision logic.

Allows pluggable subdivision criteria via dispatch.

# Built-in Criteria
- `MaxElementsCriterion(max_elements)` - Subdivide if too many elements
- `SizeCriterion(h_min)` - Subdivide until box small enough
- `AndCriterion(criteria...)` - All criteria must be satisfied

# Example
```julia
criterion = AndCriterion((
    MaxElementsCriterion(50),
    SizeCriterion(0.1)
))
```
"""
abstract type SubdivisionCriterion end

"""
    MaxElementsCriterion <: SubdivisionCriterion

Subdivide box if number of elements exceeds threshold.

# Fields
- `max_elements::Int`: Maximum elements before subdivision
"""
struct MaxElementsCriterion <: SubdivisionCriterion
    max_elements::Int
end

"""
    SizeCriterion{C<:Real} <: SubdivisionCriterion

Subdivide box if size exceeds threshold.

# Fields
- `h_min::C`: Minimum box size (stop subdividing when reached)
"""
struct SizeCriterion{C <: Real} <: SubdivisionCriterion
    h_min::C
end

"""
    AndCriterion <: SubdivisionCriterion

Combine multiple criteria - all must be satisfied for subdivision.

# Fields
- `criteria::Tuple{Vararg{SubdivisionCriterion}}`: Criteria to combine
"""
struct AndCriterion <: SubdivisionCriterion
    criteria::Tuple{Vararg{SubdivisionCriterion}}
end

# Convenience constructor
AndCriterion(criteria...) = AndCriterion(criteria)

"""
    should_subdivide(criterion::SubdivisionCriterion, tree, box_idx) -> Bool

Determine if box should be subdivided based on criterion.

# Arguments
- `criterion`: Subdivision criterion to evaluate
- `tree`: Spatial tree
- `box_idx`: Index of box to check

# Returns
`true` if box should be subdivided, `false` otherwise.
"""
function should_subdivide end

# Implementations for built-in criteria
function should_subdivide(c::MaxElementsCriterion, tree, box_idx)
    return length(tree.element_lists[box_idx]) > c.max_elements
end

function should_subdivide(c::SizeCriterion, tree, box_idx)
    return box_size(tree, box_idx) > c.h_min
end

function should_subdivide(c::AndCriterion, tree, box_idx)
    return all(should_subdivide(crit, tree, box_idx) for crit in c.criteria)
end

"""
    can_subdivide(criterion::SubdivisionCriterion, tree, box_idx) -> Bool

Check if box CAN be subdivided based on physical constraints only.

Unlike `should_subdivide`, this ignores content-based criteria (like element count)
and only checks physical limits (like minimum size). Used during balancing where
subdivision is required for geometric correctness, not optimization.

# Arguments
- `criterion`: Subdivision criterion (only size constraints are checked)
- `tree`: Spatial tree
- `box_idx`: Index of box to check

# Returns
`true` if box can physically be subdivided, `false` if at minimum size limit.

# Example
```julia
# For balancing, we only respect size limits
if needs_balancing(leaf)
    if can_subdivide(criterion, tree, leaf)
        subdivide!(tree, leaf)
    end
end
```
"""
function can_subdivide end

# Only check size constraints
can_subdivide(c::MaxElementsCriterion, tree, box_idx) = true  # No physical limit

function can_subdivide(c::SizeCriterion, tree, box_idx)
    return box_size(tree, box_idx) > c.h_min
end

function can_subdivide(c::AndCriterion, tree, box_idx)
    # Check all criteria, but MaxElements always returns true
    return all(can_subdivide(crit, tree, box_idx) for crit in c.criteria)
end

#=============================================================================
Utility Functions
=============================================================================#

"""
    box_size(tree::AbstractSpatialTree, box_idx::Int) -> Real

Get edge length of box.

# Required
Trees must implement this function.
"""
function box_size end

"""
    is_leaf(tree::AbstractSpatialTree, box_idx::Int) -> Bool

Check if box is a leaf (has no children).

# Required
Trees must implement this function.
"""
function is_leaf end

"""
    has_children(tree::AbstractSpatialTree, box_idx::Int) -> Bool

Check if box has been subdivided.

# Required
Trees must implement this function.
"""
function has_children end
