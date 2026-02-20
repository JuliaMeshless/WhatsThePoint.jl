# Layer 0: Trait System for Spatial Trees
#
# This file defines abstract types and traits that enable extensible
# spatial tree implementations. Designed for future extraction to
# standalone SpatialTrees.jl package.

#=============================================================================
Abstract Types
=============================================================================#

"""
    AbstractSpatialTree{N,E,T}

Abstract type for N-dimensional spatial trees.

# Type Parameters
- `N::Int`: Spatial dimensionality (2 for quadtree, 3 for octree)
- `E`: Element type stored in tree (e.g., Int for indices)
- `T<:Real`: Coordinate numeric type (Float64, Float32, etc.)

# Interface Requirements

Trees must implement:
- `find_leaf(tree, point)` - Locate leaf containing point
- `bounding_box(tree)` - Get overall bounds
- `num_elements(tree)` - Total elements stored

Optional:
- `find_neighbors(tree, box_idx, direction)` - Neighbor queries
- `balance!(tree)` - Enforce refinement constraints
"""
abstract type AbstractSpatialTree{N, E, T <: Real} end

"""
    AbstractOctree{E,T} = AbstractSpatialTree{3,E,T}

Convenience alias for 3D spatial trees (octrees).
"""
const AbstractOctree{E, T} = AbstractSpatialTree{3, E, T}

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
    SizeCriterion{T<:Real} <: SubdivisionCriterion

Subdivide box if size exceeds threshold.

# Fields
- `h_min::T`: Minimum box size (stop subdividing when reached)
"""
struct SizeCriterion{T <: Real} <: SubdivisionCriterion
    h_min::T
end

"""
    AndCriterion{T<:Tuple} <: SubdivisionCriterion

Combine multiple criteria - all must be satisfied for subdivision.

# Fields
- `criteria::T`: Tuple of criteria to combine (parametrized for type stability)
"""
struct AndCriterion{T <: Tuple} <: SubdivisionCriterion
    criteria::T
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
