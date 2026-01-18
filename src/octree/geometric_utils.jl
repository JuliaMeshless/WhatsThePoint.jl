# Layer 2: Geometric Utilities for Triangle Operations
#
# This file provides geometric primitives needed for triangle-octree implementation.
# Functions work on raw coordinates (SVector) for generality and performance.
# Unit handling should be done at a higher level.
#
# References:
# - AdjointMeshlessCube geoUtilities.jl
# - Ericson, "Real-Time Collision Detection" (2004)

using StaticArrays
using LinearAlgebra

#=============================================================================
Point-Triangle Distance
=============================================================================#

"""
    closest_point_on_triangle(
        P::SVector{3,T},
        v1::SVector{3,T},
        v2::SVector{3,T},
        v3::SVector{3,T}
    ) where {T<:Real} -> SVector{3,T}

Compute the closest point on triangle (v1, v2, v3) to point P.

Uses barycentric coordinate method from Ericson's "Real-Time Collision Detection".
The closest point is computed by:
1. Projecting P onto the triangle plane
2. Computing barycentric coordinates
3. Clamping to triangle if outside

# Algorithm
The triangle can be parameterized as:
    T(u,v) = v1 + u*(v2-v1) + v*(v3-v1)  for u,v ≥ 0, u+v ≤ 1

We find the closest point by solving a constrained minimization problem.

# Returns
Point on triangle surface closest to P (may be on edge or vertex).

# References
Ericson, "Real-Time Collision Detection", Chapter 5.1.5
"""
function closest_point_on_triangle(
    P::SVector{3,T},
    v1::SVector{3,T},
    v2::SVector{3,T},
    v3::SVector{3,T}
) where {T<:Real}
    # Triangle vertices
    a = v1
    b = v2
    c = v3

    # Edge vectors
    ab = b - a
    ac = c - a
    ap = P - a

    # Barycentric coordinates computation
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    # Check if P is in vertex region outside A
    if d1 <= 0 && d2 <= 0
        return a  # Barycentric (1,0,0)
    end

    # Check if P is in vertex region outside B
    bp = P - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0 && d4 <= d3
        return b  # Barycentric (0,1,0)
    end

    # Check if P is in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 / (d1 - d3)
        return a + v * ab  # Barycentric (1-v, v, 0)
    end

    # Check if P is in vertex region outside C
    cp = P - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    if d6 >= 0 && d5 <= d6
        return c  # Barycentric (0,0,1)
    end

    # Check if P is in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 / (d2 - d6)
        return a + w * ac  # Barycentric (1-w, 0, w)
    end

    # Check if P is in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)  # Barycentric (0, 1-w, w)
    end

    # P is inside triangle face region
    # Use barycentric coordinates
    denom = 1 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w  # Barycentric (1-v-w, v, w)
end

"""
    distance_point_triangle(
        P::SVector{3,T},
        v1::SVector{3,T},
        v2::SVector{3,T},
        v3::SVector{3,T},
        normal::SVector{3,T}
    ) where {T<:Real} -> T

Compute signed distance from point P to triangle (v1, v2, v3).

The distance is:
- **Positive** if P is on the side of the triangle that the normal points to
- **Negative** if P is on the opposite side
- **Zero** if P is on the triangle plane

# Algorithm
1. Find closest point Q on triangle to P
2. Compute distance ||P - Q||
3. Determine sign based on which side of triangle P is on

# Arguments
- `P`: Query point
- `v1, v2, v3`: Triangle vertices in counterclockwise order
- `normal`: Outward-pointing unit normal vector

# Returns
Signed distance (positive = outside, negative = inside for closed surface)

# Example
```julia
using StaticArrays

# Triangle in xy-plane
v1 = SVector(0.0, 0.0, 0.0)
v2 = SVector(1.0, 0.0, 0.0)
v3 = SVector(0.0, 1.0, 0.0)
normal = SVector(0.0, 0.0, 1.0)

# Point above triangle
P = SVector(0.25, 0.25, 1.0)
d = distance_point_triangle(P, v1, v2, v3, normal)
# d ≈ 1.0 (positive, on normal side)
```
"""
function distance_point_triangle(
    P::SVector{3,T},
    v1::SVector{3,T},
    v2::SVector{3,T},
    v3::SVector{3,T},
    normal::SVector{3,T}
) where {T<:Real}
    # Find closest point on triangle
    Q = closest_point_on_triangle(P, v1, v2, v3)

    # Compute unsigned distance
    diff = P - Q
    dist = norm(diff)

    # Determine sign based on which side of triangle P is on
    # Use vector from v1 to P
    v1_to_P = P - v1

    # If dot product is positive, P is on the normal side (outside)
    sign = dot(v1_to_P, normal) >= 0 ? 1 : -1

    return sign * dist
end

"""
    distance_point_triangle(
        P::SVector{3,T},
        v1::SVector{3,T},
        v2::SVector{3,T},
        v3::SVector{3,T}
    ) where {T<:Real} -> T

Compute unsigned distance from point P to triangle (v1, v2, v3).

# Returns
Unsigned distance (always positive or zero)
"""
function distance_point_triangle(
    P::SVector{3,T},
    v1::SVector{3,T},
    v2::SVector{3,T},
    v3::SVector{3,T}
) where {T<:Real}
    # Find closest point on triangle
    Q = closest_point_on_triangle(P, v1, v2, v3)

    # Compute unsigned distance
    return norm(P - Q)
end

#=============================================================================
Triangle-Box Intersection
=============================================================================#

"""
    _triangle_axis_test(
        axis::SVector{3,T},
        v0::SVector{3,T},
        v1::SVector{3,T},
        v2::SVector{3,T},
        half::SVector{3,T}
    ) where {T<:Real} -> Bool

Internal helper for triangle-box intersection separating axis test.

Tests if the projection intervals of the triangle vertices and box
overlap along the given axis.

# Arguments
- `axis`: Separating axis direction
- `v0, v1, v2`: Triangle vertices in box-centered coordinates
- `half`: Box half-extents

# Returns
`true` if intervals overlap (potential intersection), `false` if separated
"""
@inline function _triangle_axis_test(
    axis::SVector{3,T},
    v0::SVector{3,T},
    v1::SVector{3,T},
    v2::SVector{3,T},
    half::SVector{3,T}
) where {T<:Real}
    # Project triangle vertices onto axis
    p0 = dot(v0, axis)
    p1 = dot(v1, axis)
    p2 = dot(v2, axis)

    # Box projection radius along axis
    r = abs(axis[1]) * half[1] + abs(axis[2]) * half[2] + abs(axis[3]) * half[3]

    # Check if intervals overlap
    return min(p0, p1, p2) <= r && max(p0, p1, p2) >= -r
end

"""
    triangle_box_intersection(
        v1::SVector{3,T},
        v2::SVector{3,T},
        v3::SVector{3,T},
        box_min::SVector{3,T},
        box_max::SVector{3,T}
    ) where {T<:Real} -> Bool

Test if triangle (v1, v2, v3) intersects axis-aligned box.

Uses the Separating Axis Theorem (SAT) with 13 potential separating axes:
- 3 box face normals (x, y, z axes)
- 1 triangle normal
- 9 edge-edge cross products

If any axis separates the triangle and box, they don't intersect.

# Algorithm
1. Translate triangle and box so box is centered at origin
2. Test each potential separating axis
3. Return false if any axis separates, true otherwise

# References
- Akenine-Möller, "Fast 3D Triangle-Box Overlap Testing" (2001)
- Ericson, "Real-Time Collision Detection", Chapter 5.2.9

# Performance
Optimized with early-out tests. Average case is much faster than worst case.
"""
function triangle_box_intersection(
    v1::SVector{3,T},
    v2::SVector{3,T},
    v3::SVector{3,T},
    box_min::SVector{3,T},
    box_max::SVector{3,T}
) where {T<:Real}
    # Box center and half-extents
    box_center = (box_min + box_max) / 2
    box_half = (box_max - box_min) / 2

    # Translate triangle to box-centered coordinates
    v0 = v1 - box_center
    v1_t = v2 - box_center
    v2_t = v3 - box_center

    # Triangle edges
    e0 = v1_t - v0
    e1 = v2_t - v1_t
    e2 = v0 - v2_t

    # Test 1: AABB face normals (3 tests)
    # X-axis
    min_val = min(v0[1], v1_t[1], v2_t[1])
    max_val = max(v0[1], v1_t[1], v2_t[1])
    if min_val > box_half[1] || max_val < -box_half[1]
        return false
    end

    # Y-axis
    min_val = min(v0[2], v1_t[2], v2_t[2])
    max_val = max(v0[2], v1_t[2], v2_t[2])
    if min_val > box_half[2] || max_val < -box_half[2]
        return false
    end

    # Z-axis
    min_val = min(v0[3], v1_t[3], v2_t[3])
    max_val = max(v0[3], v1_t[3], v2_t[3])
    if min_val > box_half[3] || max_val < -box_half[3]
        return false
    end

    # Test 2: Triangle normal
    normal = cross(e0, e1)
    d = -dot(normal, v0)

    # Box vertices in normal direction
    r = abs(normal[1]) * box_half[1] + abs(normal[2]) * box_half[2] + abs(normal[3]) * box_half[3]

    if abs(d) > r
        return false
    end

    # Test 3: Edge-edge cross products (9 tests)
    # X-axis × triangle edges
    axis = SVector{3,T}(0, -e0[3], e0[2])
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    axis = SVector{3,T}(0, -e1[3], e1[2])
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    axis = SVector{3,T}(0, -e2[3], e2[2])
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    # Y-axis × triangle edges
    axis = SVector{3,T}(e0[3], 0, -e0[1])
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    axis = SVector{3,T}(e1[3], 0, -e1[1])
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    axis = SVector{3,T}(e2[3], 0, -e2[1])
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    # Z-axis × triangle edges
    axis = SVector{3,T}(-e0[2], e0[1], 0)
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    axis = SVector{3,T}(-e1[2], e1[1], 0)
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    axis = SVector{3,T}(-e2[2], e2[1], 0)
    if !_triangle_axis_test(axis, v0, v1_t, v2_t, box_half)
        return false
    end

    # All tests passed - triangle intersects box
    return true
end

"""
    boxes_intersected_by_triangle(
        v1::SVector{3,T},
        v2::SVector{3,T},
        v3::SVector{3,T},
        parent_min::SVector{3,T},
        parent_size::T,
        subdivision::Int = 2
    ) where {T<:Real} -> Vector{NTuple{3,Int}}

Find which child boxes of a subdivided parent box are intersected by a triangle.

For a parent box subdivided into `subdivision³` children, returns the (i,j,k)
coordinates of each child box that the triangle intersects.

# Arguments
- `v1, v2, v3`: Triangle vertices
- `parent_min`: Minimum corner of parent box
- `parent_size`: Edge length of parent box
- `subdivision`: Number of subdivisions per axis (default: 2 for octree)

# Returns
Vector of (i,j,k) tuples where each component ∈ [0, subdivision-1]

# Example
```julia
# Triangle spanning multiple octree children
v1 = SVector(0.0, 0.0, 0.0)
v2 = SVector(10.0, 0.0, 0.0)
v3 = SVector(5.0, 10.0, 0.0)

parent_min = SVector(0.0, 0.0, 0.0)
parent_size = 10.0

boxes = boxes_intersected_by_triangle(v1, v2, v3, parent_min, parent_size)
# Returns: [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]  # 4 children in xy-plane
```
"""
function boxes_intersected_by_triangle(
    v1::SVector{3,T},
    v2::SVector{3,T},
    v3::SVector{3,T},
    parent_min::SVector{3,T},
    parent_size::T,
    subdivision::Int = 2
) where {T<:Real}
    child_size = parent_size / subdivision
    intersected = NTuple{3,Int}[]

    # Check each child box
    for k in 0:(subdivision-1), j in 0:(subdivision-1), i in 0:(subdivision-1)
        # Child box bounds
        box_min = parent_min + SVector{3,T}(i, j, k) * child_size
        box_max = box_min + SVector{3,T}(child_size, child_size, child_size)

        # Test intersection
        if triangle_box_intersection(v1, v2, v3, box_min, box_max)
            push!(intersected, (i, j, k))
        end
    end

    return intersected
end
