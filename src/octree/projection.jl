# Projection utilities for boundary point optimization

"""
    project_to_mesh(point, octree::TriangleOctree) -> (projected_point, normal, triangle_idx)

Project a point onto the nearest triangle in the mesh using octree spatial index.

Returns:
- `projected_point::SVector{3,T}`: Closest point on mesh surface
- `normal::SVector{3,T}`: Unit normal at projection location
- `triangle_idx::Int`: Index of triangle containing projection

Uses fast octree traversal to find nearest triangle, then computes exact projection.
"""
function project_to_mesh(
        point::SVector{3, T},
        octree::TriangleOctree{M, C, T},
    ) where {M, C, T <: Real}
    # Find nearest triangle using octree
    state = NearestTriangleState{T}(point)
    _nearest_triangle_octree!(point, octree.tree, octree.mesh, 1, state)

    if state.closest_idx == 0
        # No triangle found (should not happen for valid mesh)
        error("No nearest triangle found for point $point")
    end

    # Return projection point, normal, and triangle index
    tri_idx = state.closest_idx
    projected_pt = state.closest_pt
    normal = _get_triangle_normal(T, octree.mesh, tri_idx)

    return projected_pt, normal, tri_idx
end

"""
    project_to_mesh(point::Point, octree) -> (Point, normal_vector)

Project Meshes.jl Point onto mesh surface.

Returns Point and normal vector (as SVector) at projection location.
"""
function project_to_mesh(
        point::Point{𝔼{3}, C},
        octree::TriangleOctree{𝔼{3}, C, T},
    ) where {C, T}
    # Extract coordinates
    sv = _extract_vertex(T, point)

    # Project
    projected_sv, normal, _ = project_to_mesh(sv, octree)

    # Convert back to Point
    projected_point = Point(projected_sv[1], projected_sv[2], projected_sv[3])

    return projected_point, normal
end
