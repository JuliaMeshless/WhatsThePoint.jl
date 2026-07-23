# Graded Poisson-disk sampling of a triangle-mesh surface.
#
# Counterpart of the volume `:bridson` placement (src/discretization/algorithms/
# octree.jl, whose `_BridsonGrid` machinery this reuses): dart throwing on the
# continuous surface — area-weighted triangle pick, uniform sample within the
# triangle — accepted under the global criterion `‖xᵢ−xⱼ‖ ≥ min(rᵢ, rⱼ)` with
# `r = factor·h(x)`. Sampling the continuous surface (rather than thinning the
# imported face centers) both removes tessellation artifacts (near-coincident
# face centers at sphere poles, sliver triangles) and fills regions where the
# tessellation is coarser than the target spacing.
#
# The dart-throwing core (`_sample_surface`) works directly on a
# `TriangleIndex` over an arbitrary triangle range — the `SimpleMesh` door
# samples the whole index, the `Triangulation` door samples one patch range at
# a time into named surfaces. Normals always come from the index's precomputed
# face normals, so a `Triangulation`'s obstacle patches (winding reversed at
# load) automatically yield point normals flipped out of the fluid.
#
# Note: separation is measured by 3D Euclidean distance, so two opposite faces
# of a wall thinner than the local disk radius block each other's samples. Fine
# for geometry resolved by `h`; revisit if sub-`h` thin sheets become a use case.

"""
    sample_surface(mesh::SimpleMesh, spacing; factor=0.75, max_points=10_000_000,
                   stall_limit=2000) -> PointSurface

Sample the surface of `mesh` with a graded Poisson-disk distribution: blue-noise
points with pairwise separation at least `min(rᵢ, rⱼ)` where
`r = factor·spacing(x)`, by construction. An alternative to the face-center
sampling of `PointBoundary(mesh)` whose point density follows the prescribed
spacing instead of the mesh tessellation.

Each sample carries its parent triangle's normal; point areas preserve the total
mesh surface area, distributed proportionally to `r²` (equal shares for constant
spacing).

Dart throwing runs until saturation — `stall_limit` consecutive rejections — or
until `max_points` is reached (with a warning, since a truncated pass leaves the
surface under-sampled).
"""
function sample_surface(
        mesh::SimpleMesh{𝔼{3}, C},
        spacing::AbstractSpacing;
        factor::Real = 0.75,
        max_points::Int = 10_000_000,
        stall_limit::Int = 2000,
    ) where {C}
    T = CoordRefSystems.mactype(C)
    nelements(mesh) > 0 || throw(ArgumentError("mesh has no elements"))
    index = TriangleIndex(T, mesh)
    return _sample_surface(
        index, 1:num_triangles(index), spacing;
        factor = factor, max_points = max_points, stall_limit = stall_limit,
    )
end

# The dart-throwing core over a triangle range of a TriangleIndex. `tri_range`
# selects the triangles to sample (a whole index, or one patch of a
# Triangulation); point normals are read from `index.face`, so whatever
# orientation the index carries is what the samples get.
function _sample_surface(
        index::TriangleIndex{T},
        tri_range::AbstractUnitRange{Int},
        spacing::AbstractSpacing;
        factor::Real = 0.75,
        max_points::Int = 10_000_000,
        stall_limit::Int = 2000,
    ) where {T <: Real}
    factor > 0 || throw(ArgumentError("factor must be positive"))
    stall_limit > 0 || throw(ArgumentError("stall_limit must be positive"))
    n_tri = length(tri_range)
    n_tri > 0 || throw(ArgumentError("no triangles to sample"))
    f = T(factor)

    # Triangle areas (for area-weighted sampling) and domain bounds.
    tri_areas = Vector{T}(undef, n_tri)
    gmin = SVector{3, T}(typemax(T), typemax(T), typemax(T))
    gmax = SVector{3, T}(typemin(T), typemin(T), typemin(T))
    r_min = typemax(T)
    for (i, tri_idx) in enumerate(tri_range)
        v1, v2, v3 = _get_triangle_vertices(index, tri_idx)
        tri_areas[i] = norm(cross(v2 - v1, v3 - v1)) / 2
        gmin = min.(gmin, min.(v1, min.(v2, v3)))
        gmax = max.(gmax, max.(v1, max.(v2, v3)))
        r_min = min(r_min, f * _spacing_value(T, spacing, (v1 + v2 + v3) / 3))
    end
    cum_areas = cumsum(tri_areas)
    total_area = cum_areas[end]
    total_area > 0 || throw(ArgumentError("triangles have zero surface area"))
    r_min = max(r_min, sqrt(eps(T)))

    grid = _BridsonGrid(gmin, gmax, r_min)
    pts = SVector{3, T}[]
    rs = T[]
    tri_of = Int[]

    misses = 0
    while length(pts) < max_points && misses < stall_limit
        tri = tri_range[clamp(searchsortedfirst(cum_areas, rand(T) * total_area), 1, n_tri)]
        v1, v2, v3 = _get_triangle_vertices(index, tri)
        # Uniform point in the triangle (square-root trick).
        su = sqrt(rand(T))
        v = rand(T)
        c = (1 - su) * v1 + su * (1 - v) * v2 + su * v * v3
        r_c = f * _spacing_value(T, spacing, c)
        if _bridson_separated(grid, pts, rs, c, r_c)
            push!(pts, c)
            push!(rs, r_c)
            push!(tri_of, tri)
            _grid_insert!(grid, c)
            misses = 0
        else
            misses += 1
        end
    end
    isempty(pts) && throw(ArgumentError("surface sampling produced no points — check spacing vs mesh size"))
    length(pts) >= max_points &&
        @warn "Surface sampling truncated by max_points before saturation — the surface is under-sampled" max_points

    len_unit = index.len_unit
    sample_points = [Point((c .* len_unit)...) for c in pts]
    sample_normals = [@inbounds index.face[t] for t in tri_of]
    # Total-area-preserving shares, proportional to the local disk area.
    w = rs .^ 2
    sample_areas = (total_area / sum(w)) .* w .* len_unit^2

    return PointSurface(sample_points, sample_normals, sample_areas)
end

"""
    PointBoundary(mesh::SimpleMesh, spacing::AbstractSpacing; kwargs...)

Create a `PointBoundary` by Poisson-disk sampling the mesh surface at the
prescribed `spacing` (see [`sample_surface`](@ref) for the keywords), instead
of taking the tessellation's face centers as `PointBoundary(mesh)` does.
"""
function PointBoundary(mesh::SimpleMesh{𝔼{3}}, spacing::AbstractSpacing; kwargs...)
    surf = sample_surface(mesh, spacing; kwargs...)
    M = manifold(surf)
    C = crs(surf)
    return PointBoundary(LittleDict{Symbol, PointSurface{M, C}}(:surface1 => surf))
end

"""
    PointBoundary(tri::Triangulation, spacing::AbstractSpacing; kwargs...)

Create a `PointBoundary` by Poisson-disk sampling every patch of the
triangulation at the prescribed `spacing` (see [`sample_surface`](@ref) for the
keywords). Each patch becomes a named surface — patch name → surface name — so
boundary conditions stay taggable per patch.

Point normals come from the merged triangle index: container patches point
outward, obstacle patches (winding reversed at load) point into the obstacle —
all **out of the fluid domain**, matching the `TriangleOctree` convention.

Patches are sampled independently, so separation is enforced within each patch,
not across patches — fine by design (obstacles must not touch the container);
the volume fill still enforces boundary↔interior separation, seeded from these
points.
"""
function PointBoundary(tri::Triangulation, spacing::AbstractSpacing; kwargs...)
    pairs = map(patches(tri)) do name
        name => _sample_surface(tri.index, patch_range(tri, name), spacing; kwargs...)
    end
    M = manifold(first(pairs).second)
    C = crs(first(pairs).second)
    return PointBoundary(LittleDict{Symbol, PointSurface{M, C}}(pairs))
end
