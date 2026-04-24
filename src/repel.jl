"""
    repel(cloud::PointCloud, spacing;
          force_model=InverseDistanceForce(β), β=0.2,
          α=auto, k=21, max_iters=1000, tol=1e-6, convergence=nothing)

Optimize point distribution via node repulsion (Miotti 2023).
Only volume points move; escaped points are discarded via `isinside` filtering.

The returned cloud has `NoTopology` since points have moved.

Pass a `Vector{Float64}` via the `convergence` keyword to collect the convergence history.

The force law is controlled by `force_model`, any subtype of [`RepelForceModel`](@ref).
The default [`InverseDistanceForce`](@ref) reproduces the original Miotti (2023) behavior.
Use [`SpacingEquilibriumForce`](@ref) for a force that vanishes at `r = s` (target
spacing) rather than relying on damping alone to stop movement. `β` is kept as a
convenience kwarg that feeds the default `InverseDistanceForce`; it is ignored when
`force_model` is passed explicitly.
"""
function repel(
        cloud::PointCloud{𝔼{N}, C},
        spacing;
        β = 0.2,
        force_model::RepelForceModel = InverseDistanceForce(β),
        α = minimum(spacing.(to(cloud))) * 0.05,
        k = 21,
        max_iters = 1000,
        tol = 1.0e-6,
        convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
    ) where {N, C <: CRS}
    α = ustrip(α)
    p = copy(volume(cloud).points)
    p_old = deepcopy(p)
    npoints = length(p)
    all_p = points(cloud)
    method = KNearestSearch(all_p, k)

    vol_spacings = ustrip.(spacing.(p))
    convergence_fn = let s = vol_spacings
        (p, p_old) -> norm(ustrip.(norm.(p .- p_old)) ./ s, Inf)
    end

    conv = Float64[]
    i = 1
    while i <= max_iters
        p_old .= p
        tmap!(p, 1:npoints) do id
            xi = p_old[id]
            ids, dists = searchdists(xi, method)
            ids = @view ids[2:end]
            neighborhood = @view all_p[ids]
            rij = norm.(@view dists[2:end])
            s = spacing(xi)

            repel_force = sum(zip(neighborhood, rij)) do z
                xj, r = z
                @inbounds compute_force(force_model, r / s) * (xi - xj) / r
            end

            return xi + Vec(s * α * repel_force)
        end
        push!(conv, convergence_fn(p, p_old))
        if conv[end] < tol
            @info "Node repel finished in $i iterations" convergence = conv[end]
            break
        end
        i = i + 1
    end
    if i > max_iters
        @warn "Node repel reached maximum iterations" max_iters convergence = conv[end]
    end
    if !isnothing(convergence)
        append!(convergence, conv)
    end
    new_volume = PointVolume(filter(x -> isinside(x, cloud), p))
    return PointCloud(boundary(cloud), new_volume, NoTopology())
end

"""
    repel(cloud::PointCloud, spacing, octree::TriangleOctree;
          force_model=InverseDistanceForce(β), β=0.2,
          α=auto, k=21, max_iters=1000, tol=1e-6, convergence=nothing)

Optimize point distribution via node repulsion (Miotti 2023) with boundary projection.

All points (boundary and volume) participate in repulsion. Points that escape the domain
are projected back to the nearest mesh triangle. Boundary points are always projected
back to the surface after each iteration.

The returned cloud has `NoTopology` since points have moved. The boundary is returned as a
single unified surface named `:boundary`. Use `split_surface!(cloud.boundary, angle)` to
re-establish surface distinctions if needed.

Pass a `Vector{Float64}` via the `convergence` keyword to collect the convergence history.

`force_model` accepts any subtype of [`RepelForceModel`](@ref); see the method without
`octree` for details on the available force laws.
"""
function repel(
        cloud::PointCloud{𝔼{3}, C},
        spacing,
        octree::TriangleOctree;
        β = 0.2,
        force_model::RepelForceModel = InverseDistanceForce(β),
        α = minimum(spacing.(to(cloud))) * 0.05,
        k = 21,
        max_iters = 1000,
        tol = 1.0e-6,
        convergence::Union{Nothing, AbstractVector{<:AbstractFloat}} = nothing,
    ) where {C <: CRS}
    α = ustrip(α)

    all_p = points(cloud)
    npoints_total = length(all_p)
    n_boundary = length(boundary(cloud))

    p = copy(all_p)
    p_old = copy(all_p)
    method = KNearestSearch(all_p, k)

    all_spacings = ustrip.(spacing.(p))
    convergence_fn = let s = all_spacings
        (p, p_old) -> norm(ustrip.(norm.(p .- p_old)) ./ s, Inf)
    end

    sample_coords = Meshes.to(first(all_p))
    len_unit = Unitful.unit(sample_coords[1])

    offset_dist = 1.0e-6 * norm(octree.mesh_bbox_max - octree.mesh_bbox_min)

    tri_indices = zeros(Int, npoints_total)

    conv = Float64[]
    i = 1
    while i <= max_iters
        p_old .= p
        tmap!(p, 1:npoints_total) do id
            xi = p_old[id]
            ids, dists = searchdists(xi, method)
            ids = @view ids[2:end]
            neighborhood = @view all_p[ids]
            rij = norm.(@view dists[2:end])
            s = spacing(xi)

            repel_force = sum(zip(neighborhood, rij)) do z
                xj, r = z
                @inbounds compute_force(force_model, r / s) * (xi - xj) / r
            end

            x_proposed = xi + Vec(s * α * repel_force)

            sv_proposed = _extract_vertex(Float64, x_proposed)
            sv_original = _extract_vertex(Float64, xi)
            is_bnd = id <= n_boundary

            sv_result, tri_idx = _constrain_to_domain(
                sv_proposed, sv_original, octree, is_bnd, offset_dist
            )
            tri_indices[id] = tri_idx

            return Point(
                sv_result[1] * len_unit, sv_result[2] * len_unit, sv_result[3] * len_unit
            )
        end

        push!(conv, convergence_fn(p, p_old))
        if conv[end] < tol
            @info "Node repel finished in $i iterations" convergence = conv[end]
            break
        end
        i += 1
    end
    if i > max_iters
        @warn "Node repel reached maximum iterations" max_iters convergence = conv[end]
    end
    if !isnothing(convergence)
        append!(convergence, conv)
    end

    return _reconstruct_cloud(cloud, p, tri_indices, n_boundary, octree)
end

function _reconstruct_cloud(
        cloud::PointCloud{𝔼{3}, C},
        p::AbstractVector{<:Point{𝔼{3}, C}},
        tri_indices::Vector{Int},
        n_boundary::Int,
        octree::TriangleOctree,
    ) where {C <: CRS}
    npoints_total = length(p)

    orig_normals = normal(boundary(cloud))
    orig_areas = area(boundary(cloud))

    new_bnd_pts = Point{𝔼{3}, C}[]
    new_bnd_normals = SVector{3, Float64}[]
    new_bnd_areas = eltype(orig_areas)[]
    new_vol_pts = Point{𝔼{3}, C}[]

    for id in 1:npoints_total
        if id <= n_boundary
            push!(new_bnd_pts, p[id])
            if tri_indices[id] > 0
                push!(new_bnd_normals, _get_triangle_normal(Float64, octree.mesh, tri_indices[id]))
            else
                push!(new_bnd_normals, orig_normals[id])
            end
            push!(new_bnd_areas, orig_areas[id])
        else
            push!(new_vol_pts, p[id])
        end
    end

    new_surf = PointSurface(new_bnd_pts, new_bnd_normals, new_bnd_areas)
    new_bnd = PointBoundary(LittleDict{Symbol, typeof(new_surf)}(:boundary => new_surf))
    new_vol = isempty(new_vol_pts) ? PointVolume{𝔼{3}, C}() : PointVolume(new_vol_pts)

    return PointCloud(new_bnd, new_vol, NoTopology())
end

function _project_to_boundary(
        sv::SVector{3, T}, octree::TriangleOctree, offset_dist::T,
    ) where {T <: Real}
    state = NearestTriangleState{T}(sv)
    _nearest_triangle_octree!(sv, octree.tree, octree.mesh, 1, state)
    state.closest_idx == 0 && return (sv, 0)

    v1, v2, v3 = _get_triangle_vertices(T, octree.mesh, state.closest_idx)
    projected = closest_point_on_triangle(sv, v1, v2, v3)

    n = _get_triangle_normal(T, octree.mesh, state.closest_idx)
    projected = projected - offset_dist * n

    return (projected, state.closest_idx)
end

function _constrain_to_domain(
        sv_proposed::SVector{3, T}, sv_original::SVector{3, T},
        octree::TriangleOctree, is_boundary::Bool, offset_dist::T,
    ) where {T <: Real}
    if is_boundary
        sv_result, tri_idx = _project_to_boundary(sv_proposed, octree, offset_dist)
        if tri_idx > 0
            return (sv_result, tri_idx)
        end
        return _project_to_boundary(sv_original, octree, offset_dist)
    end
    if isinside(sv_proposed, octree)
        return (sv_proposed, 0)
    end
    return (sv_original, 0)
end
