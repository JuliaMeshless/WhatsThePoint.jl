"""
    repel(cloud::PointCloud, spacing, octree; β=0.2, α=auto, k=21, max_iters=1000, tol=1e-6)

Optimize point distribution via spacing-aware node repulsion (Miotti 2023).

Moves both boundary and volume points to satisfy spacing prescription:
- Boundary points are projected back to mesh surface after each iteration
- Normals are recomputed at new projection locations
- Volume points are filtered to remain inside the domain

Returns `(new_cloud, convergence_vector)` tuple.
The returned cloud has `NoTopology` since points have moved.

# Arguments
- `cloud`: PointCloud to optimize
- `spacing`: Spacing function (ConstantSpacing, BoundaryLayerSpacing, etc.)
- `octree`: TriangleOctree for fast mesh projection
- `β`: Repulsion strength parameter (default: 0.2)
- `α`: Step size (default: 5% of minimum spacing)
- `k`: Number of nearest neighbors for force computation (default: 21)
- `max_iters`: Maximum iterations (default: 1000)
- `tol`: Convergence tolerance (default: 1e-6)
"""
function repel(
    cloud::PointCloud{𝔼{3},C},
    spacing,
    octree::TriangleOctree{𝔼{3},C,T};
    β=0.2,
    α=nothing,
    k=21,
    max_iters=1.0e3,
    tol=1.0e-6,
) where {C<:CRS,T<:Real}
    if isnothing(α)
        α = minimum(spacing.(to(cloud))) * 0.05
    end
    α = ustrip(α)
    alpha_step = Float64(α)

    # Extract all points (boundary + volume)
    boundary_pts = collect(points(boundary(cloud)))
    volume_pts = collect(points(volume(cloud)))
    n_boundary = length(boundary_pts)
    n_volume = length(volume_pts)
    n_total = n_boundary + n_volume

    # Combine into single array for efficient k-NN
    all_pts = vcat(boundary_pts, volume_pts)
    all_pts_old = deepcopy(all_pts)

    # Spacing evaluation
    all_spacings = spacing.(all_pts)
    convergence = let s = all_spacings
        (p, p_old) -> ustrip.(norm.(p .- p_old, Inf) ./ s)
    end

    # Repulsion force function
    F = let β = β
        r -> 1 / (r^2 + β)^2
    end

    conv = Vector{Vector{Float64}}()
    i = 1

    while i < max_iters
        # Build k-NN search with current positions
        method = KNearestSearch(all_pts, k)
        all_pts_old .= all_pts

        # Update all points with repulsion forces
        tmap!(all_pts, 1:n_total) do id
            xi = all_pts_old[id]
            ids, dists = searchdists(xi, method)
            ids = @view ids[2:end]
            neighborhood = @view all_pts_old[ids]
            rij = norm.(@view dists[2:end])
            s = spacing(xi)

            s_val = Float64(ustrip(s))
            if !isfinite(s_val) || s_val <= eps(Float64)
                return xi
            end

            # Force is dimensionless because (xi - xj) has units of length and r has units of length.
            repel_force = zero((xi - xi) / s)
            for (xj, r) in zip(neighborhood, rij)
                r_val = Float64(ustrip(r))
                if !isfinite(r_val) || r_val <= eps(Float64)
                    continue
                end

                f_val = Float64(ustrip(F(r / s)))
                if !isfinite(f_val)
                    continue
                end

                repel_force += f_val * (xi - xj) / r
            end

            delta = s * alpha_step * repel_force
            if !all(isfinite, ustrip.(delta))
                return xi
            end

            return xi + Vec(delta)
        end

        # Project boundary points back to mesh surface
        for id in 1:n_boundary
            if !all(isfinite, ustrip.(to(all_pts[id])))
                all_pts[id] = all_pts_old[id]
                continue
            end
            projected_pt, _ = project_to_mesh(all_pts[id], octree)
            all_pts[id] = projected_pt
        end

        # Check convergence
        push!(conv, convergence(all_pts, all_pts_old))
        if all(<(tol), conv[end])
            println("Node repel finished in $i iterations. Convergence = $(maximum(conv[end]))")
            break
        end
        i += 1
    end

    if i == max_iters
        @warn "Node repel reached maximum number of iterations ($max_iters), Convergence = $(maximum(conv[end]))\n"
    end

    # Rebuild boundary with updated points and recomputed normals
    updated_boundary_pts = all_pts[1:n_boundary]
    updated_volume_pts = all_pts[(n_boundary+1):end]

    # Filter volume points to keep only those inside
    inside_mask = isinside(updated_volume_pts, octree)
    filtered_volume_pts = updated_volume_pts[inside_mask]

    # Recompute normals and areas for boundary points
    new_normals = Vector{SVector{3,T}}(undef, n_boundary)
    for i in 1:n_boundary
        p = to(updated_boundary_pts[i])
        _, normal, _ = project_to_mesh(
            SVector{3,T}(
                T(ustrip(p[1])),
                T(ustrip(p[2])),
                T(ustrip(p[3]))
            ), octree
        )
        new_normals[i] = normal
    end

    # Estimate areas (simple approximation: keep original ratios)
    # TODO: Better area estimation based on local point density
    old_boundary = boundary(cloud)
    first_surface = first(values(old_boundary.surfaces))
    old_areas = similar(area(first_surface), 0)
    for surf in values(old_boundary.surfaces)
        append!(old_areas, area(surf))
    end
    new_areas = old_areas  # Keep original areas for now

    # Rebuild surfaces (assuming single surface for now - TODO: handle multiple surfaces)
    surf_name = first(keys(old_boundary.surfaces))
    new_surface = PointSurface(updated_boundary_pts, new_normals, new_areas)
    new_boundary = PointBoundary(LittleDict(surf_name => new_surface))

    # Rebuild volume
    new_volume = PointVolume(filtered_volume_pts)

    # Return new cloud
    new_cloud = PointCloud(new_boundary, new_volume, NoTopology())
    return (new_cloud, conv)
end
