"""
    repel(cloud::PointCloud, spacing; β=0.2, α=auto, k=21, max_iters=1000, tol=1e-6)

Optimize point distribution via node repulsion (Miotti 2023).
Returns `(new_cloud, convergence_vector)` tuple.

The returned cloud has `NoTopology` since points have moved.
"""
function repel(
    cloud::PointCloud{𝔼{N},C},
    spacing;
    β=0.2,
    α=minimum(spacing(to(cloud))) * 0.05,
    k=21,
    max_iters=1e3,
    tol=1e-6,
) where {N,C<:CRS}
    # Miotti 2023
    α = ustrip(α)
    p = collect(volume(cloud).points)
    p_old = deepcopy(p)
    npoints = length(p)
    all_p = pointify(cloud)
    method = KNearestSearch(all_p, k)

    all_spacings = spacing(all_p)
    convergence = let s = all_spacings
        (p, p_old) -> norm(p .- p_old, Inf) ./ s
    end

    conv = Float64[]
    i = 1
    F = let β = β
        r -> 1 / (r^2 + β)^2
    end
    while i < max_iters
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
                F(r / s) * (xi - xj) / r
            end

            return xi + Vec(s * α * repel_force)
        end
        push!(conv, convergence(p, p_old))
        if all(x -> norm(x, Inf) < tol, conv[end])
            println("Node repel finished in $i iterations. Convergence = ($conv[end])")
            break
        end
        i = i + 1
    end
    if i == max_iters
        @warn "Node repel reached maximum number of iterations ($max_iters), Convergence = ($(conv[end]))\n"
    end
    new_volume = PointVolume(filter(x -> isinside(x, cloud), p))
    new_cloud = PointCloud(boundary(cloud), new_volume, NoTopology())
    return (new_cloud, conv)
end
