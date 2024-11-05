function noderepel!(
    cloud::PointCloud,
    spacing;
    β=0.2,
    α=minimum(spacing(to(cloud))) * 0.05,
    k=21,
    max_iters=1e3,
    tol=1e-6,
)
    # Miotti 2023
    p = to(cloud.volume)
    p_old = deepcopy(p)
    N = length(p)
    all_p = cloud.points
    method = KNearestSearch(all_p, k)

    all_spacings = spacing(all_p)
    convergence = let s = all_spacings
        (p, p_old) -> norm(p .- p_old, Inf) ./ s
    end

    Δ = similar(p)
    conv = Float64[]
    i = 1
    F = let β = β
        r -> 1 / (r^2 + β)^2
    end
    while i < max_iters
        p_old .= p
        tmap!(p, 1:N) do id
            xi = p_old[id]
            #=
            ids, dists = knn(tree, xi, k + 1, true)
            ids = @view ids[2:end]
            neighborhood = @view all_p[ids]
            rij = @view dists[2:end]
            sbar = spacing(xi)

            repel = sum(zip(neighborhood, rij)) do z
                xj, r = z
                F(r / sbar) * (xi - xj) / r
            end
            =#

            ids, dists = searchdists(xi, method)
            ids = @view ids[2:end]
            neighborhood = @view all_p[ids]
            rij = norm.(@view dists[2:end])
            sbar = spacing(xi)

            repel = sum(zip(neighborhood, rij)) do z
                xj, r = z
                F(r / sbar) * (xi - xj) / r
            end

            return p_old[id] + sbar * α * repel
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
    cloud.volume.points .= Point.(p)
    return conv
end
