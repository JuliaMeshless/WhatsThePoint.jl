struct SlakKosec <: AbstractNodeGenerationAlgorithm
    n::Int
    SlakKosec(n::Int) = new(n)
end
SlakKosec() = SlakKosec(10)

function discretize!(
    cloud::PointCloud{𝔼{3},C}, spacing::AbstractSpacing, alg::SlakKosec; max_points=10_000
) where {C}
    seeds = Point.(to(cloud.surfaces))
    tree = _kdtree(seeds)
    new_points = Point{𝔼{3},T}[]

    i = 0
    while !isempty(seeds) && i < max_points
        p = popfirst!(seeds)
        r = spacing(p)
        candidates = _get_candidates(p, r; n=alg.n)
        for c in candidates
            if isinside(c, cloud)
                _, dist = nn(tree, c)
                if dist > r
                    push!(seeds, c)
                    push!(new_points, c)
                    tree = _kdtree(seeds)
                    i += 1
                end
            end
        end
        if i > max_points
            @warn "discretization stopping early, reached max points ($max_points)"
            break
        end
    end

    return new_points
end

function _get_candidates(p::Point{𝔼{3},C}, r::Real; n=10) where {C}
    T = Meshes.lentype(C)
    u = rand(T, n)
    v = rand(T, n)

    ϕ = @. acos(2u - 1) - π / 2
    λ = 2π * v
    coords = to(p)
    unit_points = @. Point(r * cos(λ) * cos(ϕ), r * sin(λ) * cos(ϕ), r * sin(ϕ))
    return (coords,) .+ unit_points
end
