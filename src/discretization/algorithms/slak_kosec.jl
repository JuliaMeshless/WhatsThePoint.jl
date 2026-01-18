struct SlakKosec <: AbstractNodeGenerationAlgorithm
    n::Int
    SlakKosec(n::Int) = new(n)
end
SlakKosec() = SlakKosec(10)

function _discretize_volume(
    cloud::PointCloud{𝔼{3},C},
    spacing::AbstractSpacing,
    alg::SlakKosec;
    max_points=1_000,
) where {C}
    seeds = copy(points(boundary(cloud)))
    search_method = KNearestSearch(seeds, 1)
    new_points = Point{𝔼{3},C}[]

    # NOTE: InsideAccelerator disabled for discretization by default
    # The local Green's approximation (k_local neighbors) is inaccurate for interior points.
    # Discretization uses full Green's function (all boundary points) for correctness.

    i = 0
    points_since_rebuild = 0
    rebuild_interval = 100  # Rebuild KD-tree every N points for efficiency

    prog = ProgressMeter.Progress(
        max_points; desc="Generating nodes (SlakKosec): ", dt=1.0, barlen=40
    )
    while !isempty(seeds) && i < max_points
        p = popfirst!(seeds)
        r = spacing(p)
        candidates = _get_candidates(p, r; n=alg.n)
        for c in candidates
            # Use full Green's function for accurate isinside test
            if isinside(c, cloud)
                _, dist = searchdists(c, search_method)
                if first(dist) > r
                    push!(seeds, c)
                    push!(new_points, c)
                    points_since_rebuild += 1

                    # Rebuild KD-tree periodically instead of every iteration
                    if points_since_rebuild >= rebuild_interval
                        search_method = KNearestSearch(seeds, 1)
                        points_since_rebuild = 0
                    end

                    i += 1
                    ProgressMeter.update!(
                        prog, i; showvalues=[("Seeds remaining", length(seeds))]
                    )
                end
            end
        end

        if i > max_points
            @warn "discretization stopping early, reached max points ($max_points)"
            break
        end
    end

    ProgressMeter.finish!(prog)
    @info "Discretization complete" total_points = i
    return PointVolume(new_points)
end

function _get_candidates(p::Point{𝔼{3},C}, r; n=10) where {C}
    T = CoordRefSystems.mactype(C)

    u = rand(T, n)
    v = rand(T, n)

    one_T = one(T)
    ϕ = @. acos(2u - one_T) - oftype(one_T, π / 2)
    λ = 2π * v
    coords = to(p)
    unit_points = @. Point(r * cos(λ) * cos(ϕ), r * sin(λ) * cos(ϕ), r * sin(ϕ))
    return Ref(coords) .+ unit_points
end
