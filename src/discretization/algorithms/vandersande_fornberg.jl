struct VanDerSandeFornberg <: AbstractNodeGenerationAlgorithm end

function _discretize_volume(
    cloud::PointCloud{𝔼{3},C},
    spacing::ConstantSpacing,
    ::VanDerSandeFornberg;
    max_points=10_000_000,
) where {C}
    # NOTE: InsideAccelerator disabled for discretization by default
    # The local Green's approximation (k_local neighbors) is inaccurate for interior points.
    # Discretization uses full Green's function (all boundary points) for correctness.

    ninit = calculate_ninit(cloud, spacing)
    bbox = boundingbox(cloud)
    xmin, ymin, _ = to(bbox.min)
    xmax, ymax, _ = to(bbox.max)
    dx = (xmax - xmin) / (ninit[1] - 1)
    dy = (ymax - ymin) / (ninit[2] - 1)

    S = Meshes.Shadow(:xy)
    pdp_grid = CartesianGrid(S(bbox.min), S(bbox.max), (dx, dy))
    pdp = Meshes.vertices(pdp_grid)
    T = CoordRefSystems.mactype(C)
    heights =
        rand(T, length(pdp)) * spacing(points(cloud)[1]) * 0.01 .+ coords(bbox.min).z
    _, current_id = findmin_turbo(heights)
    p = pdp[current_id]
    new_points = Vector{Point{𝔼{3},C}}(undef, max_points)

    dotnr = 1
    c = coords(p)
    new_points[dotnr] = Point(c.x, c.y, heights[current_id])
    r = spacing(new_points[dotnr]) * 0.99
    search_method = BallSearch(pdp, MetricBall(r))

    prog = ProgressUnknown(; desc="generating nodes", spinner=true)
    while coords(new_points[dotnr]).z < coords(bbox.max).z
        if dotnr > (max_points - 1)
            @warn "discretization stopping early, reached max points ($max_points)"
            break
        end

        ProgressMeter.next!(prog; spinner=spinner_icons)
        inside_ids = search(p, search_method)

        xydist = map(pdp_inside -> norm(p - pdp_inside), @view pdp[inside_ids])
        new_heights = @. sqrt(r^2 - xydist^2) + heights[current_id]
        heights[inside_ids] .= max.(new_heights, heights[inside_ids])

        # naive search
        # TODO implement moving window search
        _, current_id = findmin_turbo(heights)

        p = pdp[current_id]
        dotnr += 1
        c = coords(p)
        new_points[dotnr] = Point(c.x, c.y, heights[current_id])
    end

    # Filter points using full Green's function (no accelerator)
    new_points = filter(x -> isinside(x, cloud), new_points[1:dotnr])

    ProgressMeter.finish!(prog)

    return PointVolume(new_points)
end
