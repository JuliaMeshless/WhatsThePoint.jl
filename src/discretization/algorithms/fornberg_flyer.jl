struct FornbergFlyer <: AbstractNodeGenerationAlgorithm end

function _discretize_volume(
    cloud::PointCloud{𝔼{2},C}, spacing::ConstantSpacing, ::FornbergFlyer; max_points=10_000
) where {C}
    ninit = calculate_ninit(cloud, spacing)
    bbox = boundingbox(cloud)
    xmin, _ = to(bbox.min)
    xmax, _ = to(bbox.max)
    dx = (xmax - xmin) / (ninit - 1)
    b = boundary(cloud)
    r = spacing()

    x = xmin:dx:xmax

    heights = rand(length(x)) * spacing(b[1]) * 0.01 .+ bbox.min.coords.y
    pdp = Point.(x, heights)
    _, current_id = findmin_turbo(heights)
    p = pdp[current_id]
    new_points = Vector{Point{𝔼{2},C}}(undef, max_points)

    dotnr = 1
    new_points[dotnr] = Point(p.coords.x, heights[current_id])

    # Build BallTree using NearestNeighbors.jl
    pdp_matrix = reduce(hcat, [SVector(ustrip.(to(pt))...) for pt in pdp])
    tree = BallTree(pdp_matrix)
    r_val = ustrip(r)

    prog = ProgressUnknown(; desc="generating nodes", spinner=true)
    while new_points[dotnr].coords.y < bbox.max.coords.y && dotnr < max_points
        ProgressMeter.next!(prog; spinner=spinner_icons)

        p_vec = SVector(ustrip.(to(p))...)
        inside_ids = NearestNeighbors.inrange(tree, p_vec, r_val)

        dist = p .- pdp[inside_ids]
        new_heights = sqrt.(r^2 .- (getindex.(dist, 1)) .^ 2) .- getindex.(dist, 2)
        heights[inside_ids] .= max.(heights[current_id] .+ new_heights, heights[inside_ids])

        # naive search
        # TODO implement moving window search
        _, current_id = findmin_turbo(heights)

        p = pdp[current_id]
        dotnr += 1
        new_points[dotnr] = Point(p.coords.x, heights[current_id])
    end

    new_points = filter(x -> isinside(x, cloud), new_points[1:dotnr])
    ProgressMeter.finish!(prog)

    return PointVolume(new_points)
end
