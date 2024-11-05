struct FornbergFlyer <: AbstractNodeGenerationAlgorithm end

function discretize!(
    cloud::PointCloud{𝔼{2},C}, spacing::AbstractSpacing, ::FornbergFlyer; max_points=10_000
) where {C}
    ninit = calculate_ninit(cloud, spacing)
    bbox = boundingbox(cloud)
    xmin, _ = to(bbox.min)
    xmax, _ = to(bbox.max)
    dx = (xmax - xmin) / (ninit - 1)

    x = xmin:dx:xmax

    heights = rand(length(x)) * spacing(cloud.points[1]) * 0.01 .+ last(bbox.min.coords)
    pdp = Point.(x, heights)
    _, current_id = findmin_turbo(heights)
    p = pdp[current_id]
    points = Vector{Point{𝔼{2},T}}(undef, max_points)

    dotnr = 1
    points[dotnr] = Point(p.coords[1], heights[current_id])
    tree = _balltree(pdp)

    prog = ProgressUnknown(; desc="generating nodes", spinner=true)
    while last(points[dotnr].coords) < last(bbox.max.coords) && dotnr < max_points
        ProgressMeter.next!(prog; spinner=spinner_icons)

        r = spacing(points[dotnr]) * 0.99
        inside_ids = _inrange(tree, p, r)

        dist = p .- pdp[inside_ids]
        new_heights = sqrt.(r^2 .- (getindex.(dist, 1)) .^ 2) .- getindex.(dist, 2)
        heights[inside_ids] .= max.(heights[current_id] .+ new_heights, heights[inside_ids])

        # naive search
        # TODO implement moving window search
        _, current_id = findmin_turbo(heights)

        p = pdp[current_id]
        dotnr += 1
        points[dotnr] = Point(p.coords[1], heights[current_id])
    end
    # REMOVE this
    shadow = mapreduce(vcat, surfaces(cloud)) do surf
        p = point(surf)
        p .- spacing(p[1]) * parent(surf).normal
    end
    inside_points = mapreduce(vcat, surfaces(cloud)) do surf
        p = point(surf)
        p .- spacing(p[1]) / 4 * parent(surf).normal
    end
    #shadow = Point.(generate_shadows(cloud, ShadowPoints(spacing, 1)))
    points = vfilter(x -> isinside(x, inside_points), points[1:dotnr])
    #points = vcat(points, shadow)

    #points = vfilter(x -> isinside(x, inside_points), points[1:dotnr])

    ProgressMeter.finish!(prog)

    N = length(cloud.points)
    append!(cloud.points, points)
    cloud.volume = PointVolume(view(cloud.points, (N + 1):length(cloud.points)))
    return nothing
end
