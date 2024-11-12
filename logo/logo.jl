using PointClouds
using Unitful: m, ustrip
using Luxor

function get_xy(cloud)
    points = cloud.points
    x = map(p -> ustrip(PointClouds.coords(p).x), points)
    y = map(p -> ustrip(PointClouds.coords(p).y), points)
    return x, y
end

function circle_nodes(O)
    r = 0.75
    N = 12
    θ = 0:(π / N):(2π - π / (2N))
    spacing = ConstantSpacing(3 * r / N * m)

    points = PointClouds.Point.([(r * cos(i), r * sin(i)) for i in θ]) .+ (O,)
    part = PointPart(points)
    cloud = discretize(part, spacing; max_points=2000)

    return get_xy(cloud)
end

Oy = 2cos(π / 6) - 1
origins = [Vec(-1, Oy), Vec(1, Oy), Vec(0, -1)] #.+ Ref(Vec(2.5, -2.5))
nodes = circle_nodes.(origins)
colors = (Luxor.julia_red, Luxor.julia_green, Luxor.julia_purple)

function draw_logo(colors, nodes, scale, r)
    for (color, node) in zip(colors, nodes)
        setcolor(color)
        circle.(Luxor.Point.((scale .* node)...), r; action=:fill)
    end
end

begin
    s = 1000
    N = 3.8 * s
    Drawing(N, N, joinpath(@__DIR__, "../docs/src/assets/logo.svg"))
    origin()
    draw_logo(colors, nodes, s, 0.06 * s)
    finish()
    preview()
end
