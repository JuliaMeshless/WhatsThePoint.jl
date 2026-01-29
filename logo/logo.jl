using Pkg
Pkg.activate(@__DIR__)
using WhatsThePoint
using Unitful: m, ustrip
using Luxor
using Meshes

function get_xy(cloud)
    points = pointify(cloud)
    x = map(p -> ustrip(WhatsThePoint.coords(p).x), points)
    y = map(p -> ustrip(WhatsThePoint.coords(p).y), points)
    return x, y
end

function circle_nodes(origins)
    r = 0.75
    N = 12
    θ = 0:(π/N):(2π-π/(2N))
    s = 3 * r / N
    spacing = ConstantSpacing(s * m)

    points = WhatsThePoint.Point.([(r * cos(i), r * sin(i)) for i in θ]) .+ (origins,)
    boundary = PointBoundary(points)
    cloud = discretize(boundary, spacing; max_points = 200)

    α = s / 10
    for _ = 1:5
        repel!(cloud, spacing; α = α, β = 0.2, k = 12, max_iters = 10_000, tol = 1.0e-6)
        α /= 2
    end

    return get_xy(cloud)
end

Oy = 2cos(π / 6) - 1
origins = [Vec(-1, Oy), Vec(1, Oy), Vec(0, -1)] #.+ Ref(Vec(2.5, -2.5))
nodes = circle_nodes.(origins)

colors = (Luxor.julia_red, Luxor.julia_green, Luxor.julia_purple)

function draw_logo(colors, nodes, scale, r)
    for (color, node) in zip(colors, nodes)
        setcolor(color)
        circle.(Luxor.Point.((scale .* node)...), r; action = :fill)
    end
    return
end

begin
    s = 1000
    N = 3.8 * s
    for file_type in (:png, :svg)
        @info "Generating logo.$(file_type)..."
        Drawing(N, N, joinpath(@__DIR__, "../docs/src/assets/logo.$(file_type)"))
        origin()
        draw_logo(colors, nodes, s, 0.06 * s)
        finish()
        preview()
    end
end
