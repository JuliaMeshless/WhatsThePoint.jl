# bench_isinside.jl — TriangleOctree construction + isinside query throughput.
# Run before/after the pseudonormal change:  julia --project=. -t 1 bench_isinside.jl
using Pkg
Pkg.activate(@__DIR__)
using WhatsThePoint, Meshes, GeoIO
using StaticArrays: SVector
using LinearAlgebra: norm
using Random, Printf
using Unitful: ustrip

# Corrected (outward-winding) annular cavity, same generator as
# validate_cavity.jl / shape_opt_tradeoff.jl.
function _make_sphere_mesh(R, nθ, nφ)
    pts = [Meshes.Point(R * sin(π * i / nθ) * cos(2π * j / nφ),
                        R * sin(π * i / nθ) * sin(2π * j / nφ),
                        R * cos(π * i / nθ))
           for j in 0:(nφ - 1), i in 0:nθ][:]
    conn = Connectivity{Triangle}[]
    for i in 0:(nθ - 1), j in 0:(nφ - 1)
        a = i * nφ + j + 1
        b = i * nφ + (j + 1) % nφ + 1
        c = (i + 1) * nφ + j + 1
        d = (i + 1) * nφ + (j + 1) % nφ + 1
        i > 0 && push!(conn, connect((a, c, b)))
        i < nθ - 1 && push!(conn, connect((b, c, d)))
    end
    return SimpleMesh(pts, conn)
end

function _make_cavity_mesh(r_inner)
    nθ, nφ = 24, 48
    outer = _make_sphere_mesh(1.0, nθ, nφ)
    inner = _make_sphere_mesh(r_inner, nθ, nφ)
    n_outer = length(collect(Meshes.vertices(outer)))
    all_v = vcat(collect(Meshes.vertices(outer)), collect(Meshes.vertices(inner)))
    all_conn = Connectivity{Triangle}[]
    for c in Meshes.topology(outer)
        push!(all_conn, c)
    end
    for c in Meshes.topology(inner)
        pts = Meshes.indices(c)
        push!(all_conn, connect((pts[1] + n_outer, pts[3] + n_outer, pts[2] + n_outer)))
    end
    return SimpleMesh(all_v, all_conn)
end

mesh = _make_cavity_mesh(0.547)

function bench(label, m; n_uniform=200_000, n_shell=200_000, shellgen)
    # warm
    TriangleOctree(m; classify_leaves=true)
    tb = minimum((@elapsed TriangleOctree(m; classify_leaves=true)) for _ in 1:3)
    o = TriangleOctree(m; classify_leaves=true)
    lo, hi = o.mesh_bbox_min, o.mesh_bbox_max
    rng = MersenneTwister(11)
    uni = [lo .+ rand(rng, SVector{3,Float64}) .* (hi .- lo) for _ in 1:n_uniform]
    shell = [shellgen(rng) for _ in 1:n_shell]
    # warm
    foreach(q -> isinside(q, o), uni[1:1000]); foreach(q -> isinside(q, o), shell[1:1000])
    tu = @elapsed (s = 0; for q in uni; s += isinside(q, o); end; s)
    ts = @elapsed (s = 0; for q in shell; s += isinside(q, o); end; s)
    nin_u = count(q -> isinside(q, o), uni)
    nin_s = count(q -> isinside(q, o), shell)
    @printf("%-8s build %.2fs   uniform %.0f ns/q (in %.1f%%)   near-surface %.0f ns/q (in %.1f%%)\n",
        label, tb, 1e9 * tu / n_uniform, 100nin_u / n_uniform,
        1e9 * ts / n_shell, 100nin_s / n_shell)
    return nothing
end

# cavity: near-surface = radii around both shells
function cavity_shell(rng)
    d = randn(rng, SVector{3,Float64}); d /= norm(d)
    R = rand(rng) < 0.5 ? 0.547 : 1.0
    return (R + 0.03 * (2rand(rng) - 1)) * d
end
bench("cavity", mesh; shellgen=cavity_shell)

bunny_raw = GeoIO.load(joinpath(@__DIR__, "bunny.stl")).geometry
bunny = SimpleMesh(
    [Meshes.Point((1.0 .* Meshes.to(v))...) for v in Meshes.vertices(bunny_raw)],
    Meshes.topology(bunny_raw),
)
println("bunny: ", Meshes.nelements(bunny), " facets")
bverts = [Float64.(ustrip.(Meshes.to(v))) for v in Meshes.vertices(bunny)]
bb_lo = reduce((a, b) -> min.(a, b), bverts); bb_hi = reduce((a, b) -> max.(a, b), bverts)
scale = norm(bb_hi - bb_lo)
function bunny_shell(rng)
    v = bverts[rand(rng, 1:length(bverts))]
    return SVector{3,Float64}(v...) .+ 0.005 * scale .* randn(rng, SVector{3,Float64})
end
bench("bunny", bunny; shellgen=bunny_shell)
