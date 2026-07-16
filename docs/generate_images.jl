#!/usr/bin/env julia
#
# Generate documentation images for WhatsThePoint.jl
#
# Usage:
#   julia --project=docs docs/generate_images.jl
#
# Renders are produced manually (never in CI) and committed to the repo.
# All static PNGs use transparent backgrounds so they read on both light and
# dark pages; the turntable GIF uses white (GIF alpha is 1-bit and dithers).

import WhatsThePoint as WTP
using CairoMakie
using LinearAlgebra: dot, normalize
using Unitful: m, ustrip

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
const PUBLIC_DIR = joinpath(@__DIR__, "src", "public")
const STL = joinpath(ASSETS_DIR, "bunny.stl")
mkpath(ASSETS_DIR)
mkpath(PUBLIC_DIR)

# ============================================================================
# Shared style
# ============================================================================

const BOUNDARY_RED = "#CB3C33"   # Julia red
const VOLUME_BLUE = "#4F7CAC"    # slate blue
const JULIA_PURPLE = "#9558B2"
const CONTEXT_GRAY = "#c4c4c4"
const RED_GRAD = cgrad(["#8a2620", BOUNDARY_RED, "#f0897e"])
const BLUE_GRAD = cgrad(["#3b6187", VOLUME_BLUE, "#7ea6cd"])
const LIGHT_DIR = normalize([-0.4, -0.55, 0.73])
const AZIMUTH = 1.275π
const ELEVATION = π / 8
const CAMDIR = [
    cos(ELEVATION) * cos(AZIMUTH),
    cos(ELEVATION) * sin(AZIMUTH),
    sin(ELEVATION),
]

# iCloud (which hosts this repo) can stall `close` on large files written
# incrementally, killing the render at the very end. Encode to a local temp
# file first, then move the finished bytes into place, retrying the move.
function save_atomic(path, fig; px_per_unit)
    tmp = joinpath(mktempdir(), basename(path))
    CairoMakie.save(tmp, fig; px_per_unit)
    for attempt in 1:3
        try
            mv(tmp, path; force = true)
            return
        catch e
            attempt == 3 && rethrow(e)
            @warn "move to $path failed (attempt $attempt), retrying" exception = e
            sleep(5)
        end
    end
    return
end

function coords_xyz(obj)
    cs = WTP.to(obj)
    x = ustrip.(getindex.(cs, 1))
    y = ustrip.(getindex.(cs, 2))
    z = ustrip.(getindex.(cs, 3))
    return x, y, z
end

surface_normals(boundary) = first(values(WTP.surfaces(boundary))).geoms.normal

# Baked Lambert shading: point clouds rendered as flat-colored spheres read as
# 2D; shading each point by its surface normal restores depth.
lambert_shades(normals) = [clamp(dot(n, LIGHT_DIR), 0, 1) for n in normals]

view_depth(x, y, z) = @. x * CAMDIR[1] + y * CAMDIR[2] + z * CAMDIR[3]

function bare_ax3(pos; kwargs...)
    ax = Axis3(
        pos; azimuth = AZIMUTH, elevation = ELEVATION, aspect = :data,
        viewmode = :fitzoom, protrusions = 0,
        xypanelvisible = false, xzpanelvisible = false, yzpanelvisible = false,
        kwargs...,
    )
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

function bare_ax2(pos; kwargs...)
    ax = Axis(pos; aspect = DataAspect(), backgroundcolor = :transparent, kwargs...)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

# CairoMakie composites separate plot objects in creation order, so mixed
# boundary/volume scenes must go through a single meshscatter sorted
# back-to-front along the camera direction.
function depth_sorted_meshscatter!(ax, x, y, z, colors, sizes)
    ord = sortperm(view_depth(x, y, z))
    return meshscatter!(ax, x[ord], y[ord], z[ord]; markersize = sizes[ord], color = colors[ord])
end

# ============================================================================
# Shared scene: Poisson-disk sampled boundary + SlakKosec volume fill
# ============================================================================

function build_scene(; h = 1.5, max_points = 500_000)
    mesh = WTP.import_mesh(STL, m)
    spacing = WTP.ConstantSpacing(h * m)
    boundary = WTP.PointBoundary(mesh, spacing)
    octree = WTP.TriangleOctree(mesh; classify_leaves = true)
    cloud = WTP.discretize(boundary, spacing; alg = WTP.SlakKosec(octree), max_points)

    bx, by, bz = coords_xyz(boundary)
    bnormals = surface_normals(boundary)
    bshade = lambert_shades(bnormals)
    vx, vy, vz = coords_xyz(WTP.volume(cloud))
    println("scene: $(length(boundary)) boundary + $(length(WTP.volume(cloud))) volume points")
    return (; bx, by, bz, bnormals, bshade, vx, vy, vz)
end

boundary_colors(scene) = [RGBAf(get(RED_GRAD, s)) for s in scene.bshade]

# Cutaway: remove the shell quarter {x > 0, y < 0} facing the camera and show
# the volume fill inside it. The wedge is inset by ~one spacing so interior
# points don't speckle through the pores of the remaining shell.
function cutaway_arrays(scene; inset = 1.2)
    (; bx, by, bz, bshade, vx, vy, vz) = scene
    keep = findall(i -> !(bx[i] > 0 && by[i] < 0), eachindex(bx))
    wedge = findall(i -> (vx[i] > inset && vy[i] < -inset), eachindex(vx))

    bcol = [RGBAf(get(RED_GRAD, s)) for s in bshade[keep]]
    vdepth = view_depth(vx[wedge], vy[wedge], vz[wedge])
    lo, hi = extrema(vdepth)
    vcol = [RGBAf(get(BLUE_GRAD, (d - lo) / (hi - lo))) for d in vdepth]

    x = vcat(bx[keep], vx[wedge])
    y = vcat(by[keep], vy[wedge])
    z = vcat(bz[keep], vz[wedge])
    colors = vcat(bcol, vcol)
    sizes = vcat(fill(0.55, length(keep)), fill(0.5, length(wedge)))
    return x, y, z, colors, sizes
end

function boundary_panel!(figpos, scene)
    ax = bare_ax3(figpos)
    depth_sorted_meshscatter!(
        ax, scene.bx, scene.by, scene.bz,
        boundary_colors(scene), fill(0.55, length(scene.bx)),
    )
    return ax
end

function cutaway_panel!(figpos, scene)
    ax = bare_ax3(figpos)
    depth_sorted_meshscatter!(ax, cutaway_arrays(scene)...)
    return ax
end

# Stencil close-up: one interior point, its k-nearest neighbors, and the
# surrounding context points.
function stencil_panel!(figpos, scene; target = (15.0, -5.0, 28.0), k = 20, radius = 4.8)
    (; vx, vy, vz) = scene
    d2t = @. (vx - target[1])^2 + (vy - target[2])^2 + (vz - target[3])^2
    ci = argmin(d2t)
    c = (vx[ci], vy[ci], vz[ci])

    d2c = @. (vx - c[1])^2 + (vy - c[2])^2 + (vz - c[3])^2
    ball = findall(<=(radius^2), d2c)
    ball = ball[sortperm(d2c[ball])]
    neigh = ball[2:(k + 1)]

    # clear the sight line so context points don't bury the stencil
    chat = CAMDIR ./ sqrt(sum(abs2, CAMDIR))
    function occludes(i)
        d = [vx[i] - c[1], vy[i] - c[2], vz[i] - c[3]]
        t = dot(d, chat)
        perp = sqrt(max(sum(abs2, d) - t^2, 0.0))
        return t > 0.3 && perp < 3.4
    end
    context = [i for i in ball[(k + 2):end] if !occludes(i)]

    ax = bare_ax3(figpos)
    meshscatter!(ax, vx[context], vy[context], vz[context]; markersize = 0.24, color = (CONTEXT_GRAY, 0.5))
    meshscatter!(ax, vx[neigh], vy[neigh], vz[neigh]; markersize = 0.42, color = VOLUME_BLUE)
    meshscatter!(ax, [c[1]], [c[2]], [c[3]]; markersize = 0.62, color = JULIA_PURPLE)
    segments = Point3f[]
    for j in neigh
        push!(segments, Point3f(c...), Point3f(vx[j], vy[j], vz[j]))
    end
    linesegments!(ax, segments; color = JULIA_PURPLE, linewidth = 3.5)
    return ax
end

# ============================================================================
# Hero images
# ============================================================================

function generate_hero(scene)
    fig = Figure(; size = (1100, 1100), backgroundcolor = :transparent)
    cutaway_panel!(fig[1, 1], scene)
    save_atomic(joinpath(PUBLIC_DIR, "hero.png"), fig; px_per_unit = 2)
    return println("  Saved public/hero.png")
end

function generate_hero_banner(scene)
    fig = Figure(; size = (2800, 900), backgroundcolor = :transparent)
    boundary_panel!(fig[1, 1], scene)
    cutaway_panel!(fig[1, 2], scene)
    stencil_panel!(fig[1, 3], scene)
    colgap!(fig.layout, 40)
    save_atomic(joinpath(ASSETS_DIR, "hero-banner.png"), fig; px_per_unit = 1)
    return println("  Saved hero-banner.png")
end

function generate_bunny_pair(scene)
    fig = Figure(; size = (900, 900), backgroundcolor = :transparent)
    boundary_panel!(fig[1, 1], scene)
    save_atomic(joinpath(ASSETS_DIR, "bunny-boundary.png"), fig; px_per_unit = 2)

    fig2 = Figure(; size = (900, 900), backgroundcolor = :transparent)
    cutaway_panel!(fig2[1, 1], scene)
    save_atomic(joinpath(ASSETS_DIR, "bunny-discretized.png"), fig2; px_per_unit = 2)
    return println("  Saved bunny-boundary.png, bunny-discretized.png")
end

# ============================================================================
# Turntable GIF
# ============================================================================

function generate_turntable(scene; nframes = 45, framerate = 15)
    fig = Figure(; size = (640, 640), backgroundcolor = :white)
    ax = bare_ax3(fig[1, 1])
    plt = meshscatter!(
        ax, scene.bx, scene.by, scene.bz;
        markersize = 0.55, color = scene.bshade, colormap = RED_GRAD,
    )

    raw = joinpath(ASSETS_DIR, "turntable-raw.gif")
    path = joinpath(ASSETS_DIR, "turntable.gif")
    azimuths = range(AZIMUTH, AZIMUTH + 2π; length = nframes + 1)[1:(end - 1)]
    record(fig, raw, azimuths; framerate) do az
        # rotate the light with the camera so the visible side stays lit
        Δ = az - AZIMUTH
        rot = [cos(Δ) -sin(Δ) 0; sin(Δ) cos(Δ) 0; 0 0 1]
        light = rot * LIGHT_DIR
        plt.color = [clamp(dot(n, light), 0, 1) for n in scene.bnormals]
        ax.azimuth = az
    end

    # Makie's raw GIF is ~16 MB; palette quantization + downscale brings it
    # under the 5 MB README budget without visible quality loss.
    vf = "fps=12,scale=520:-1:flags=lanczos,split[a][b];" *
        "[a]palettegen=max_colors=96:stats_mode=diff[p];" *
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    run(`$(Makie.FFMPEG_jll.ffmpeg()) -y -loglevel error -i $raw -vf $vf $path`)
    rm(raw)
    size_mb = round(filesize(path) / 1024^2; digits = 2)
    return println("  Saved turntable.gif ($(size_mb) MB)")
end

# ============================================================================
# 2D Stanford Bunny silhouette (projected from 3D bunny.stl)
# ============================================================================

function bunny_silhouette(; n_bins = 200)
    boundary3d = WTP.PointBoundary(STL, m)
    coords = WTP.to(boundary3d)

    # Project to XZ plane (side view)
    xs = [ustrip(c[1]) for c in coords]
    zs = [ustrip(c[3]) for c in coords]

    cx = sum(xs) / length(xs)
    cz = sum(zs) / length(zs)

    # Angular binning — take outermost point per bin for silhouette
    angles = atan.(zs .- cz, xs .- cx)
    dists = @. sqrt((xs - cx)^2 + (zs - cz)^2)

    bin_edges = range(-π, π; length = n_bins + 1)
    outline_x = Float64[]
    outline_z = Float64[]

    for i in 1:n_bins
        mask = @. (bin_edges[i] <= angles) & (angles < bin_edges[i + 1])
        if any(mask)
            idx = findall(mask)
            best = idx[argmax(dists[idx])]
            push!(outline_x, xs[best])
            push!(outline_z, zs[best])
        end
    end

    pts = WTP.Point.(collect(zip(outline_x, outline_z)))
    return WTP.PointBoundary(pts)
end

function silhouette_cloud()
    boundary = bunny_silhouette()
    coords = WTP.to(boundary)
    xs = [ustrip(c[1]) for c in coords]
    dx = maximum(xs) - minimum(xs)
    spacing = WTP.ConstantSpacing((dx / 60) * m)
    cloud = WTP.discretize(boundary, spacing; alg = WTP.FornbergFlyer())
    return cloud, spacing
end

function scatter_cloud2d!(figpos, cloud; volume_size = 5, boundary_size = 7)
    ax = bare_ax2(figpos)
    vc = WTP.to(WTP.volume(cloud))
    bc = WTP.to(WTP.boundary(cloud))
    scatter!(ax, [ustrip(p[1]) for p in vc], [ustrip(p[2]) for p in vc]; color = VOLUME_BLUE, markersize = volume_size)
    scatter!(ax, [ustrip(p[1]) for p in bc], [ustrip(p[2]) for p in bc]; color = BOUNDARY_RED, markersize = boundary_size)
    return ax
end

# ============================================================================
# 2D discretization (Quick Start page)
# ============================================================================

function generate_2d_discretization()
    println("Generating 2D Stanford Bunny discretization...")
    cloud, _ = silhouette_cloud()
    fig = Figure(; size = (700, 640), backgroundcolor = :transparent)
    scatter_cloud2d!(fig[1, 1], cloud)
    save_atomic(joinpath(ASSETS_DIR, "2d-discretization.png"), fig; px_per_unit = 2)
    return println("  Saved 2d-discretization.png ($(length(cloud)) points)")
end

# ============================================================================
# Repulsion before/after (Node Repulsion page)
# ============================================================================

function generate_repel_comparison()
    println("Generating repulsion before/after comparison...")
    cloud_before, spacing = silhouette_cloud()
    conv = Float64[]
    cloud_after = WTP.repel(cloud_before, spacing; max_iters = 500, convergence = conv)

    fig = Figure(; size = (1400, 640), backgroundcolor = :transparent)
    scatter_cloud2d!(fig[1, 1], cloud_before; volume_size = 4, boundary_size = 6)
    scatter_cloud2d!(fig[1, 2], cloud_after; volume_size = 4, boundary_size = 6)
    save_atomic(joinpath(ASSETS_DIR, "repel-comparison.png"), fig; px_per_unit = 2)
    println("  Saved repel-comparison.png")
    return println("  Convergence: $(conv[end]) after $(length(conv)) iterations")
end

# ============================================================================
# Algorithm comparison — 3D (SlakKosec vs VanDerSandeFornberg)
# ============================================================================

function generate_algorithm_comparison()
    println("Generating 3D algorithm comparison...")
    boundary = WTP.PointBoundary(STL, m)
    spacing = WTP.ConstantSpacing(3m)

    fig = Figure(; size = (1400, 640), backgroundcolor = :transparent)
    for (idx, alg) in enumerate([WTP.SlakKosec(), WTP.VanDerSandeFornberg()])
        cloud = WTP.discretize(boundary, spacing; alg, max_points = 400_000)
        vx, vy, vz = coords_xyz(WTP.volume(cloud))
        vdepth = view_depth(vx, vy, vz)
        lo, hi = extrema(vdepth)
        vcol = [RGBAf(get(BLUE_GRAD, (d - lo) / (hi - lo))) for d in vdepth]
        ax = bare_ax3(fig[1, idx])
        depth_sorted_meshscatter!(ax, vx, vy, vz, vcol, fill(1.1, length(vx)))
        println("  $(nameof(typeof(alg))): $(length(cloud)) points")
    end
    save_atomic(joinpath(ASSETS_DIR, "algorithm-comparison.png"), fig; px_per_unit = 2)
    return println("  Saved algorithm-comparison.png")
end

# ============================================================================
# Run all
# ============================================================================

function main()
    println("Generating documentation images...")
    scene = build_scene()
    generate_hero(scene)
    generate_hero_banner(scene)
    generate_bunny_pair(scene)
    generate_turntable(scene)
    generate_2d_discretization()
    generate_repel_comparison()
    generate_algorithm_comparison()
    return println("Done! Images saved to $ASSETS_DIR and $PUBLIC_DIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
