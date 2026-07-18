#!/usr/bin/env julia
#
# Generate documentation images for WhatsThePoint.jl
#
# Usage:
#   julia --project=docs docs/generate_images.jl
#
# Renders are produced manually (never in CI) and committed to the repo.
# All static PNGs use transparent backgrounds so they read on both light and
# dark pages; the pipeline GIF uses white (GIF alpha is 1-bit and dithers).

import WhatsThePoint as WTP
using CairoMakie
using LinearAlgebra: dot, norm, normalize
using StaticArrays: SVector
using Unitful: m, ustrip

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
const PUBLIC_DIR = joinpath(@__DIR__, "src", "public")
const STL = joinpath(ASSETS_DIR, "bunny.stl")
const BIFURCATION_STL = joinpath(dirname(@__DIR__), "test", "data", "bifurcation.stl")
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
const SPACING_GRAD = cgrad([BOUNDARY_RED, JULIA_PURPLE, VOLUME_BLUE])  # fine wall → coarse bulk
const LABEL_GRAY = "#888888"     # reads on both GitHub light and dark
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

# 12 unique cube edges as Point3f pairs for one linesegments! call per box set.
# Corners are indexed by bits (x=1, y=2, z=4); each edge joins corners differing
# in exactly one bit, emitted once via the a < a|bit guard.
function box_wireframe_segments(lo, hi)
    corner(b) = Point3f(
        b & 1 == 0 ? lo[1] : hi[1],
        b & 2 == 0 ? lo[2] : hi[2],
        b & 4 == 0 ? lo[3] : hi[3],
    )
    segs = Point3f[]
    for a in 0:7, bit in (1, 2, 4)
        b = a | bit
        a < b && push!(segs, corner(a), corner(b))
    end
    return segs
end

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
# Shared scene: Poisson-disk sampled boundary + Octree/Bridson volume fill
# ============================================================================

function build_scene(; h = 1.5)
    mesh = WTP.import_mesh(STL, m)
    spacing = WTP.ConstantSpacing(h * m)
    boundary = WTP.PointBoundary(mesh, spacing)
    octree = WTP.TriangleOctree(mesh; classify_leaves = true)
    cloud = WTP.discretize(boundary, spacing; alg = WTP.Octree(octree))

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
# Bifurcation scene: graded Poisson-disk surface + Octree/Bridson volume fill
# ============================================================================

function bifurcation_scene(; at_wall = 0.0006, bulk = 0.0025, layer_thickness = 0.004)
    mesh = WTP.import_mesh(BIFURCATION_STL, m)
    octree = WTP.TriangleOctree(mesh; classify_leaves = true)
    spacing = WTP.BoundaryLayerSpacing(
        WTP.points(WTP.PointBoundary(mesh));
        at_wall = at_wall * m, bulk = bulk * m, layer_thickness = layer_thickness * m,
    )
    boundary = WTP.PointBoundary(mesh, spacing)
    alg = WTP.Octree(mesh; spacing, alpha = 1.0, max_growth = 0.15)
    cloud = WTP.discretize(boundary, spacing; alg)
    println("bifurcation scene: $(length(boundary)) boundary + $(length(WTP.volume(cloud))) volume points")
    return (; mesh, octree, spacing, boundary, cloud, at_wall, bulk)
end

# Cross-section slab colored by the prescribed spacing field h(x). The vessel
# is thinnest along z, so a mid-z slab shows the wall-to-bulk grading of both
# branches at once — a 3D cutaway would bury the interior behind the shell.
function generate_bifurcation_spacing(bif)
    println("Generating bifurcation spacing figure...")
    vpts = WTP.points(WTP.volume(bif.cloud))
    vx, vy, vz = coords_xyz(WTP.volume(bif.cloud))
    zmid = (minimum(vz) + maximum(vz)) / 2
    half = 0.75 * bif.bulk
    keep = findall(z -> abs(z - zmid) <= half, vz)
    h_mm = [1000 * ustrip(m, bif.spacing(vpts[i])) for i in keep]

    bx, by, bz = coords_xyz(bif.boundary)
    bkeep = findall(z -> abs(z - zmid) <= half, bz)

    limits_mm = (1000 * bif.at_wall, 1000 * bif.bulk)
    fig = Figure(; size = (1500, 800), backgroundcolor = :transparent)
    ax = bare_ax2(fig[1, 1])
    scatter!(ax, bx[bkeep], by[bkeep]; color = (CONTEXT_GRAY, 0.9), markersize = 4)
    scatter!(
        ax, vx[keep], vy[keep];
        color = h_mm, colormap = SPACING_GRAD, colorrange = limits_mm, markersize = 5,
    )
    Colorbar(
        fig[1, 2]; colormap = SPACING_GRAD, limits = limits_mm, width = 18,
        label = "h(x)  [mm]", labelcolor = LABEL_GRAY,
        ticklabelcolor = LABEL_GRAY, tickcolor = LABEL_GRAY, spinewidth = 0,
    )
    save_atomic(joinpath(ASSETS_DIR, "bifurcation-spacing.png"), fig; px_per_unit = 1)
    return println("  Saved bifurcation-spacing.png ($(length(keep)) slab points)")
end

# Drawing every leaf is unreadable (~28k boundary leaves alone), so only the
# leaves straddling the mid-z plane are shown, viewed top-down along z so the
# slice reads like an adaptive quadtree: fine boxes hugging the wall, coarser
# ones in the lumen. Exterior leaves are excluded; they extend to the cubic
# root box and would dwarf the vessel. Surface context is limited to the
# near-plane ring — the full shell would bury the boxes.
function generate_octree_leaves(bif)
    println("Generating octree leaf wireframe figure...")
    tree = bif.octree.tree
    bx, by, bz = coords_xyz(bif.boundary)
    zmid = (minimum(bz) + maximum(bz)) / 2
    segs = Point3f[]
    nleaves = 0
    for i in WTP.all_leaves(tree)
        cls = bif.octree.leaf_classification[i]
        (cls == WTP.LEAF_BOUNDARY || cls == WTP.LEAF_INTERIOR) || continue
        lo, hi = WTP.box_bounds(tree, i)
        (lo[3] < zmid < hi[3]) || continue
        append!(segs, box_wireframe_segments(lo, hi))
        nleaves += 1
    end
    wall = findall(z -> abs(z - zmid) < 2 * bif.at_wall, bz)
    println("  drawing $nleaves plane-straddling leaves, $(length(wall)) wall points")

    fig = Figure(; size = (1500, 800), backgroundcolor = :transparent)
    ax = bare_ax3(fig[1, 1]; azimuth = -π / 2, elevation = π / 2 - 0.001)
    scatter!(ax, bx[wall], by[wall], bz[wall]; color = BOUNDARY_RED, markersize = 6)
    linesegments!(ax, segs; color = (JULIA_PURPLE, 0.55), linewidth = 1.3)
    save_atomic(joinpath(ASSETS_DIR, "octree-leaves.png"), fig; px_per_unit = 1)
    return println("  Saved octree-leaves.png")
end

# A regular grid of query points on the mid-z plane classified by the
# octree-accelerated isinside — a lattice makes the inside/outside frontier
# crisp where random samples read as noise.
function generate_isinside_classification(bif)
    println("Generating isinside classification figure...")
    bx, by, bz = coords_xyz(bif.boundary)
    zmid = (minimum(bz) + maximum(bz)) / 2
    margin = 0.003
    step = 0.0005
    xs = range(minimum(bx) - margin, maximum(bx) + margin; step)
    ys = range(minimum(by) - margin, maximum(by) + margin; step)
    # the octree's machine type follows the STL parse (Float32); isinside
    # dispatches on matching scalar types
    query = [SVector{3, Float32}(x, y, zmid) for x in xs for y in ys]
    inside = WTP.isinside(query, bif.octree)
    println("  $(count(inside)) / $(length(query)) grid points inside")

    qx = getindex.(query, 1)
    qy = getindex.(query, 2)
    wall = findall(z -> abs(z - zmid) < 0.001, bz)

    fig = Figure(; size = (1300, 750), backgroundcolor = :transparent)
    ax = bare_ax2(fig[1, 1])
    scatter!(ax, qx[.!inside], qy[.!inside]; color = (CONTEXT_GRAY, 0.5), markersize = 3.5)
    scatter!(ax, qx[inside], qy[inside]; color = VOLUME_BLUE, markersize = 6)
    scatter!(ax, bx[wall], by[wall]; color = BOUNDARY_RED, markersize = 5)
    save_atomic(joinpath(ASSETS_DIR, "isinside-classification.png"), fig; px_per_unit = 1)
    return println("  Saved isinside-classification.png")
end

# ============================================================================
# Hero images
# ============================================================================

function generate_hero(scene)
    fig = Figure(; size = (1100, 1100), backgroundcolor = :transparent)
    cutaway_panel!(fig[1, 1], scene)
    save_atomic(joinpath(PUBLIC_DIR, "hero.png"), fig; px_per_unit = 0.6)
    return println("  Saved public/hero.png")
end

function generate_hero_banner(scene)
    fig = Figure(; size = (2800, 1000), backgroundcolor = :transparent)
    boundary_panel!(fig[1, 1], scene)
    cutaway_panel!(fig[1, 3], scene)
    stencil_panel!(fig[1, 5], scene)
    for col in (2, 4)
        Label(fig[1, col], "→"; fontsize = 110, color = CONTEXT_GRAY, tellheight = false)
    end
    captions = ("1 · sample the surface", "2 · fill the volume", "3 · connect stencils")
    for (col, text) in zip((1, 3, 5), captions)
        Label(fig[2, col], text; fontsize = 46, color = LABEL_GRAY, tellwidth = false)
    end
    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 4)
    save_atomic(joinpath(ASSETS_DIR, "hero-banner.png"), fig; px_per_unit = 0.5)
    return println("  Saved hero-banner.png")
end

function generate_bunny_pair(scene)
    fig = Figure(; size = (900, 900), backgroundcolor = :transparent)
    boundary_panel!(fig[1, 1], scene)
    save_atomic(joinpath(ASSETS_DIR, "bunny-boundary.png"), fig; px_per_unit = 1)

    fig2 = Figure(; size = (900, 900), backgroundcolor = :transparent)
    cutaway_panel!(fig2[1, 1], scene)
    save_atomic(joinpath(ASSETS_DIR, "bunny-discretized.png"), fig2; px_per_unit = 1)
    return println("  Saved bunny-boundary.png, bunny-discretized.png")
end

# ============================================================================
# Pipeline build-up GIF: sample → fill → relax → connect
# ============================================================================

# Static camera keeps frame-to-frame diffs local, which is what lets the
# palette-quantized GIF stay well under the 5 MB README budget despite ~70
# frames of animation.
function generate_pipeline_gif(; h = 2.0, framerate = 12)
    println("Generating pipeline build-up GIF...")
    # Own scene at a coarser h than build_scene: fewer points keep the
    # ~70-frame palette-quantized GIF legible and under the README size budget.
    mesh = WTP.import_mesh(STL, m)
    spacing = WTP.ConstantSpacing(h * m)
    boundary = WTP.PointBoundary(mesh, spacing)
    octree = WTP.TriangleOctree(mesh; classify_leaves = true)
    cloud = WTP.discretize(boundary, spacing; alg = WTP.Octree(octree))
    println("  gif scene: $(length(boundary)) boundary + $(length(WTP.volume(cloud))) volume points")
    bx, by, bz = coords_xyz(boundary)
    bshade = lambert_shades(surface_normals(boundary))
    vx, vy, vz = coords_xyz(WTP.volume(cloud))

    inset = 0.8 * h
    keep = findall(i -> !(bx[i] > 0 && by[i] < 0), eachindex(bx))
    wedge = findall(i -> (vx[i] > inset && vy[i] < -inset), eachindex(vx))
    ns, nw = length(keep), length(wedge)
    @assert nw > 0 "empty cutaway wedge — check inset against spacing"

    shellP = Point3f.(bx[keep], by[keep], bz[keep])
    wedge0 = Point3f.(vx[wedge], vy[wedge], vz[wedge])
    shellC = [RGBAf(get(RED_GRAD, s)) for s in bshade[keep]]
    wdepth = view_depth(vx[wedge], vy[wedge], vz[wedge])
    dlo, dhi = extrema(wdepth)
    wedgeC = [RGBAf(get(BLUE_GRAD, (d - dlo) / (dhi - dlo))) for d in wdepth]

    all0 = vcat(shellP, wedge0)
    ord = sortperm(view_depth(getindex.(all0, 1), getindex.(all0, 2), getindex.(all0, 3)))
    assemble(shellQ, wedgeQ) = vcat(shellQ, wedgeQ)[ord]

    # Relax stage: chunked octree-projected repel calls (no snapshot hook
    # exists). The volume-only method's point-sampled Green's-function survivor
    # filter mass-culls near-wall points, so the mesh-projected variant is used
    # instead: escapees bounce back and identity is preserved. Boundary points
    # slide on the surface, so shell positions are tracked per chunk too.
    # Quality stops are disabled so every chunk runs exactly max_iters.
    nvol = length(WTP.volume(cloud))
    nbnd = length(boundary)
    c = cloud
    shell_chunks = Vector{Point3f}[]
    wedge_chunks = Vector{Point3f}[]
    prev = wedge0
    for chunk in 1:8
        c = WTP.repel(
            c, spacing, octree;
            max_iters = 15, stall_after = 0, cv_target = 0.0,
        )
        if length(WTP.volume(c)) != nvol || length(WTP.boundary(c)) != nbnd
            @warn "repel changed point counts; relax stage ends at chunk $(chunk - 1)"
            break
        end
        cbx, cby, cbz = coords_xyz(WTP.boundary(c))
        cvx, cvy, cvz = coords_xyz(WTP.volume(c))
        push!(shell_chunks, Point3f.(cbx[keep], cby[keep], cbz[keep]))
        cur = Point3f.(cvx[wedge], cvy[wedge], cvz[wedge])
        disp = sum(norm(a - b) for (a, b) in zip(cur, prev)) / nw / h
        println("  chunk $chunk: mean wedge displacement $(round(disp; digits = 4)) h")
        push!(wedge_chunks, cur)
        prev = cur
    end
    nchunks = length(wedge_chunks)

    # Stencil stage: connectivity on the relaxed cloud; centers are chosen so
    # every neighbor is a displayed point (wedge volume or kept shell).
    ctop = WTP.set_topology(c, WTP.KNNTopology, 21)
    nb = length(WTP.boundary(ctop))
    keepset = BitSet(keep)
    wedgeset = BitSet(wedge)
    finalS = nchunks == 0 ? shellP : shell_chunks[end]
    finalW = nchunks == 0 ? wedge0 : wedge_chunks[end]
    shellpos = Dict(zip(keep, finalS))
    wedgepos = Dict(zip(wedge, finalW))
    displayed(g) = g <= nb ? g in keepset : (g - nb) in wedgeset
    candidates = [k for k in wedge if all(displayed, WTP.neighbors(ctop, nb + k))]
    if isempty(candidates)
        @warn "no fully-displayed stencil found; spokes may end at hidden points"
        candidates = wedge
    end

    fxlo, fxhi = extrema(getindex.(finalW, 1))
    fylo, fyhi = extrema(getindex.(finalW, 2))
    fzlo, fzhi = extrema(getindex.(finalW, 3))
    centers = Int[]
    for t in ((0.35, 0.3, 0.3), (0.65, 0.45, 0.55), (0.4, 0.35, 0.8))
        tp = Point3f(
            fxlo + t[1] * (fxhi - fxlo),
            fylo + t[2] * (fyhi - fylo),
            fzlo + t[3] * (fzhi - fzlo),
        )
        free = [k for k in candidates if !(k in centers)]
        isempty(free) && break
        push!(centers, free[argmin([sum(abs2, wedgepos[k] - tp) for k in free])])
    end

    segs = Point3f[]
    for k in centers, g in WTP.neighbors(ctop, nb + k)
        p = g <= nb ? get(shellpos, g, nothing) : get(wedgepos, g - nb, nothing)
        p === nothing && continue
        push!(segs, wedgepos[k], p)
    end

    # Storyboard @ 12 fps: surface sweep → hold → fill in generation order →
    # relax (chunk endpoint + midpoint tween) → stencil fade-in + hold.
    n_surface = 14
    n_fill = 22
    n_fade = 6
    fill_start = n_surface + 2 + 1
    relax_start = fill_start + n_fill
    relax_end = relax_start + 2 * nchunks - 1
    fade_start = relax_end + 1
    total = relax_end + n_fade + 10

    zlo, zhi = extrema(bz[keep])
    shell_rf = [clamp(ceil(Int, (z - zlo) / (zhi - zlo) * n_surface), 1, n_surface) for z in bz[keep]]
    wedge_rf = [fill_start - 1 + ceil(Int, r / nw * n_fill) for r in 1:nw]
    rf = vcat(shell_rf, wedge_rf)[ord]
    fullsize = vcat(fill(Float32(0.37 * h), ns), fill(Float32(0.33 * h), nw))[ord]
    sizes_at(f) = Float32[(rf[i] > f ? 0.0f0 : rf[i] == f ? 0.5f0 : 1.0f0) * fullsize[i] for i in eachindex(rf)]

    function positions_at(f)
        (f < relax_start || nchunks == 0) && return assemble(shellP, wedge0)
        f > relax_end && return assemble(finalS, finalW)
        step = f - relax_start + 1
        ch = cld(step, 2)
        sa = ch == 1 ? shellP : shell_chunks[ch - 1]
        wa = ch == 1 ? wedge0 : wedge_chunks[ch - 1]
        sb = shell_chunks[ch]
        wb = wedge_chunks[ch]
        return isodd(step) ?
            assemble(0.5f0 .* (sa .+ sb), 0.5f0 .* (wa .+ wb)) :
            assemble(sb, wb)
    end

    fig = Figure(; size = (640, 640), backgroundcolor = :white)
    ax = bare_ax3(fig[1, 1])
    pos = Observable(positions_at(1))
    siz = Observable(sizes_at(0))
    meshscatter!(ax, pos; markersize = siz, color = vcat(shellC, wedgeC)[ord])
    stencil_col = Observable((JULIA_PURPLE, 0.0))
    isempty(segs) || linesegments!(ax, segs; color = stencil_col, linewidth = 4)
    center_col = Observable((JULIA_PURPLE, 0.0))
    isempty(centers) || meshscatter!(
        ax, [wedgepos[k] for k in centers];
        markersize = 0.7 * h, color = center_col,
    )

    raw = joinpath(ASSETS_DIR, "pipeline-raw.gif")
    path = joinpath(ASSETS_DIR, "pipeline.gif")
    record(fig, raw, 1:total; framerate) do f
        pos[] = positions_at(f)
        siz[] = sizes_at(f)
        if f >= fade_start
            alpha = min((f - fade_start + 1) / n_fade, 1.0)
            stencil_col[] = (JULIA_PURPLE, alpha)
            center_col[] = (JULIA_PURPLE, alpha)
        end
    end

    # Makie's raw GIF is large; palette quantization + downscale brings it
    # under the 5 MB README budget without visible quality loss.
    vf = "fps=12,scale=520:-1:flags=lanczos,split[a][b];" *
        "[a]palettegen=max_colors=96:stats_mode=diff[p];" *
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    run(`$(Makie.FFMPEG_jll.ffmpeg()) -y -loglevel error -i $raw -vf $vf $path`)
    rm(raw)
    size_mb = round(filesize(path) / 1024^2; digits = 2)
    return println("  Saved pipeline.gif ($size_mb MB, $total frames)")
end

# ============================================================================
# Normal orientation before/after (Hoppe MST+DFS)
# ============================================================================

function normals_panel!(figpos, x, y, z, origins, dirs, colors)
    ax = bare_ax3(figpos)
    meshscatter!(ax, x, y, z; markersize = 0.7, color = (CONTEXT_GRAY, 0.6))
    arrows3d!(ax, origins, dirs; color = colors)
    return ax
end

# PCA gives each normal an axis but an arbitrary sign — the "before" panel
# colors the arrows that MST+DFS orientation will flip.
function generate_normals_orientation(; k = 10, n_arrows = 300, shaft = 2.5)
    println("Generating normals orientation figure...")
    mesh = WTP.import_mesh(STL, m)
    bnd = WTP.PointBoundary(mesh, WTP.ConstantSpacing(4.0m))
    pts = WTP.points(bnd)
    n_before = WTP.compute_normals(pts; k)
    n_after = copy(n_before)
    WTP.orient_normals!(n_after, pts; k)

    x, y, z = coords_xyz(bnd)
    sel = 1:cld(length(pts), n_arrows):length(pts)
    flipped = [dot(n_before[i], n_after[i]) < 0 for i in sel]
    println("  $(length(pts)) points, $(count(flipped)) / $(length(sel)) shown arrows flipped")
    origins = Point3f.(x[sel], y[sel], z[sel])
    before_dirs = [Vec3f((shaft * n_before[i])...) for i in sel]
    after_dirs = [Vec3f((shaft * n_after[i])...) for i in sel]
    before_cols = [f ? BOUNDARY_RED : "#8a8a8a" for f in flipped]

    fig = Figure(; size = (1600, 900), backgroundcolor = :transparent)
    normals_panel!(fig[1, 1], x, y, z, origins, before_dirs, before_cols)
    normals_panel!(fig[1, 2], x, y, z, origins, after_dirs, fill(JULIA_PURPLE, length(sel)))
    save_atomic(joinpath(ASSETS_DIR, "normals-orientation.png"), fig; px_per_unit = 1)
    return println("  Saved normals-orientation.png")
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
    save_atomic(joinpath(ASSETS_DIR, "2d-discretization.png"), fig; px_per_unit = 1)
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
    save_atomic(joinpath(ASSETS_DIR, "repel-comparison.png"), fig; px_per_unit = 1)
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
    save_atomic(joinpath(ASSETS_DIR, "algorithm-comparison.png"), fig; px_per_unit = 1)
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
    generate_pipeline_gif()
    generate_normals_orientation()
    bif = bifurcation_scene()
    generate_bifurcation_spacing(bif)
    generate_octree_leaves(bif)
    generate_isinside_classification(bif)
    generate_2d_discretization()
    generate_repel_comparison()
    generate_algorithm_comparison()
    return println("Done! Images saved to $ASSETS_DIR and $PUBLIC_DIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
