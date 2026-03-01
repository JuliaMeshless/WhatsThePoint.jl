#!/usr/bin/env julia
#
# Generate documentation images for WhatsThePoint.jl
#
# Usage:
#   julia --project=docs docs/generate_images.jl
#
# Requires: CairoMakie, Unitful (add to docs/Project.toml if not present)

import WhatsThePoint as WTP
using CairoMakie
using Unitful: m, mm, °, ustrip

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS_DIR)

# ============================================================================
# 2D Stanford Bunny silhouette (projected from 3D bunny.stl)
# ============================================================================

function bunny_silhouette(; n_bins = 200)
    stl_path = joinpath(@__DIR__, "src", "assets", "bunny.stl")
    boundary3d = WTP.PointBoundary(stl_path)
    coords = WTP.to(boundary3d)

    # Project to XZ plane (side view)
    xs = [ustrip(c[1]) for c in coords]
    zs = [ustrip(c[3]) for c in coords]

    # Centroid
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

# ============================================================================
# 1. 2D discretization — Stanford Bunny silhouette (for Quick Start page)
# ============================================================================

function generate_2d_discretization()
    println("Generating 2D Stanford Bunny discretization...")

    boundary = bunny_silhouette()
    # Compute spacing from bounding box (~1/60th of width)
    coords = WTP.to(boundary)
    xs = [ustrip(c[1]) for c in coords]
    dx = maximum(xs) - minimum(xs)
    spacing = WTP.ConstantSpacing((dx / 60) * m)
    cloud = WTP.discretize(boundary, spacing; alg = WTP.FornbergFlyer())

    # Extract coordinates
    bnd_coords = WTP.to(WTP.boundary(cloud))
    bnd_x = [ustrip(c[1]) for c in bnd_coords]
    bnd_y = [ustrip(c[2]) for c in bnd_coords]

    vol_coords = WTP.to(WTP.volume(cloud))
    vol_x = [ustrip(c[1]) for c in vol_coords]
    vol_y = [ustrip(c[2]) for c in vol_coords]

    fig = Figure(; size = (700, 600))
    ax = Axis(
        fig[1, 1]; aspect = DataAspect(),
        xlabel = "x", ylabel = "y",
        title = "2D Discretization — Stanford Bunny"
    )

    scatter!(ax, vol_x, vol_y; color = :steelblue, markersize = 4, label = "Volume")
    scatter!(ax, bnd_x, bnd_y; color = :red, markersize = 6, label = "Boundary")
    axislegend(ax; position = :rt)

    CairoMakie.save(joinpath(ASSETS_DIR, "2d-discretization.png"), fig; px_per_unit = 2)
    return println("  Saved 2d-discretization.png ($(length(cloud)) points)")
end

# ============================================================================
# 2. Repulsion before/after (for Node Repulsion page)
# ============================================================================

function generate_repel_comparison()
    println("Generating repulsion before/after comparison (Stanford Bunny)...")

    boundary = bunny_silhouette()
    coords = WTP.to(boundary)
    xs = [ustrip(c[1]) for c in coords]
    dx = maximum(xs) - minimum(xs)
    spacing = WTP.ConstantSpacing((dx / 60) * m)
    cloud_before = WTP.discretize(boundary, spacing; alg = WTP.FornbergFlyer())

    cloud_after, convergence = WTP.repel(cloud_before, spacing; max_iters = 500)

    fig = Figure(; size = (1400, 600))

    # Before repulsion
    ax1 = Axis(
        fig[1, 1]; aspect = DataAspect(),
        xlabel = "x", ylabel = "y",
        title = "Before Repulsion"
    )

    bnd_coords = WTP.to(WTP.boundary(cloud_before))
    bnd_x = [ustrip(c[1]) for c in bnd_coords]
    bnd_y = [ustrip(c[2]) for c in bnd_coords]
    vol_coords = WTP.to(WTP.volume(cloud_before))
    vol_x = [ustrip(c[1]) for c in vol_coords]
    vol_y = [ustrip(c[2]) for c in vol_coords]

    scatter!(ax1, vol_x, vol_y; color = :steelblue, markersize = 3)
    scatter!(ax1, bnd_x, bnd_y; color = :red, markersize = 5)

    # After repulsion
    ax2 = Axis(
        fig[1, 2]; aspect = DataAspect(),
        xlabel = "x", ylabel = "y",
        title = "After Repulsion"
    )

    bnd_coords2 = WTP.to(WTP.boundary(cloud_after))
    bnd_x2 = [ustrip(c[1]) for c in bnd_coords2]
    bnd_y2 = [ustrip(c[2]) for c in bnd_coords2]
    vol_coords2 = WTP.to(WTP.volume(cloud_after))
    vol_x2 = [ustrip(c[1]) for c in vol_coords2]
    vol_y2 = [ustrip(c[2]) for c in vol_coords2]

    scatter!(ax2, vol_x2, vol_y2; color = :steelblue, markersize = 3)
    scatter!(ax2, bnd_x2, bnd_y2; color = :red, markersize = 5)

    CairoMakie.save(joinpath(ASSETS_DIR, "repel-comparison.png"), fig; px_per_unit = 2)
    println("  Saved repel-comparison.png")
    println("  Before: $(length(cloud_before)) points, After: $(length(cloud_after)) points")
    return println("  Convergence: $(convergence[end]) after $(length(convergence)) iterations")
end

# ============================================================================
# 3. Algorithm comparison — 3D (SlakKosec vs VanDerSandeFornberg)
# ============================================================================

function generate_algorithm_comparison()
    println("Generating 3D algorithm comparison...")

    boundary = WTP.PointBoundary(joinpath(@__DIR__, "src", "assets", "bunny.stl"))
    spacing = WTP.ConstantSpacing(1m)

    cloud_sk = WTP.discretize(boundary, spacing; alg = WTP.SlakKosec(), max_points = 95_000)
    cloud_vf = WTP.discretize(boundary, spacing; alg = WTP.VanDerSandeFornberg(), max_points = 95_000)

    fig = Figure(; size = (1200, 550))

    for (idx, (cloud, title)) in enumerate(
            [
                (cloud_sk, "SlakKosec ($(length(cloud_sk)) pts)"),
                (cloud_vf, "VanDerSandeFornberg ($(length(cloud_vf)) pts)"),
            ]
        )
        ax = Axis3(fig[1, idx]; azimuth = 1.275π, elevation = π / 8, title = title)
        ax.aspect = :data

        vol = WTP.volume(cloud)
        if !isempty(vol)
            coords = WTP.to(vol)
            x = ustrip.([c[1] for c in coords])
            y = ustrip.([c[2] for c in coords])
            z = ustrip.([c[3] for c in coords])
            meshscatter!(ax, x, y, z; markersize = 0.15, color = :steelblue)
        end
    end

    CairoMakie.save(joinpath(ASSETS_DIR, "algorithm-comparison.png"), fig; px_per_unit = 2)
    return println("  Saved algorithm-comparison.png")
end

# ============================================================================
# Run all
# ============================================================================

println("Generating documentation images...")
println("Output directory: $ASSETS_DIR")
println()

generate_2d_discretization()
println()
generate_repel_comparison()
println()
generate_algorithm_comparison()

println()
println("Done! All images saved to $ASSETS_DIR")
