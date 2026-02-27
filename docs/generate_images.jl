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
# 1. 2D polygon discretization (for Quick Start page)
# ============================================================================

function generate_2d_discretization()
    println("Generating 2D polygon discretization...")

    pts = WTP.Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = WTP.PointBoundary(pts)
    spacing = WTP.ConstantSpacing(0.05m)
    cloud = WTP.discretize(boundary, spacing; alg=WTP.FornbergFlyer())

    # Extract coordinates
    bnd_coords = WTP.to(WTP.boundary(cloud))
    bnd_x = [ustrip(c[1]) for c in bnd_coords]
    bnd_y = [ustrip(c[2]) for c in bnd_coords]

    vol_coords = WTP.to(WTP.volume(cloud))
    vol_x = [ustrip(c[1]) for c in vol_coords]
    vol_y = [ustrip(c[2]) for c in vol_coords]

    fig = Figure(; size=(600, 600))
    ax = Axis(fig[1, 1]; aspect=DataAspect(),
              xlabel="x", ylabel="y",
              title="2D Discretization (FornbergFlyer)")

    scatter!(ax, vol_x, vol_y; color=:steelblue, markersize=6, label="Volume")
    scatter!(ax, bnd_x, bnd_y; color=:red, markersize=8, label="Boundary")
    axislegend(ax; position=:rt)

    CairoMakie.save(joinpath(ASSETS_DIR, "2d-discretization.png"), fig; px_per_unit=2)
    println("  Saved 2d-discretization.png ($(length(cloud)) points)")
end

# ============================================================================
# 2. Repulsion before/after (for Node Repulsion page)
# ============================================================================

function generate_repel_comparison()
    println("Generating repulsion before/after comparison...")

    pts = WTP.Point.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    boundary = WTP.PointBoundary(pts)
    spacing = WTP.ConstantSpacing(0.05m)
    cloud_before = WTP.discretize(boundary, spacing; alg=WTP.FornbergFlyer())

    cloud_after, convergence = WTP.repel(cloud_before, spacing; max_iters=500)

    fig = Figure(; size=(1200, 550))

    # Before repulsion
    ax1 = Axis(fig[1, 1]; aspect=DataAspect(),
               xlabel="x", ylabel="y",
               title="Before Repulsion")

    bnd_coords = WTP.to(WTP.boundary(cloud_before))
    bnd_x = [ustrip(c[1]) for c in bnd_coords]
    bnd_y = [ustrip(c[2]) for c in bnd_coords]
    vol_coords = WTP.to(WTP.volume(cloud_before))
    vol_x = [ustrip(c[1]) for c in vol_coords]
    vol_y = [ustrip(c[2]) for c in vol_coords]

    scatter!(ax1, vol_x, vol_y; color=:steelblue, markersize=5)
    scatter!(ax1, bnd_x, bnd_y; color=:red, markersize=7)

    # After repulsion
    ax2 = Axis(fig[1, 2]; aspect=DataAspect(),
               xlabel="x", ylabel="y",
               title="After Repulsion")

    bnd_coords2 = WTP.to(WTP.boundary(cloud_after))
    bnd_x2 = [ustrip(c[1]) for c in bnd_coords2]
    bnd_y2 = [ustrip(c[2]) for c in bnd_coords2]
    vol_coords2 = WTP.to(WTP.volume(cloud_after))
    vol_x2 = [ustrip(c[1]) for c in vol_coords2]
    vol_y2 = [ustrip(c[2]) for c in vol_coords2]

    scatter!(ax2, vol_x2, vol_y2; color=:steelblue, markersize=5)
    scatter!(ax2, bnd_x2, bnd_y2; color=:red, markersize=7)

    CairoMakie.save(joinpath(ASSETS_DIR, "repel-comparison.png"), fig; px_per_unit=2)
    println("  Saved repel-comparison.png")
    println("  Before: $(length(cloud_before)) points, After: $(length(cloud_after)) points")
    println("  Convergence: $(convergence[end]) after $(length(convergence)) iterations")
end

# ============================================================================
# 3. Algorithm comparison — 3D (SlakKosec vs VanDerSandeFornberg)
# ============================================================================

function generate_algorithm_comparison()
    println("Generating 3D algorithm comparison...")

    boundary = WTP.PointBoundary(joinpath(@__DIR__, "src", "assets", "bunny.stl"))
    spacing = WTP.ConstantSpacing(1m)

    cloud_sk = WTP.discretize(boundary, spacing; alg=WTP.SlakKosec(), max_points=50_000)
    cloud_vf = WTP.discretize(boundary, spacing; alg=WTP.VanDerSandeFornberg(), max_points=50_000)

    fig = Figure(; size=(1200, 550))

    for (idx, (cloud, title)) in enumerate([
        (cloud_sk, "SlakKosec ($(length(cloud_sk)) pts)"),
        (cloud_vf, "VanDerSandeFornberg ($(length(cloud_vf)) pts)"),
    ])
        ax = Axis3(fig[1, idx]; azimuth=1.275π, elevation=π / 8, title=title)
        ax.aspect = :data

        vol = WTP.volume(cloud)
        if !isempty(vol)
            coords = WTP.to(vol)
            x = ustrip.([c[1] for c in coords])
            y = ustrip.([c[2] for c in coords])
            z = ustrip.([c[3] for c in coords])
            meshscatter!(ax, x, y, z; markersize=0.15, color=:steelblue)
        end
    end

    CairoMakie.save(joinpath(ASSETS_DIR, "algorithm-comparison.png"), fig; px_per_unit=2)
    println("  Saved algorithm-comparison.png")
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
