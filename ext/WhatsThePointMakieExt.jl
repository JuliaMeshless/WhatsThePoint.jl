module WhatsThePointMakieExt

using WhatsThePoint
using WhatsThePoint: surfaces, names, volume, to
using WhatsThePoint: CenterlineResult, Centerline, CenterlinePoint, EmbeddedVoronoi
using WhatsThePoint: positions, radii, arc_length, min_radius, max_radius
using Meshes: 𝔼, lentype
using Unitful: ustrip
using StaticArrays: SVector

using Makie: Makie, Figure, Axis3, meshscatter!, lines!, scatter!

function WhatsThePoint.visualize(
    cloud::PointCloud{𝔼{3},C}; size=(1000, 1000), azimuth=1.275π, elevation=π / 8, kwargs...
) where {C}
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    N = length(cloud)
    Ns = sum(length, surfaces(cloud))
    surfs = surfaces(cloud)
    vol = volume(cloud)

    T = lentype(C)
    Ncolors = length(surfs)
    if Ncolors > 1
        labels = zeros(Int, N)
        x = zeros(T, N)
        y = zeros(T, N)
        z = zeros(T, N)
        prev = 0
        # plot surfaces
        for (i, (n, s)) in enumerate(zip(names(cloud), values(surfs)))
            next = length(s)
            ids = (1:next) .+ (prev)
            prev = last(ids)
            labels[ids] .= i
            coords = to(s)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            z[ids] = getindex.(coords, 3)
            println("$i -> $n")
        end

        # volume
        if !isempty(vol)
            ids = (1:length(vol)) .+ Ns
            coords = to(vol)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            z[ids] = getindex.(coords, 3)
            Ncolors += 1
            labels[ids] .= Ncolors
            println("$Ncolors -> volume")
        end

        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y),
            ustrip.(z);
            color=labels,
            colormap=:Spectral,
            kwargs...,
        )
        Makie.Colorbar(
            fig[1, 2];
            limits=(1, Ncolors),
            colormap=Makie.cgrad(:Spectral, Ncolors; categorical=true),
        )
    else
        labels = ones(Int, N)
        Ns = 1

        x = zeros(T, N)
        y = zeros(T, N)
        z = zeros(T, N)

        surf = only(collect(surfaces(cloud)))
        ids = 1:length(surf)
        coords = to(surf)
        x[ids] = getindex.(coords, 1)
        y[ids] = getindex.(coords, 2)
        z[ids] = getindex.(coords, 3)
        println("1 -> $(only(names(cloud)))")

        # volume
        if !isempty(vol)
            ids = (1:length(vol)) .+ length(surf)
            labels[ids] .= 2
            coords = to(vol)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            z[ids] = getindex.(coords, 3)
            println("2 -> volume")
            Ns += 1
        end

        if Ns > 1
            meshscatter!(
                ax,
                ustrip.(x),
                ustrip.(y),
                ustrip.(z);
                color=labels,
                colormap=:Spectral,
                kwargs...,
            )
            Makie.Colorbar(
                fig[1, 2];
                limits=(1, Ns),
                colormap=Makie.cgrad(:Spectral, Ns; categorical=true),
            )
        else
            meshscatter!(ax, ustrip.(x), ustrip.(y), ustrip.(z); kwargs...)
        end
    end

    return fig
end

function WhatsThePoint.visualize(
    cloud::PointBoundary{𝔼{3},C};
    size=(1000, 1000),
    azimuth=1.275π,
    elevation=π / 8,
    kwargs...,
) where {C}
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    N = length(cloud)
    surfs = surfaces(cloud)

    T = lentype(C)
    Ns = length(surfs)
    if Ns > 1
        labels = zeros(Int, N)
        x = zeros(T, N)
        y = zeros(T, N)
        z = zeros(T, N)
        prev = 0
        # plot surfaces
        for (i, (n, s)) in enumerate(zip(names(cloud), values(surfs)))
            next = length(s)
            ids = (1:next) .+ (prev)
            prev = last(ids)
            labels[ids] .= i
            coords = to(s)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            z[ids] = getindex.(coords, 3)
            println("$i -> $n")
        end

        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y),
            ustrip.(z);
            color=labels,
            colormap=:Spectral,
            kwargs...,
        )
        Makie.Colorbar(
            fig[1, 2]; limits=(1, Ns), colormap=Makie.cgrad(:Spectral, Ns; categorical=true)
        )
    else
        labels = ones(Int, N)
        Ns = 1

        x = zeros(T, N)
        y = zeros(T, N)
        z = zeros(T, N)

        surf = only(collect(surfaces(cloud)))
        ids = 1:length(surf)
        coords = to(surf)
        x[ids] = getindex.(coords, 1)
        y[ids] = getindex.(coords, 2)
        z[ids] = getindex.(coords, 3)
        println("1 -> $(only(names(cloud)))")

        if Ns > 1
            meshscatter!(
                ax,
                ustrip.(x),
                ustrip.(y),
                ustrip.(z);
                color=labels,
                colormap=:Spectral,
                kwargs...,
            )
            Makie.Colorbar(
                fig[1, 2];
                limits=(1, Ns),
                colormap=Makie.cgrad(:Spectral, Ns; categorical=true),
            )
        else
            meshscatter!(ax, ustrip.(x), ustrip.(y), ustrip.(z); kwargs...)
        end
    end

    return fig
end

# ============================================================================
# Centerline Visualization
# ============================================================================

"""
    visualize(result::CenterlineResult; kwargs...)

Visualize centerline extraction results, showing all centerlines colored by radius.

# Keyword Arguments
- `size=(1000, 1000)` - Figure size
- `azimuth=1.275π` - Camera azimuth angle
- `elevation=π/8` - Camera elevation angle
- `linewidth=4` - Line width for centerlines
- `colormap=:viridis` - Colormap for radius visualization
- `show_voronoi=false` - Whether to show Voronoi vertices
- `voronoi_alpha=0.1` - Alpha for Voronoi vertices
"""
function WhatsThePoint.visualize(
    result::CenterlineResult;
    size=(1000, 1000),
    azimuth=1.275π,
    elevation=π / 8,
    linewidth=4,
    colormap=:viridis,
    show_voronoi=false,
    voronoi_alpha=0.1,
    kwargs...
)
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    # Compute global radius range for consistent coloring
    all_radii = Float64[]
    for cl in result.centerlines
        append!(all_radii, radii(cl))
    end

    if isempty(all_radii)
        @warn "No centerlines to visualize"
        return fig
    end

    rmin, rmax = extrema(all_radii)

    # Show Voronoi vertices if requested
    if show_voronoi
        verts = result.voronoi.vertices
        vx = [v.center[1] for v in verts]
        vy = [v.center[2] for v in verts]
        vz = [v.center[3] for v in verts]
        scatter!(ax, vx, vy, vz; color=(:gray, voronoi_alpha), markersize=2)
    end

    # Plot each centerline
    for (i, cl) in enumerate(result.centerlines)
        pos = positions(cl)
        r = radii(cl)

        x = [p[1] for p in pos]
        y = [p[2] for p in pos]
        z = [p[3] for p in pos]

        # Color by radius
        lines!(ax, x, y, z;
            color=r,
            colorrange=(rmin, rmax),
            colormap=colormap,
            linewidth=linewidth,
            label="Centerline $i"
        )
    end

    # Add colorbar
    Makie.Colorbar(fig[1, 2];
        limits=(rmin, rmax),
        colormap=colormap,
        label="Radius"
    )

    return fig
end

"""
    visualize(cl::Centerline; kwargs...)

Visualize a single centerline colored by radius.
"""
function WhatsThePoint.visualize(
    cl::Centerline;
    size=(1000, 1000),
    azimuth=1.275π,
    elevation=π / 8,
    linewidth=4,
    colormap=:viridis,
    kwargs...
)
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    pos = positions(cl)
    r = radii(cl)

    if isempty(pos)
        @warn "Empty centerline"
        return fig
    end

    x = [p[1] for p in pos]
    y = [p[2] for p in pos]
    z = [p[3] for p in pos]

    rmin, rmax = extrema(r)

    lines!(ax, x, y, z;
        color=r,
        colorrange=(rmin, rmax),
        colormap=colormap,
        linewidth=linewidth
    )

    Makie.Colorbar(fig[1, 2];
        limits=(rmin, rmax),
        colormap=colormap,
        label="Radius"
    )

    return fig
end

"""
    visualize(voronoi::EmbeddedVoronoi; kwargs...)

Visualize the embedded Voronoi diagram (for debugging).
"""
function WhatsThePoint.visualize(
    voronoi::EmbeddedVoronoi;
    size=(1000, 1000),
    azimuth=1.275π,
    elevation=π / 8,
    colormap=:viridis,
    markersize=3,
    kwargs...
)
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    verts = voronoi.vertices

    if isempty(verts)
        @warn "No Voronoi vertices to visualize"
        return fig
    end

    x = [v.center[1] for v in verts]
    y = [v.center[2] for v in verts]
    z = [v.center[3] for v in verts]
    r = [v.radius for v in verts]

    rmin, rmax = extrema(r)

    meshscatter!(ax, x, y, z;
        color=r,
        colorrange=(rmin, rmax),
        colormap=colormap,
        markersize=markersize,
        kwargs...
    )

    Makie.Colorbar(fig[1, 2];
        limits=(rmin, rmax),
        colormap=colormap,
        label="Inscribed Sphere Radius"
    )

    return fig
end

end # module
