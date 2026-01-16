module WhatsThePointMakieExt

using WhatsThePoint
using WhatsThePoint: surfaces, names, volume, to
using Meshes: 𝔼, lentype
using Unitful: ustrip

using Makie: Makie, Figure, Axis3, meshscatter!

function WhatsThePoint.visualize(
        cloud::PointCloud{𝔼{3}, C}; size = (1000, 1000), azimuth = 1.275π, elevation = π / 8, kwargs...
    ) where {C}
    fig = Figure(; size = size)
    ax = Axis3(fig[1, 1]; azimuth = azimuth, elevation = elevation)
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
            color = labels,
            colormap = :Spectral,
            kwargs...,
        )
        Makie.Colorbar(
            fig[1, 2];
            limits = (1, Ncolors),
            colormap = Makie.cgrad(:Spectral, Ncolors; categorical = true),
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
                color = labels,
                colormap = :Spectral,
                kwargs...,
            )
            Makie.Colorbar(
                fig[1, 2];
                limits = (1, Ns),
                colormap = Makie.cgrad(:Spectral, Ns; categorical = true),
            )
        else
            meshscatter!(ax, ustrip.(x), ustrip.(y), ustrip.(z); kwargs...)
        end
    end

    return fig
end

function WhatsThePoint.visualize(
        cloud::PointBoundary{𝔼{3}, C};
        size = (1000, 1000),
        azimuth = 1.275π,
        elevation = π / 8,
        kwargs...,
    ) where {C}
    fig = Figure(; size = size)
    ax = Axis3(fig[1, 1]; azimuth = azimuth, elevation = elevation)
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
            color = labels,
            colormap = :Spectral,
            kwargs...,
        )
        Makie.Colorbar(
            fig[1, 2]; limits = (1, Ns), colormap = Makie.cgrad(:Spectral, Ns; categorical = true)
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
                color = labels,
                colormap = :Spectral,
                kwargs...,
            )
            Makie.Colorbar(
                fig[1, 2];
                limits = (1, Ns),
                colormap = Makie.cgrad(:Spectral, Ns; categorical = true),
            )
        else
            meshscatter!(ax, ustrip.(x), ustrip.(y), ustrip.(z); kwargs...)
        end
    end

    return fig
end

end # module
