_get_colorrange(labels) = isnothing(labels) ? nothing : (minimum(labels), maximum(labels))

function visualize(
    cloud::PointCloud{𝔼{3},C}; size=(1000, 1000), azimuth=1.275π, elevation=π / 8, kwargs...
) where {C}
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    N = length(cloud)
    Ns = sum(length, surfaces(cloud))
    surfs = surfaces(cloud)
    vol = volume(cloud)

    T = Meshes.lentype(C)
    Ncolors = length(surfs)
    if Ncolors > 1
        labels = zeros(Int, N)
        x = zeros(T, N)
        y = zeros(T, N)
        z = zeros(T, N)
        prev = 0
        # plot surfaces
        for (i, (n, s)) in enumerate(zip(names(cloud), surfs))
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

function visualize(
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

    T = Meshes.lentype(C)
    Ns = length(surfs)
    if Ns > 1
        labels = zeros(Int, N)
        x = zeros(T, N)
        y = zeros(T, N)
        z = zeros(T, N)
        prev = 0
        # plot surfaces
        for (i, (n, s)) in enumerate(zip(names(cloud), surfs))
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

function visualize(
    cloud::Union{PointCloud,PointBoundary},
    labels;
    size=(1000, 1000),
    colorrange=_get_colorrange(labels),
    azimuth=1.275π,
    elevation=π / 8,
    kwargs...,
)
    return visualize(
        to(cloud),
        labels;
        size=size,
        colorrange=colorrange,
        azimuth=azimuth,
        elevation=elevation,
        kwargs...,
    )
end

function visualize(
    coords::AbstractVector,
    labels=nothing;
    size=(1000, 1000),
    colorrange=_get_colorrange(labels),
    azimuth=1.275π,
    elevation=π / 8,
    levels=32,
    colormap=:Spectral,
    kwargs...,
)
    N = embeddim(first(coords))
    return if N == 2
        _visualize2d(
            coords,
            labels;
            size=size,
            colorrange=colorrange,
            levels=levels,
            colormap=colormap,
            kwargs...,
        )
    elseif N == 3
        _visualize3d(
            coords,
            labels;
            size=size,
            colorrange=colorrange,
            azimuth=azimuth,
            elevation=elevation,
            levels=levels,
            colormap=colormap,
            kwargs...,
        )
    else
        error("No plotting methods for points when dim=$N")
    end
end

function _visualize2d(
    coords::AbstractVector,
    labels=nothing;
    size=(1000, 1000),
    colorrange=_get_colorrange(labels),
    levels=32,
    colormap=colormap,
    kwargs...,
)
    fig = Figure(; size=size)
    ax = Axis(fig[1, 1]; aspect=DataAspect())

    c = coords.(points)
    x = map(c -> ustrip(c.x), c)
    y = map(c -> ustrip(c.y), c)
    if !isnothing(labels)
        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y);
            color=labels,
            colorrange=colorrange,
            colormap=cmap,
            kwargs...,
        )
        Makie.Colorbar(fig[1, 2]; colorrange=colorrange, colormap=cmap)
    else
        meshscatter!(ax, ustrip.(x), ustrip.(y); kwargs...)
    end
    return fig
end

function _visualize3d(
    points::AbstractVector,
    labels=nothing;
    size=(1000, 1000),
    colorrange=_get_colorrange(labels),
    azimuth=1.275π,
    elevation=π / 8,
    levels=32,
    colormap=cmap,
    kwargs...,
)
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1]; azimuth=azimuth, elevation=elevation)
    ax.aspect = :data

    cmap = Makie.cgrad(colormap, levels; categorical=true)

    c = coords.(points)
    x = map(c -> ustrip(c.x), c)
    y = map(c -> ustrip(c.y), c)
    z = map(c -> ustrip(c.z), c)
    if !isnothing(labels)
        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y),
            ustrip.(z);
            color=labels,
            colorrange=colorrange,
            colormap=cmap,
            kwargs...,
        )
        Makie.Colorbar(fig[1, 2]; colorrange=colorrange, colormap=cmap)
    else
        meshscatter!(ax, ustrip.(x), ustrip.(y), ustrip.(z); kwargs...)
    end
    return fig
end

function visualize(cloud::PointBoundary{𝔼{2},C}; size=(1000, 1000), kwargs...) where {C}
    fig = Figure(; size=size)
    ax = Axis(fig[1, 1]; aspect=DataAspect())
    coords = to(cloud)

    N = length(cloud)
    surfs = surfaces(cloud)

    T = Meshes.lentype(C)
    Ns = length(surfs)
    if Ns > 1
        labels = zeros(Int, N)
        x = zeros(T, N)
        y = zeros(T, N)
        prev = 0
        # plot surfaces
        for (i, (n, s)) in enumerate(zip(names(cloud), surfs))
            next = length(s)
            ids = (1:next) .+ (prev)
            prev = last(ids)
            labels[ids] .= i
            coords = to(s)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            println("$i -> $n")
        end

        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y);
            color=labels,
            colormap=:Spectral,
            shading=Makie.NoShading,
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

        surf = only(collect(surfaces(cloud)))
        ids = 1:length(surf)
        coords = to(surf)
        x[ids] = getindex.(coords, 1)
        y[ids] = getindex.(coords, 2)
        println("1 -> $(only(names(cloud)))")

        if Ns > 1
            meshscatter!(
                ax,
                ustrip.(x),
                ustrip.(y);
                color=labels,
                colormap=:Spectral,
                shading=Makie.NoShading,
                kwargs...,
            )
            Makie.Colorbar(
                fig[1, 2];
                limits=(1, Ns),
                colormap=Makie.cgrad(:Spectral, Ns; categorical=true),
            )
        else
            meshscatter!(ax, ustrip.(x), ustrip.(y); shading=Makie.NoShading, kwargs...)
        end
    end

    return fig
end

function visualize(cloud::PointCloud{𝔼{2},C}; size=(1000, 1000), kwargs...) where {C}
    fig = Figure(; size=size)
    ax = Axis(fig[1, 1]; aspect=DataAspect())
    coords = to(cloud)

    N = length(cloud)
    Ns = sum(length, surfaces(cloud))
    surfs = surfaces(cloud)
    vol = volume(cloud)

    T = Meshes.lentype(C)
    Ncolors = length(surfs)
    if Ncolors > 1
        labels = zeros(Int, N)
        x = zeros(T, N)
        y = zeros(T, N)
        prev = 0
        # plot surfaces
        for (i, (n, s)) in enumerate(zip(names(cloud), surfs))
            next = length(s)
            ids = (1:next) .+ (prev)
            prev = last(ids)
            labels[ids] .= i
            coords = to(s)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            println("$i -> $n")
        end

        # volume
        if !isempty(vol)
            ids = (1:length(vol)) .+ Ns
            coords = to(vol)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            Ncolors += 1
            labels[ids] .= Ncolors
            println("$Ncolors -> volume")
        end

        meshscatter!(
            ax,
            ustrip.(x),
            ustrip.(y);
            color=labels,
            colormap=:Spectral,
            shading=Makie.NoShading,
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

        surf = only(collect(surfaces(cloud)))
        ids = 1:length(surf)
        coords = to(surf)
        x[ids] = getindex.(coords, 1)
        y[ids] = getindex.(coords, 2)
        println("1 -> $(only(collect(names(cloud))))")

        # volume
        if !isempty(vol)
            ids = (1:length(vol)) .+ length(surf)
            labels[ids] .= 2
            coords = to(vol)
            x[ids] = getindex.(coords, 1)
            y[ids] = getindex.(coords, 2)
            println("2 -> volume")
            Ns += 1
        end

        if Ns > 1
            meshscatter!(
                ax,
                ustrip.(x),
                ustrip.(y);
                color=labels,
                colormap=:Spectral,
                shading=Makie.NoShading,
                kwargs...,
            )
            Makie.Colorbar(
                fig[1, 2];
                limits=(1, Ns),
                colormap=Makie.cgrad(:Spectral, Ns; categorical=true),
            )
        else
            meshscatter!(ax, ustrip.(x), ustrip.(y); shading=Makie.NoShading, kwargs...)
        end
    end

    return fig
end

function visualize_normals(
    cloud::PointCloud{𝔼{3},C}; size=(1000, 1000), kwargs...
) where {C}
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1])
    ax.aspect = :data
    T = Meshes.lentype(C)

    surfs = surfaces(cloud)
    coords = to(cloud)
    x = getindex.(coords, 1)
    y = getindex.(coords, 2)
    z = getindex.(coords, 3)
    labels = zeros(Int, length(cloud))

    # plot normals
    N = mapreduce(length, +, surfs)
    xn = zeros(T, N)
    yn = zeros(T, N)
    zn = zeros(T, N)
    u = zeros(T, N)
    v = zeros(T, N)
    w = zeros(T, N)

    if length(surfs) > 1
        offset = 0
        for (i, s) in enumerate(surfs)
            ids = first(s.points.indices)
            range = (offset + 1):(offset + length(ids))
            labels[range] .= i
            xn[range] .= x[ids]
            yn[range] .= y[ids]
            zn[range] .= z[ids]
            u[range] .= getindex.(s.normals, 1)
            v[range] .= getindex.(s.normals, 2)
            w[range] .= getindex.(s.normals, 3)
            offset += length(ids)
        end
        arrows!(ax, xn, yn, zn, u, v, w; linecolor=labels, arrowcolor=labels, kwargs...)
    else
        s = only(surfs)
        u = getindex.(s.normals, 1)
        v = getindex.(s.normals, 2)
        w = getindex.(s.normals, 3)
        arrows!(ax, x, y, z, u, v, w; linecolor=labels, arrowcolor=labels, kwargs...)
    end

    return fig
end

function visualize(surf::PointSurface{𝔼{3},C}; size=(1000, 1000), kwargs...) where {C}
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1])
    ax.aspect = :data

    coords = to(surf)
    x = getindex.(coords, 1)
    y = getindex.(coords, 2)
    z = getindex.(coords, 3)
    labels = zeros(Int, length(surf))

    meshscatter!(ax, ustrip.(x), ustrip.(y), ustrip.(z); color=labels, kwargs...)

    return fig
end

function visualize_normals(surf::PointSurface{𝔼{3}}; size=(1000, 1000), kwargs...)
    fig = Figure(; size=size)
    ax = Axis3(fig[1, 1])
    ax.aspect = :data

    coords = to(surf)
    x = getindex.(coords, 1)
    y = getindex.(coords, 2)
    z = getindex.(coords, 3)

    u = getindex.(parent(surf).normal, 1)
    v = getindex.(parent(surf).normal, 2)
    w = getindex.(parent(surf).normal, 3)
    arrows!(ax, x, y, z, u, v, w; kwargs...)

    return fig
end
