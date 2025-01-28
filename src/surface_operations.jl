"""
    add_surface!(boundary::PointBoundary, points::Vector{<:Point}

Add a surface to an existing boundary. Creates a new surface, unless a name of an existing
surface is given and it is added to that.

"""
function add_surface!(boundary::PointBoundary, points::Vector{<:Point}, name::Symbol)
    haskey(surfaces(boundary), name) && throw(ArgumentError("surface name already exists."))
    normals = compute_normals(points)
    areas = zeros(length(points)) * Unitful.m^2
    surfaces(boundary)[name] = PointBoundary(points, normals, areas)
    return nothing
end

function combine_surfaces!(boundary::PointBoundary, surfs...)
    for surf in surfs
        @assert hassurface(boundary, surf) "Surface does not exist. Check spelling."
    end

    combined_surfaces = filter(surf -> surf ∈ surfs, WhatsThePoint.names(boundary))

    # add new combined surface
    new_geoms = mapreduce(vcat, combined_surfaces) do name
        boundary[name].geoms
    end
    # delete old surface of new combined name
    for name in surfs
        delete!(boundary, name)
    end
    boundary[first(surfs)] = PointSurface(new_geoms, nothing)

    return nothing
end

function split_surface!(cloud::Union{PointCloud,PointBoundary}, angle::Real; k::Int=5)
    @assert length(surfaces(cloud)) == 1 "More than 1 surface in this cloud. Please specify a target surface."
    target_surf = only(names(boundary(cloud)))
    return split_surface!(cloud, target_surf, angle; k=k)
end

function split_surface!(
    cloud::Union{PointCloud,PointBoundary}, target_surf::Symbol, angle::Real; k::Int=5
)
    @assert hassurface(cloud, target_surf) "Target surface not found in cloud."
    angle = deg2rad(angle) # convert to radians
    surf = cloud[target_surf]
    delete!(boundary(cloud).surfaces, target_surf)
    return split_surface!(cloud, surf, angle; k=k)
end

function split_surface!(
    cloud::Union{PointCloud,PointBoundary}, surf::PointSurface, angle::Real; k::Int=5
)
    points = point(surf)
    normals = normal(surf)
    areas = area(surf)

    neighbors = search(surf, KNearestSearch(surf, k))

    g = SimpleGraph(length(surf))
    for n in neighbors, v in n[2:end]
        i = first(n)
        abs(_angle(parent(surf).normal[i], parent(surf).normal[v])) < angle &&
            add_edge!(g, i, v)
    end

    connec = connected_components(g)
    ranges = ranges_from_permutation(connec)

    many_permute!(points, connec, ranges)
    many_permute!(normals, connec, ranges)
    many_permute!(areas, connec, ranges)

    for (i, ids) in enumerate(ranges)
        name = _generate_surface_name(cloud, i)
        cloud[name] = PointSurface(points[ids], normals[ids], areas[ids])
    end

    return cloud

    # TODO orient normals again because you will not include points around sharp edges where
    # it is another surface, improving the normal estimation
end

function _generate_surface_name(cloud, i::Int)
    new_name = Symbol("surface" * string(i))
    return if !hassurface(boundary(cloud), new_name)
        new_name
    else
        _generate_surface_name(cloud, i + 1)
    end
end
