"""
    combine_surfaces!(boundary::PointBoundary, surfs...)

Merge multiple named surfaces into one. The first name is kept and subsequent surfaces are
merged into it. All original surfaces are removed and replaced by the combined surface.
"""
function combine_surfaces!(boundary::PointBoundary, surfs...)
    for surf in surfs
        @assert hassurface(boundary, surf) "Surface does not exist. Check spelling."
    end

    combined_surfaces = filter(surf -> surf ∈ surfs, WhatsThePoint.names(boundary))

    # add new combined surface
    new_geoms = mapreduce(vcat, combined_surfaces) do name
        boundary[name].geoms
    end
    new_normals = mapreduce(vcat, combined_surfaces) do name
        normal(boundary[name])
    end
    # delete old surface of new combined name
    for name in surfs
        delete!(boundary, name)
    end
    boundary[first(surfs)] = PointSurface(new_geoms)
    return nothing
end

"""
    split_surface!(cloud, angle; k=10)
    split_surface!(cloud, target_surf, angle; k=10)

Split a surface into sub-surfaces based on normal angle discontinuities. Builds a k-nearest
neighbor graph, removes edges where adjacent normals differ by more than `angle`, and labels
each connected component as a separate named surface.

When called on a cloud/boundary with a single surface, that surface is split automatically.
When multiple surfaces exist, specify `target_surf` by name.
"""
function split_surface!(cloud::Union{PointCloud, PointBoundary}, angle::Angle; k::Int = 10)
    @assert length(namedsurfaces(cloud)) == 1 "More than 1 surface in this cloud. Please specify a target surface."
    target_surf = only(names(boundary(cloud)))
    return split_surface!(cloud, target_surf, angle; k = k)
end

function split_surface!(
        cloud::Union{PointCloud, PointBoundary},
        target_surf::Symbol,
        angle::Angle;
        k::Int = 10,
    )
    @assert hassurface(cloud, target_surf) "Target surface not found in cloud."
    surf = cloud[target_surf]
    delete!(namedsurfaces(boundary(cloud)), target_surf)
    return split_surface!(cloud, surf, angle; k = k)
end

function split_surface!(
        cloud::Union{PointCloud, PointBoundary},
        surf::PointSurface,
        angle::Angle;
        k::Int = 10,
    )
    points = point(surf)
    normals = normal(surf)
    areas = area(surf)

    neighbors = search(surf, KNearestSearch(surf, k))

    g = SimpleGraph(length(surf))
    for n in neighbors
        i = first(n)
        for v in n[2:end]
            abs(_angle(normals[i], normals[v])) < angle && add_edge!(g, i, v)
        end
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
