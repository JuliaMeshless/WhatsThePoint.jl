"""
    add_surface!(part::PointPart, points::Vector{<:Point}

Add a surface to an existing part. Creates a new surface, unless a name of an existing
surface is given and it is added to that.

"""
function add_surface!(part::PointPart, points::Vector{<:Point}, name::Symbol)
    haskey(part.surfaces, name) && throw(ArgumentError("surface name already exists."))
    N_add = length(points)
    append!(part.points, points)
    N = length(part.points)
    part.surfaces[name] = PointSurface(part.points, (N - N_add + 1):N)
    return nothing
end

# TODO - did i finsih this? test it
function combine_surfaces!(part::PointPart, surfs...)
    for surf in surfs
        @assert hassurface(part, surf) "Surface does not exist. Check spelling."
    end

    ids = [only(part[surf].points.indices) for surf in surfs]
    combined_ids = 1:sum(length, ids)
    separate_surfs = filter(surf -> surf ∉ surfs, PointClouds.names(part))

    old_ids = []
    new_ids = [combined_ids]
    for surf in separate_surfs
        push!(old_ids, only(part[surf].points.indices))
        push!(new_ids, last(new_ids[end]) .+ (1:length(old_ids[end])))
    end
    deleteat!(new_ids, 1)
    new_order = mapfoldl(collect, vcat, vcat(ids, old_ids))
    permute!(part.points, new_order)

    fold_field(field) = mapfoldl(surf -> getfield(part[surf], field), vcat, surfs)
    combined_normals = fold_field(:normals)
    combined_areas = fold_field(:areas)

    # delete old surface of new combined name
    for name in surfs
        delete!(part.surfaces, name)
    end
    # add new combined surface
    part[first(surfs)] = PointSurface(
        part.points, combined_normals, combined_areas, combined_ids
    )

    # reconstruct surfaces which were not combined
    for (ids, surf) in zip(new_ids, separate_surfs)
        n = deepcopy(normals(part[surf]))
        a = deepcopy(areas(part[surf]))
        delete!(part.surfaces, surf)
        part[surf] = PointSurface(part.points, n, a, ids)
    end

    return nothing
end

function split_surface!(cloud::Union{PointCloud,PointPart}, angle::Real; k::Int=5)
    @assert length(surfaces(cloud)) == 1 "More than 1 surface in this cloud. Please specify a target surface."
    target_surf = only(keys(cloud.surfaces))
    return split_surface!(cloud, target_surf, angle; k=k)
end

function split_surface!(
    cloud::Union{PointCloud,PointPart}, target_surf::Symbol, angle::Real; k::Int=5
)
    @assert hassurface(cloud, target_surf) "Target surface not found in cloud."
    angle = deg2rad(angle) # convert to radians
    surf = cloud[target_surf]
    delete!(cloud.surfaces, target_surf)
    return split_surface!(cloud, surf, angle; k=k)
end

function split_surface!(
    cloud::Union{PointCloud,PointPart}, surf::PointSurface, angle::Real; k::Int=5
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
        cloud[name] = PointSurface(view(cloud, ids), normals[ids], areas[ids])
    end

    return cloud

    # TODO orient normals again because you will not include points around sharp edges where
    # it is another surface, improving the normal estimation
end

function _generate_surface_name(cloud, i::Int)
    new_name = Symbol("surface" * string(i))
    return if !haskey(cloud.surfaces, new_name)
        new_name
    else
        _generate_surface_name(cloud, i + 1)
    end
end
