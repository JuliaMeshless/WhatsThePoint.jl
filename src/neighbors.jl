function Meshes.KNearestSearch(
        cloud::Union{PointCloud, PointBoundary, PointSurface},
        k::Int;
        metric = Euclidean(),
    )
    return KNearestSearch(collect(points(cloud)), k; metric = metric)
end

function Meshes.search(
        cloud::Union{PointCloud, PointBoundary, PointSurface},
        method::KNearestSearch,
    )
    return search.(points(cloud), Ref(method))
end

function Meshes.searchdists(
        cloud::Union{PointCloud, PointBoundary, PointSurface},
        method::KNearestSearch,
    )
    return searchdists.(points(cloud), Ref(method))
end
