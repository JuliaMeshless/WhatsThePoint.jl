function Meshes.KNearestSearch(
    cloud::Union{PointCloud,PointBoundary,PointSurface}, k::Int; metric=Euclidean()
)
    return KNearestSearch(pointify(cloud), k; metric=metric)
end

function Meshes.search(
    cloud::Union{PointCloud,PointBoundary,PointSurface}, method::KNearestSearch
)
    return search.(pointify(cloud), Ref(method))
end

function Meshes.searchdists(
    cloud::Union{PointCloud,PointBoundary,PointSurface}, method::KNearestSearch
)
    return searchdists.(pointify(cloud), Ref(method))
end

function Meshes.searchdists(p::Vec, method::KNearestSearch)
    return searchdists(Point(Tuple(p)), method)
end
