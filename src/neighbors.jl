function Meshes.KNearestSearch(
    cloud::Union{PointCloud,PointSurface}, k::Int; metric=Euclidean()
)
    return KNearestSearch(point(cloud), k; metric=metric)
end

function Meshes.search(cloud::Union{PointCloud,PointSurface}, method::KNearestSearch)
    return search.(point(cloud), Ref(method))
end

function Meshes.searchdists(cloud::Union{PointCloud,PointSurface}, method::KNearestSearch)
    return norm.(getindex.(searchdists.(point(cloud), Ref(method)), 2))
end
