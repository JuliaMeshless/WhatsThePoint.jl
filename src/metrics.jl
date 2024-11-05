function metrics(cloud::PointCloud; k=20)
    method = KNearestSearch(cloud, k)
    _, r = searchdists(cloud, method)
    r = map(x -> x[2:end], r) # strip the first element, which is the point itself
    println("Cloud Metrics")
    println("-------------")
    println("avg. distance to $k nearest neighbors: $(mean(mean.(r)))")
    println("std. distance to $k nearest neighbors: $(mean(std.(r)))")
    println("max. distance to $k nearest neighbors: $(mean(maximum.(r)))")
    println("min. distance to $k nearest neighbors: $(mean(minimum.(r)))")
    return nothing
end
