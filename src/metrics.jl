function metrics(cloud::PointCloud; k = 20)
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)
    r = map(x -> x[2][2:end], results) # x[2] = distances, [2:end] skips self
    println("Cloud Metrics")
    println("-------------")
    println("avg. distance to $k nearest neighbors: $(mean(mean.(r)))")
    println("std. distance to $k nearest neighbors: $(mean(std.(r)))")
    println("max. distance to $k nearest neighbors: $(mean(maximum.(r)))")
    println("min. distance to $k nearest neighbors: $(mean(minimum.(r)))")
    return nothing
end
