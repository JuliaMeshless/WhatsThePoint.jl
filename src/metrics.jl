"""
    metrics(cloud::PointCloud; k=20)

Compute distance statistics (mean, std, max, min) to the `k` nearest neighbors for all points
in the cloud. Useful for assessing point distribution quality before and after repulsion.

Returns a `NamedTuple` with fields `avg`, `std`, `max`, `min`, and `k`.
"""
function metrics(cloud::PointCloud; k = 20)
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)
    r = map(x -> x[2][2:end], results) # x[2] = distances, [2:end] skips self
    avg = mean(mean.(r))
    σ = mean(std.(r))
    mx = mean(maximum.(r))
    mn = mean(minimum.(r))
    println("Cloud Metrics")
    println("-------------")
    println("avg. distance to $k nearest neighbors: $avg")
    println("std. distance to $k nearest neighbors: $σ")
    println("max. distance to $k nearest neighbors: $mx")
    println("min. distance to $k nearest neighbors: $mn")
    return (; avg, std = σ, max = mx, min = mn, k)
end
