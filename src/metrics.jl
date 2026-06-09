"""
    metrics(cloud::PointCloud; k=20)

Compute distance statistics (mean, std, max, min) to the `k` nearest neighbors for all points
in the cloud. Useful for assessing point distribution quality before and after repulsion.

Also reports the global *separation* and *fill* distances and their ratio, the
quasi-uniformity quality measure most relevant to meshless stencil conditioning:

- `separation` — the smallest nearest-neighbor distance anywhere in the cloud.
  Small values signal near-coincident points (the source of singular RBF-FD stencils).
- `fill` — the largest nearest-neighbor distance (a proxy for the worst covering gap).
- `mesh_ratio` — `fill / separation` (≥ 1). Closer to 1 means a more uniform,
  blue-noise-like cloud; large values indicate clustering and voids coexisting.

Returns a `NamedTuple` with fields `avg`, `std`, `max`, `min`, `separation`, `fill`,
`mesh_ratio`, and `k`.
"""
function metrics(cloud::PointCloud; k = 20)
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)
    r = map(x -> x[2][2:end], results) # x[2] = distances, [2:end] skips self
    nn = map(first, r) # per-point nearest-neighbor distance
    avg = mean(mean.(r))
    σ = mean(std.(r))
    mx = mean(maximum.(r))
    mn = mean(minimum.(r))
    separation = minimum(nn)
    fill = maximum(nn)
    mesh_ratio = ustrip(fill / separation)
    println("Cloud Metrics")
    println("-------------")
    println("avg. distance to $k nearest neighbors: $avg")
    println("std. distance to $k nearest neighbors: $σ")
    println("max. distance to $k nearest neighbors: $mx")
    println("min. distance to $k nearest neighbors: $mn")
    println("separation (min nearest-neighbor distance): $separation")
    println("fill (max nearest-neighbor distance):       $fill")
    println("mesh ratio (fill / separation, ≥1):         $mesh_ratio")
    return (; avg, std = σ, max = mx, min = mn, separation, fill, mesh_ratio, k)
end

"""
    spacing_metrics(cloud::PointCloud, spacing::AbstractSpacing; k=20)

Measure how closely the point distribution matches the target `spacing` function.

For each point `xᵢ`, the local actual spacing is estimated as the mean distance to
its `k` nearest neighbors (self excluded). The per-point relative error is

    errorᵢ = |r̄ᵢ − s(xᵢ)| / s(xᵢ)

Returns a `NamedTuple` `(max_error, mean_error, std_error, k)`. Use before and after
`repel` (or any placement step) to quantify spacing preservation.
"""
function spacing_metrics(cloud::PointCloud, spacing::AbstractSpacing; k = 20)
    pts = points(cloud)
    method = KNearestSearch(cloud, k)
    results = searchdists(cloud, method)

    target = ustrip.(spacing.(pts))
    actual = map(r -> mean(ustrip.(@view r[2][2:end])), results)
    errors = @. abs(actual - target) / target

    return (;
        max_error = maximum(errors),
        mean_error = mean(errors),
        std_error = std(errors),
        k,
    )
end
