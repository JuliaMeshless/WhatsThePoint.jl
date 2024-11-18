abstract type AbstractNodeGenerationAlgorithm end

include("algorithms/fornberg_flyer.jl")
include("algorithms/vandersande_fornberg.jl")
include("algorithms/slak_kosec.jl")

function discretize(
    boundary::PointBoundary{𝔼{3}},
    spacing::AbstractSpacing;
    alg::AbstractNodeGenerationAlgorithm=SlakKosec(),
    max_points=10_000_000,
)
    cloud = PointCloud(boundary)
    discretize!(cloud, spacing; alg=alg, max_points=max_points)
    return cloud
end

function discretize(
    boundary::PointBoundary{𝔼{2}},
    spacing::AbstractSpacing;
    alg::AbstractNodeGenerationAlgorithm=FornbergFlyer(),
    max_points=10_000_000,
)
    @warn "Only FornbergFlyer algorithm is implemented for 2D point clouds. Using it."
    cloud = PointCloud(boundary)
    discretize!(cloud, spacing; alg=FornbergFlyer(), max_points=max_points)
    return cloud
end

function discretize!(
    cloud::PointCloud,
    spacing::AbstractSpacing;
    alg::AbstractNodeGenerationAlgorithm=SlakKosec(),
    max_points=10_000_000,
)
    return discretize!(cloud, spacing, alg; max_points=max_points)
end

function calculate_ninit(cloud::PointCloud{𝔼{3}}, s::VariableSpacing)
    min_s = s(first(pointify(cloud)))
    bbox = boundingbox(cloud)
    extent = bbox.max - bbox.min
    return (ceil(Int, extent[1] * 10 / min_s), ceil(Int, extent[2] * 10 / min_s))
end

function calculate_ninit(cloud::PointCloud{𝔼{3}}, s::ConstantSpacing)
    bbox = boundingbox(cloud)
    extent = bbox.max - bbox.min
    return (ceil(Int, extent[1] * 10 / s.Δx), ceil(Int, extent[2] * 10 / s.Δx))
end

function calculate_ninit(cloud::PointCloud{𝔼{2}}, s::ConstantSpacing)
    bbox = boundingbox(cloud)
    extent = bbox.max - bbox.min
    return ceil(Int, extent[1] * 10 / s.Δx)
end
