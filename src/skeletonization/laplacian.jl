# Adaptive Laplacian construction for LBC skeletonization

"""
    build_adaptive_laplacian(points::AbstractVector{<:Point{𝔼{N}}}, k::Int) where {N}

Build an adaptive graph Laplacian matrix with Gaussian weights.

The local scale σᵢ is determined by the distance to the k-th nearest neighbor,
allowing the operator to adapt to varying point densities.

Returns `(L, volumes, scales)` where:
- `L::SparseMatrixCSC` - Graph Laplacian matrix (n × n), L = D - W
- `volumes::Vector{Float64}` - Local volume estimates (σᵢ³)
- `scales::Vector{Float64}` - Local scales (distance to k-th neighbor)

# Reference
Based on "Skeleton Extraction by Mesh Contraction" - Au et al. (2008)
"""
function build_adaptive_laplacian(
    points::AbstractVector{<:Point{𝔼{N}}}, k::Int
) where {N}
    n = length(points)
    k = min(k, n - 1)  # Can't have more neighbors than points - 1

    # Find k-nearest neighbors (exclude self)
    method = KNearestSearch(points, k + 1)
    neighbors_dists = [searchdists(p, method) for p in points]

    # Extract neighbor indices and compute scales
    neighbors = [nd[1][2:end] for nd in neighbors_dists]  # Exclude self (first)
    scales = [ustrip(nd[2][end]) for nd in neighbors_dists]  # Distance to k-th neighbor

    # Build Gaussian weight matrix
    W = _build_gaussian_weights(points, neighbors, scales)

    # Compute Laplacian L = D - W
    L = _compute_laplacian(W)

    # Estimate local volumes as scale^N (2D: area, 3D: volume)
    volumes = scales .^ N

    return L, volumes, scales
end

"""
    _build_gaussian_weights(points, neighbors, scales) -> SparseMatrixCSC

Build sparse symmetric Gaussian weight matrix.

W[i,j] = exp(-||pᵢ - pⱼ||² / (σᵢ × σⱼ))

Uses σᵢ × σⱼ in denominator to ensure symmetry (W[i,j] = W[j,i]).
"""
function _build_gaussian_weights(
    points::AbstractVector{<:Point{𝔼{N}}},
    neighbors::Vector{Vector{Int}},
    scales::Vector{Float64},
) where {N}
    n = length(points)

    # Extract unitless coordinates
    coords = [SVector{N}(ustrip.(to(p))) for p in points]

    # Preallocate COO format arrays
    I = Int[]
    J = Int[]
    V = Float64[]

    for i in 1:n
        σi = scales[i]
        for j in neighbors[i]
            if j != i  # Skip self-loops
                dist_sq = sum((coords[i] .- coords[j]) .^ 2)
                σj = scales[j]
                weight = exp(-dist_sq / (σi * σj))

                push!(I, i)
                push!(J, j)
                push!(V, weight)
            end
        end
    end

    W = sparse(I, J, V, n, n)

    # Symmetrize: W = (W + W') / 2
    # This handles cases where j ∈ neighbors(i) but i ∉ neighbors(j)
    W = (W + W') / 2

    return W
end

"""
    _compute_laplacian(W::SparseMatrixCSC) -> SparseMatrixCSC

Compute the unnormalized graph Laplacian L = D - W.

D is the diagonal degree matrix where D[i,i] = Σⱼ W[i,j].
"""
function _compute_laplacian(W::SparseMatrixCSC)
    n = size(W, 1)

    # Compute degree (row sums)
    degrees = vec(sum(W; dims=2))

    # Create diagonal matrix D
    D = sparse(1:n, 1:n, degrees, n, n)

    # Laplacian L = D - W
    L = D - W

    return L
end
