# Laplacian-Based Contraction (LBC) iterative solver

"""
    contract_lbc(surf::PointSurface{𝔼{N},C}, alg::LBCSkeletonization) where {N,C}

Perform Laplacian-Based Contraction on a point surface.

The algorithm iteratively contracts the point cloud onto its medial axis by:
1. Computing an adaptive graph Laplacian on current geometry
2. Updating attraction weights based on local volume collapse
3. Solving a balanced system: (WL×L + WH)×X_new = WH×X_current

Returns `ContractedSurface` with contracted positions and convergence history.

# Reference
Based on "Skeleton Extraction by Mesh Contraction" - Au et al. (2008)
"""
function contract_lbc(
    surf::PointSurface{𝔼{N},C}, alg::LBCSkeletonization
) where {N,C<:CRS}
    # Extract points
    pts = collect(point(surf))
    n = length(pts)

    if n == 0
        return ContractedSurface(pts, surf, Float64[], Float64[])
    end

    # Compute initial Laplacian and volumes
    L, init_volumes, _ = build_adaptive_laplacian(pts, alg.k)

    # Initialize weights
    WL = alg.WL_init
    WH_diag = ones(n)  # Diagonal of attraction weight matrix

    # Track convergence
    convergence = Float64[]
    curr_volumes = copy(init_volumes)

    # Get coordinate unit for reconstruction
    coord_unit = unit(to(first(pts))[1])

    for iter in 1:alg.max_iters
        # Rebuild Laplacian on current geometry
        L, curr_volumes, _ = build_adaptive_laplacian(pts, alg.k)

        # Update attraction weights based on volume collapse
        # Points that have collapsed more get stronger attraction (freeze in place)
        _update_attraction_weights!(WH_diag, init_volumes, curr_volumes)

        # Build and solve the linear system
        new_pts = _solve_contraction_step(L, WL, WH_diag, pts, coord_unit)

        # Compute convergence (max displacement)
        max_disp = _compute_max_displacement(pts, new_pts)
        push!(convergence, max_disp)

        # Update points
        pts = new_pts

        # Check convergence
        if max_disp < alg.tol
            @info "LBC converged in $iter iterations (displacement = $max_disp)"
            break
        end

        # Increase contraction weight
        WL *= alg.WL_factor
    end

    if length(convergence) == alg.max_iters && convergence[end] >= alg.tol
        @warn "LBC reached maximum iterations ($(alg.max_iters)), final displacement = $(convergence[end])"
    end

    return ContractedSurface(pts, surf, curr_volumes, convergence)
end

"""
    _update_attraction_weights!(WH_diag, init_volumes, curr_volumes; eps_vol=1e-10)

Update diagonal attraction weights based on volume collapse ratio.

As local volume approaches zero, the weight increases quadratically,
effectively freezing points that have already collapsed.
"""
function _update_attraction_weights!(
    WH_diag::Vector{Float64},
    init_volumes::Vector{Float64},
    curr_volumes::Vector{Float64};
    eps_vol::Float64=1e-10,
)
    for i in eachindex(WH_diag)
        ratio = init_volumes[i] / (curr_volumes[i] + eps_vol)
        # Quadratic increase in attraction as volume collapses
        WH_diag[i] = ratio^2
    end
    return nothing
end

"""
    _solve_contraction_step(L, WL, WH_diag, points, coord_unit)

Solve the SPD linear system: (WL×L + WH)×X_new = WH×X_current

Uses Cholesky factorization for efficiency and numerical stability.
Returns new point positions.
"""
function _solve_contraction_step(
    L::SparseMatrixCSC,
    WL::Float64,
    WH_diag::Vector{Float64},
    points::Vector{<:Point{𝔼{N},C}},
    coord_unit,
) where {N,C}
    n = length(points)

    # Extract coordinate matrix (n × N)
    X = Matrix{Float64}(undef, n, N)
    for i in 1:n
        X[i, :] .= ustrip.(to(points[i]))
    end

    # Build system matrix A = WL×L + WH (SPD)
    WH = Diagonal(WH_diag)
    A = WL * L + WH

    # RHS: WH × X
    B = WH * X

    # Add small regularization for numerical stability
    A_reg = A + 1e-10 * I

    # Solve via Cholesky (A is SPD)
    F = cholesky(Symmetric(A_reg))
    X_new = F \ B

    # Reconstruct points with original units
    # Use Point constructor directly - it will infer the type from coordinates
    new_points = [Point(ntuple(j -> X_new[i, j] * coord_unit, N)...) for i in 1:n]

    return new_points
end

"""
    _compute_max_displacement(old_points, new_points)

Compute the maximum displacement between old and new point positions.
Used for convergence checking.
"""
function _compute_max_displacement(
    old_points::Vector{<:Point{𝔼{N}}},
    new_points::Vector{<:Point{𝔼{N}}},
) where {N}
    max_disp = 0.0
    for (p_old, p_new) in zip(old_points, new_points)
        disp = sqrt(sum((ustrip.(to(p_old)) .- ustrip.(to(p_new))) .^ 2))
        max_disp = max(max_disp, disp)
    end
    return max_disp
end
