```@meta
CurrentModule = WhatsThePoint
```

# Node Repulsion

## Why Repulsion Matters

Meshless PDE methods (RBF-FD, generalized finite differences) are sensitive to point distribution quality. Irregular spacing leads to poorly conditioned interpolation matrices and reduced accuracy. Node repulsion iteratively pushes points apart to achieve a more uniform distribution while respecting the domain boundary.

## Usage

```julia
cloud, convergence = repel(cloud, spacing; β=0.2, max_iters=1000)
```

[`repel`](@ref) returns a tuple of `(new_cloud, convergence_vector)`. The new cloud has [`NoTopology`](@ref) since points have moved — call [`set_topology`](@ref) again after repulsion.

!!! note "Boundary points are fixed"
    Only volume (interior) points are moved during repulsion. Boundary points remain in place to preserve the original surface geometry.

!!! note "Points pushed outside are removed"
    At each iteration, any volume point that has been pushed outside the domain boundary is automatically filtered out via `isinside`. This means the total point count may decrease slightly after repulsion.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `β` | `0.2` | Repulsion strength — controls force magnitude |
| `α` | `0.05 × min(spacing)` | Step size — distance points move per iteration |
| `k` | `21` | Number of nearest neighbors used in repulsion stencil |
| `max_iters` | `1000` | Maximum number of repulsion iterations |
| `tol` | `1e-6` | Convergence tolerance on relative point movement |

## Tuning Guide

- **`β` (repulsion strength):** Values in the range 0.1–0.5 work well for most problems. Smaller values give gentler repulsion (slower convergence, more stable). Larger values produce stronger forces (faster convergence, risk of oscillation).
- **`k` (neighbor count):** Should roughly match the stencil size your meshless solver will use. Too small and points only feel local pressure; too large and the computation slows without benefit.
- **`α` (step size):** The default (5% of minimum spacing) is conservative. Increase for faster convergence on well-behaved geometries; decrease if points escape the domain.
- **`max_iters`:** 1000 is usually sufficient. Check the convergence vector to see if more iterations are needed.

## Algorithm Details

Each iteration:

1. Build a k-nearest neighbor tree of all points
2. For each point, compute a repulsive force from its k neighbors using

```math
F(r) = \frac{1}{(r^2 + \beta)^2}
```

where `r` is the distance to a neighbor. The `β` parameter prevents singularity at `r = 0` and controls the force shape.

3. Move each point by `α` in the direction of the net repulsive force
4. Filter out any points that have moved outside the domain using `isinside`
5. Record the maximum relative displacement as the convergence metric

## Convergence Monitoring

The convergence vector records the maximum relative point displacement at each iteration:

```julia
cloud, conv = repel(cloud, spacing; β=0.2, max_iters=500)

# Check if converged
println("Final displacement: ", conv[end])
println("Iterations used: ", length(conv))
```

If `conv[end]` is still large (say > 1e-3), the distribution may benefit from more iterations or parameter tuning.

## Verifying Distribution Quality

Use `metrics` to quantify the point distribution before and after repulsion:

```julia
# Before repulsion
metrics(cloud)

# After repulsion
cloud_repelled, conv = repel(cloud, spacing)
metrics(cloud_repelled)
```

`metrics` prints the average, standard deviation, maximum, and minimum distances to each point's k nearest neighbors. A lower standard deviation indicates a more uniform distribution.

## Reference

- Miotti, M. (2023). Node repulsion for meshless discretizations.
