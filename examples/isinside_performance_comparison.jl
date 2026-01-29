using CairoMakie

# Theoretical complexity constants (relative scaling factors)
# These are calibrated to show realistic crossover points:
# - Octree breaks even at ~30 queries (low setup, moderate per-query)
# - FMM breaks even at ~5000 queries (expensive setup, cheap per-query)

const C_BRUTE_BUILD = 1.0          # Essentially no build cost
const C_OCTREE_BUILD = 6.0         # O(N log N) - tree construction + leaf classification
const C_FMM_BUILD = 5000.0         # O(N) but LARGE constant - multipole expansions, translation operators

const C_BRUTE_QUERY = 1.0          # O(N) per query - sum over all boundary elements
const C_OCTREE_QUERY = 50.0        # O(log N + k) - tree traversal + k triangle checks
const K_OCTREE = 15                # Average triangles checked per query
const C_FMM_QUERY = 200.0          # O(1) constant - table lookup/interpolation, independent of N

# Boundary sizes to evaluate
N_values = 10 .^ range(2, 6, length = 100)

# Query counts to evaluate
Q_values = 10 .^ range(0, 6, length = 100)

# Fixed N for total cost plot
N_fixed = 70_000

# Calculate theoretical costs
function build_cost_brute(N)
    return C_BRUTE_BUILD
end

function build_cost_octree(N)
    return C_OCTREE_BUILD * N * log10(N)
end

function build_cost_fmm(N)
    return C_FMM_BUILD * N
end

function query_cost_brute(N)
    return C_BRUTE_QUERY * N
end

function query_cost_octree(N)
    return C_OCTREE_QUERY * (log10(N) + K_OCTREE)
end

function query_cost_fmm(_)
    return C_FMM_QUERY  # True O(1) - constant regardless of N
end

function total_cost(build_fn, query_fn, N, Q)
    return build_fn(N) + Q * query_fn(N)
end

# Create figure with 4 subplots
fig = Figure(size = (1200, 1000), fontsize = 14)

# Color scheme
colors = (brute = :steelblue, octree = :coral, fmm = :seagreen)

# Plot 1: Build/Setup Cost vs Boundary Size
ax1 = Axis(
    fig[1, 1],
    xlabel = "Number of Boundary Points (N)",
    ylabel = "Relative Build Cost",
    xscale = log10,
    yscale = log10,
    title = "Build/Setup Cost vs Boundary Size",
)

lines!(
    ax1,
    N_values,
    build_cost_brute.(N_values),
    label = "Brute Force ~O(1)",
    color = colors.brute,
    linewidth = 2,
)
lines!(
    ax1,
    N_values,
    build_cost_octree.(N_values),
    label = "Octree O(N log N)",
    color = colors.octree,
    linewidth = 2,
)
lines!(
    ax1,
    N_values,
    build_cost_fmm.(N_values),
    label = "FMM O(N)",
    color = colors.fmm,
    linewidth = 2,
)
axislegend(ax1, position = :lt)

# Plot 2: Per-Query Cost vs Boundary Size
ax2 = Axis(
    fig[1, 2],
    xlabel = "Number of Boundary Points (N)",
    ylabel = "Relative Per-Query Cost",
    xscale = log10,
    yscale = log10,
    title = "Per-Query Cost vs Boundary Size",
)

lines!(
    ax2,
    N_values,
    query_cost_brute.(N_values),
    label = "Brute Force O(N)",
    color = colors.brute,
    linewidth = 2,
)
lines!(
    ax2,
    N_values,
    query_cost_octree.(N_values),
    label = "Octree O(log N + k)",
    color = colors.octree,
    linewidth = 2,
)
lines!(
    ax2,
    N_values,
    query_cost_fmm.(N_values),
    label = "FMM O(1)*",
    color = colors.fmm,
    linewidth = 2,
)
axislegend(ax2, position = :lt)

# Plot 3: Total Cost vs Number of Queries for Fixed N
ax3 = Axis(
    fig[2, 1],
    xlabel = "Number of Queries (Q)",
    ylabel = "Total Cost (build + Q × query)",
    xscale = log10,
    yscale = log10,
    title = "Total Cost vs Queries (N = 70,000)",
)

total_brute = [total_cost(build_cost_brute, query_cost_brute, N_fixed, Q) for Q in Q_values]
total_octree =
    [total_cost(build_cost_octree, query_cost_octree, N_fixed, Q) for Q in Q_values]
total_fmm = [total_cost(build_cost_fmm, query_cost_fmm, N_fixed, Q) for Q in Q_values]

lines!(
    ax3,
    Q_values,
    total_brute,
    label = "Brute Force",
    color = colors.brute,
    linewidth = 2,
)
lines!(ax3, Q_values, total_octree, label = "Octree", color = colors.octree, linewidth = 2)
lines!(ax3, Q_values, total_fmm, label = "FMM", color = colors.fmm, linewidth = 2)

# Find and mark crossover points
function find_crossover(costs1, costs2, Q_vals)
    for i = 2:length(Q_vals)
        if (costs1[i-1] <= costs2[i-1]) && (costs1[i] > costs2[i])
            return Q_vals[i]
        elseif (costs1[i-1] >= costs2[i-1]) && (costs1[i] < costs2[i])
            return Q_vals[i]
        end
    end
    return nothing
end

crossover_octree = find_crossover(total_brute, total_octree, Q_values)
crossover_fmm = find_crossover(total_brute, total_fmm, Q_values)
crossover_octree_fmm = find_crossover(total_octree, total_fmm, Q_values)

if !isnothing(crossover_octree)
    vlines!(ax3, [crossover_octree], color = colors.octree, linestyle = :dash, alpha = 0.5)
end
if !isnothing(crossover_fmm)
    vlines!(ax3, [crossover_fmm], color = colors.fmm, linestyle = :dash, alpha = 0.5)
end
if !isnothing(crossover_octree_fmm)
    vlines!(ax3, [crossover_octree_fmm], color = :purple, linestyle = :dot, alpha = 0.7)
end

axislegend(ax3, position = :lt)

# Plot 4: Speedup Factor vs Number of Queries
ax4 = Axis(
    fig[2, 2],
    xlabel = "Number of Queries (Q)",
    ylabel = "Speedup Factor (vs Brute Force)",
    xscale = log10,
    yscale = log10,
    title = "Speedup vs Brute Force (N = 70,000)",
)

speedup_octree = total_brute ./ total_octree
speedup_fmm = total_brute ./ total_fmm

lines!(
    ax4,
    Q_values,
    speedup_octree,
    label = "Octree",
    color = colors.octree,
    linewidth = 2,
)
lines!(ax4, Q_values, speedup_fmm, label = "FMM", color = colors.fmm, linewidth = 2)
hlines!(ax4, [1.0], color = :gray, linestyle = :dash, linewidth = 1, label = "Break-even")
axislegend(ax4, position = :rb)

# Add overall title
Label(
    fig[0, :],
    "isinside() Performance Comparison: Brute Force vs Octree vs FMM",
    fontsize = 18,
    font = :bold,
)

# Save figure
output_path = joinpath(@__DIR__, "isinside_performance.png")
save(output_path, fig, px_per_unit = 2)
println("Saved performance comparison to: $output_path")

# Print summary statistics
println("\nSummary for N = $N_fixed boundary points:")
println("="^50)
println("\nBuild costs (relative):")
println("  Brute Force: $(round(build_cost_brute(N_fixed), digits = 1))")
println("  Octree:      $(round(build_cost_octree(N_fixed), digits = 1))")
println("  FMM:         $(round(build_cost_fmm(N_fixed), digits = 1))")

println("\nPer-query costs (relative):")
println("  Brute Force: $(round(query_cost_brute(N_fixed), digits = 1))")
println("  Octree:      $(round(query_cost_octree(N_fixed), digits = 1))")
println("  FMM:         $(round(query_cost_fmm(N_fixed), digits = 1))")

println("\nBreak-even points (vs Brute Force):")
if !isnothing(crossover_octree)
    println("  Octree: ~$(round(Int, crossover_octree)) queries")
end
if !isnothing(crossover_fmm)
    println("  FMM:    ~$(round(Int, crossover_fmm)) queries")
end
if !isnothing(crossover_octree_fmm)
    println("\nOctree vs FMM crossover: ~$(round(Int, crossover_octree_fmm)) queries")
    println("  (FMM becomes faster than Octree above this point)")
end

println("\nAt 1,000,000 queries:")
println("  Octree speedup: $(round(total_brute[end] / total_octree[end], digits = 1))x")
println("  FMM speedup:    $(round(total_brute[end] / total_fmm[end], digits = 1))x")

println("\nRecommendation:")
println("  Q < $(round(Int, something(crossover_octree, 1))): Use Brute Force")
if !isnothing(crossover_octree_fmm)
    println(
        "  $(round(Int, something(crossover_octree, 1))) < Q < $(round(Int, crossover_octree_fmm)): Use Octree",
    )
    println("  Q > $(round(Int, crossover_octree_fmm)): Use FMM")
else
    println("  Q > $(round(Int, something(crossover_octree, 1))): Use Octree")
end
