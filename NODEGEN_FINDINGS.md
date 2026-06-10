# Node-generation assessment & repel fix — session findings (2026-06-06, updated 2026-06-09)

Goal: perfect WhatsThePoint node generation (Octree fill + repel refinement) so it
produces a "perfect" node cloud for downstream RBF-FD meshless solving (Macchiato
shape-opt). Validation target: annular cavity (R=1, r=0.547), quality judged by
`validate_cavity.jl` (separation, spacing CV, coordination, per-stencil degree-3
Vandermonde σ_min/σ_max).

## Strategic goal (the why behind this work)

The real target is a **significantly better-performing octree-based node generation
algorithm** so that **repel refinement can be brought *inside* the shape-optimization
loop**, not just run once as a preprocessing step. That is hard because:

- During repel the **number of boundary points changes** (escaped points are
  re-projected/discarded, deficit fills added), which in turn **changes the
  connectivity / neighbor (stencil) matrix** every iteration. A node cloud that is
  re-relaxed each design step must therefore re-relax *fast* and *predictably*.
- So the gating requirement is **repel convergence speed and robustness**, not just
  final quality.

## Current status (2026-06-09)

**Quality gate: PASS** (was MARGINAL on 2026-06-07).

| Metric (cavity, Δ=0.08, 11450 pts, 300 iters) | 2026-06-06 | 2026-06-07 | 2026-06-09 | target |
|---|---|---|---|---|
| separation / Δ | 0.0 | 0.140 | **0.500** | > 0.1 ✓ |
| coincident points | 4 | 0 | **0** | 0 ✓ |
| spacing CV (d_NN/h) | — | — | **0.140** | < 0.15 ✓ |
| p05 (d_NN/h) | — | — | **0.541** | — |
| p95 (d_NN/h) | — | — | **0.863** | — |
| coordination (≤1.4h) | — | — | **18.6** | 12–14 |
| singular stencils (k=50, deg 3) | 0/11450 | 0/11450 | **0/11403** | 0 ✓ |
| min σ_min/σ_max | 7.7e-5 | 4.7e-4 | **3.8e-4** | — |
| median σ_min/σ_max | 1.45e-2 | 1.46e-2 | **1.49e-2** | — |
| **verdict** | FAIL | MARGINAL | **PASS** | — |

### What changed since 2026-06-07

The standoff diagnosis from `repel_convergence_ideas.md` §0 was confirmed: the
closest pair freezes at a fixed `r/s` for hundreds of iterations (balanced standoff,
not overshoot). Three fixes applied:

1. **Adaptive per-point step** (`α_i = clamp(1/|F|, α_lo, α_max)`) — eliminates
   oscillation, each point takes the maximum safe step proportional to its local
   force magnitude.
2. **Force-norm convergence** (`max_i(|F_i| * s_i)`) — replaces displacement-based
   metric; detects true equilibrium earlier, avoids false stops from oscillating
   points.
3. **Stochastic frozen-pair kick** (`kick_after=10`) — tracks the closest pair; if
   the same pair stays at the same `r/s` for 10 consecutive iterations, applies a
   small random displacement (0.1·s) to the volume point. Breaks the balanced
   standoff symmetry. This was the key lever that moved separation from 0.14·Δ to
   0.50·Δ.

Also implemented:
- **Displacement cap** (`|Δp| ≤ s_i`) — prevents runaway from strong forces; safety
  harness for stronger force laws.
- **`StrongSpacingForce(β, γ)`** — force law with tunable singularity exponent
  (γ=3 default vs γ=2 for `SpacingEquilibriumForce`). Tested but found to make
  standoffs *worse* (stronger core moves the equilibrium closer, not farther).
  Kept as an option but not the default.
- **`spacing_fidelity_metrics`** — new quality function returning p05/p50/p95 of
  d_NN/h, CV, and coordination. Replaces mesh_ratio as the primary spread metric.

### What's still above target

- **coordination = 18.6** (target 12–14). The cloud is denser than a perfect
  blue-noise packing. Likely cause: the repel hasn't fully converged (force-norm
  plateau ~0.75 after 300 iters), so points haven't spread out enough. The
  coordination threshold (1.4h) may also be too generous for the current spacing
  distribution.

## Quality indicators (revised 2026-06-09)

`mesh_ratio` (fill/separation) dropped as a primary metric — it's a min/max ratio
sensitive to outliers. Replaced by:

| Indicator | Definition | Target |
|---|---|---|
| separation/Δ | min(d_NN) / Δ | > 0.1 |
| spacing CV | std(d_NN/h) / mean(d_NN/h) | < 0.15 |
| p05 (d_NN/h) | 5th percentile of per-point spacing fidelity | — |
| p95 (d_NN/h) | 95th percentile of per-point spacing fidelity | — |
| coordination | mean count of neighbors within 1.4h | 12–14 |
| singular stencils | count of σ_min/σ_max < 1e-8 | 0 |

## Branch state
- `shape_optimization_utils` is the most advanced branch. All octree + repel +
  force-model work is here.

## Architecture verdict
The Octree→repel split is the right decomposition:
- **Octree** (`src/discretization/algorithms/octree.jl`): graded *density* (node count
  follows `h(x)`), works on arbitrary STL.
- **Repel** (`src/repel.jl`): local *regularity* (blue-noise spacing) — what actually
  conditions RBF-FD stencils and removes the close-pair `SingularException` failure.

### Defects found and fixed

1. **Frozen neighbor graph** (2026-06-06) — kd-tree built once, never rebuilt.
   Fixed: `rebuild_every=1` default, live graph.
2. **Wrong default force** (2026-06-06) — `InverseDistanceForce` (no root, cloud
   drifts). Fixed: `SpacingEquilibriumForce` default.
3. **Coincident-point NaN** (2026-06-07) — `0/0` direction at r=0. Fixed:
   `_safe_direction` returns random unit vector.
4. **Balanced standoff** (2026-06-09) — closest pair freezes at r/s≈0.15–0.19,
   forces balance exactly. Fixed: `kick_after` stochastic perturbation.

### Defects found, NOT fixed

1. **Float32 STL → Octree type mismatch.** `box.stl` loads as Float32; Octree
   emits Float64 volume points → CRS type mismatch. Workaround: promote mesh
   to Float64. Worth fixing in `src/discretization/algorithms/octree.jl`.

## Changes applied (on `shape_optimization_utils`, uncommitted)

`src/repel.jl`:
- Live neighbor graph (`rebuild_every=1`)
- Default force → `SpacingEquilibriumForce(β)`
- `_safe_direction` (coincident point fix)
- Adaptive per-point step size
- Force-norm convergence criterion
- Displacement cap (`|Δp| ≤ s`)
- `kick_after` parameter (stochastic frozen-pair kick)
- `_kick_frozen_pair!` helper
- `_trace_closest_pair!` diagnostic

`src/repel_forces.jl`:
- `StrongSpacingForce(β, γ)` — tunable singularity exponent

`src/metrics.jl`:
- `spacing_fidelity_metrics` — p05/p50/p95 of d_NN/h, CV, coordination

`src/WhatsThePoint.jl`:
- Exported `StrongSpacingForce`, `spacing_fidelity_metrics`

`test/repel.jl`:
- Fixed `β kwarg` test for `SpacingEquilibriumForce` default
- Adjusted convergence `tol` for force-norm scale

`validate_cavity.jl`:
- Added `kick_after=10` run
- Switched to `spacing_fidelity_metrics`
- Updated gate to use separation/Δ + CV (dropped mesh_ratio)

## Validation
- `validate_cavity.jl` (repo root): annular cavity through Octree→repel→kick→cull.
  Latest run = PASS. Rerun: `jlrun validate_cavity.jl [Δ]`
- `validate_repel.jl` (repo root): box.stl frozen-vs-live comparison.

## Roadmap (remaining work)

**Quality (if needed):**
- Coordination is 18.6 vs target 12–14. May need more iterations or a better
  initial placement to reach blue-noise density.

**Speed (defer to wall-clock phase):**
1. Octree-based NN search (`plan_octree_nn_search.md`) — O(N) in-place rebuild
2. `:jittered` / `:bridson` initial placement — fewer repel iterations
3. Momentum / multi-grid — faster convergence

**Housekeeping:**
- Update `CLAUDE.md` repel docs
- Float32/Float64 type mismatch fix in octree.jl
