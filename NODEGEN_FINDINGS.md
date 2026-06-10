# Node-generation assessment & repel fix — session findings (2026-06-06, updated 2026-06-10)

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

## Session 2026-06-10 — refactor, cull philosophy, deposition

### repel.jl refactor (548 → ~470 lines, no behavior change)
Single `_relax!` driver shared by both `repel` methods, parameterized by a
`constrain(id, xi, x_proposed)` callback. Closest-pair trace/kick now read the
sweep's k-NN data (was a per-iteration O(N²) scan). Fixed a latent unit bug
(kick hardcoded `Unitful.m`). Benchmarked vs pre-refactor HEAD: identical
(0.34 s vs 0.36 s / 10 iters at 46.8k pts; 0.73 GB alloc both — the churn is
pre-existing `searchdists` allocation + per-iteration kd rebuild, Tier C target).

### Cull: fixed, and reframed as a defect signal
- Test bug: `m = metrics(...)` assigned over imported `Unitful.m` (error).
- Scenario bug (the real lesson): the test prescribed `h = bbox/8 ≈ 5.4 m` on a
  boundary tessellated at 0.22 m (box.stl is a mm-authored STL read as meters,
  46,786 face centers) — 24× mismatch; the cull was being asked to decimate the
  boundary. **Always check spacing vs geometry resolution before trusting a
  failing assertion.**
- Mask now uses `BallSearch` (exact guarantee for clusters of any size; the old
  fixed k=8 silently violated it; cheaper in healthy clouds too).
- **Design principle (Davide): the cull should NEVER fire in healthy generation —
  activation = upstream defect.** Both `repel` methods now `@warn` when the cull
  removes anything. The cavity currently culls ~42–59 points every run (boundary
  projection parks pairs on shared edges/vertices) — that is the open defect the
  warning now surfaces.

### Deposition: emergent boundary sampling (`deposit_ratio`)
Face-center import hard-couples boundary node count to tessellation
(`n_boundary ≡ n_triangles`), which is backwards: the volume density is what the
PDE solve needs; surface sampling should *emerge* from containing it.
Implemented in the octree `repel`: an escaped volume point is projected onto the
nearest triangle and **converted to a boundary point** — accepted only if no
boundary point already sits within `deposit_ratio · spacing` of the landing site
(self-limiting; conversion is one-way; serial post-sweep pass to avoid
deposition races). Boundary membership is a per-point `is_bnd::Vector{Bool}`
(benchmarked: zero cost vs the old index-prefix convention).

First result (sparse-boundary box, h=3 m, 234 seeds + 600 volume): **174 points
deposited, boundary 234 → 408**, totals conserved, valid triangle normals.
Confirmed pressure-driven: an under-filled volume deposits nothing (it expands
inward instead) — the volume must be at target density.

### SOTA assessment (honest)
- **Quality**: competitive on the internal gate (0 singular, sep 0.5·h, CV 0.14)
  but coordination 18.6 vs 12–14 ideal, and no head-to-head benchmark vs the
  reference direct generators (Medusa/Slak-Kosec PNP) has been run.
- **Speed**: behind direct generation — PNP gets sep ≥ 0.5·h *by construction*
  in one O(N log N) pass; we need ~300 iterations + kick + cull.
- **Differentiator**: incremental re-relaxation in the shape-opt loop +
  deposition (emergent boundary, no published equivalent known). One experiment
  old — needs curved geometry, variable h, scale.

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

## Changes applied

2026-06-09 work (adaptive step, kick, force-norm convergence, `StrongSpacingForce`,
`spacing_fidelity_metrics`) is **committed** (`9d685d9` + follow-up). Uncommitted on
`shape_optimization_utils` as of 2026-06-10 EOD:

`src/repel.jl`:
- Refactor: shared `_relax!` loop, `_closest_pair`/`_maybe_kick!` from sweep k-NN
  data, kick unit fix
- `_near_duplicate_keep_mask` → `BallSearch` (exact cull guarantee)
- `@warn` when the cull removes points (cull = defect signal)
- `deposit_ratio` + `is_bnd` mask + serial deposition pass (volume→boundary
  conversion with acceptance test)

`test/repel.jl`:
- Cull testitem: `Unitful.m` shadow fix, consistent spacing (0.25 m ≈ box.stl
  resolution), renamed to "enforces the separation guarantee"
- Mask testitem: 12-point cluster case (locks the BallSearch guarantee)
- New testitem: "deposit_ratio grows the boundary from escaped points"

## Validation
- `validate_cavity.jl` (repo root): annular cavity through Octree→repel→kick→cull.
  PASS (re-confirmed 2026-06-10 after refactor + deposition, deposit off:
  sep/Δ=0.500, CV=0.140, 0 singular). Rerun: `jlrun validate_cavity.jl [Δ]`
- `validate_repel.jl` (repo root): box.stl frozen-vs-live comparison.
- Full `Pkg.test()` green (142k+ assertions) as of 2026-06-10.

## Path forward (agreed 2026-06-10, resume 2026-06-11)

**Phase 1 — deposition on real geometry (next session):**
1. Cavity with deposition: decimate the imported boundary to ~h spacing (or
   sparser) + `deposit_ratio=0.5`; compare gate vs face-center baseline.
   Success = PASS **and cull silent (0 removed)**.
2. Boundary-free entry point: discretize from `TriangleOctree` + spacing alone
   (pure volume fill), deposition grows the boundary from nothing. Needs
   empty-`PointBoundary` plumbing.

**Phase 2 — make the cull permanently silent:**
3. Fix boundary-projection collisions (pairs parked on shared edges/vertices) —
   e.g. tangential separation at the landing site, or an acceptance-style
   collision check for existing boundary points during projection.

**Phase 3 — SOTA benchmark (the claim needs evidence):**
4. Benchmark harness vs Medusa (Slak–Kosec PNP) on identical geometries:
   separation/Δ, CV, p05/p95, coordination, stencil conditioning, wall-clock.
5. Close the coordination gap (18.6 → 12–14): more convergence, better seeding,
   or recalibrate the 1.4h threshold.

**Phase 4 — speed (Tier C, now measurement-backed):**
6. Octree NN search (`plan_octree_nn_search.md`) — justified: 0.73 GB alloc per
   10 iters at 46.8k pts is `searchdists` allocations + per-iteration kd rebuild.
7. `:jittered`/`:bridson` seeding, momentum, multi-grid.

**Housekeeping:**
- Update `CLAUDE.md` repel docs (`α_min`, `kick_after`, `cull_ratio` warn,
  `deposit_ratio`, `trace`)
- Float32/Float64 type mismatch fix in octree.jl
- Commit the 2026-06-10 working tree (refactor first, deposition second, if
  bisectability matters)
