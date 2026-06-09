# Cavity-geometry validation — findings (2026-06-07)

## Purpose

Validate WhatsThePoint's Octree→repel pipeline on the annular cavity geometry
that mirrors Macchiato.jl's `cavity_sphere_recovery_twofront` target domain.

## Geometry

Annular domain: outer sphere R=1.0, inner sphere r=0.547 (matching Macchiato's
reference cavity radius). Mesh: 24×48 per shell = 4608 triangles, 2400 vertices.
Built programmatically in `validate_cavity.jl` (no external STL needed).

## Pipeline tested

```
mesh → TriangleOctree → Octree discretization → repel (live tree, SpacingEquilibriumForce)
```

Parameters: Δ=0.08, alpha=1.0, placement=:random, max_iters=300, tol=1e-4,
rebuild_every=1, k=21 (repel), k=50 (stencil conditioning).

## Results

Pre-fix vs post-fix (`_safe_direction`), same parameters. "After Repel (pre-fix)" is
the failed run that exposed the bug; "After Repel (now)" is the re-validation with the
fix in place.

| Metric | Raw Octree | After Repel (pre-fix) | **After Repel (now)** | gate |
|--------|-----------|-----------------------|-----------------------|------|
| separation | 1.93e-3 (0.024·Δ) | **0.0** | **1.12e-2 (0.140·Δ)** | >0.1·Δ ✅ |
| mesh_ratio | 65.3 (raw 162.8) | **Inf** | **15.3** | <3.0 ❌ |
| separation/Δ | 0.024 | 0.0 | **0.140** | — |
| coincident points | — | 4 (2 pairs) | **0** | 0 ✅ |
| points | 11450 | 11450 (0 escaped) | 11450 (0 escaped) | — |
| min σ_min/σ_max (k=50) | — | 7.7e-5 | 4.7e-4 | — |
| median σ_min/σ_max (k=50) | — | 1.45e-2 | 1.46e-2 | — |
| singular stencils (<1e-8) | — | **0 / 11450** | **0 / 11450** | 0 ✅ |
| **verdict** | — | FAIL (sep=0) | **MARGINAL** | PASS |

Repel **did not converge**: it hit `max_iters=300` with residual movement 0.024 (tol
`1e-4`). The MARGINAL verdict is gated solely by mesh_ratio (15.3 vs <3), which is
driven by covering voids (`fill`=0.17 m), not clustering — i.e. an iteration-starved
relaxation, not a coincidence problem.

## Diagnosis

### 1. Raw octree is Poisson (expected)
separation=0.024·Δ, mesh_ratio=65. Consistent with `:random` placement
(Poisson-uniform per leaf). The NODEGEN_FINDINGS.md already documents this.

### 2. Repel produced ZERO separation — FIXED

**Root cause: `0/0` NaN singularity in force calculation.**

Line 79 (volume-only) and 186 (boundary-projected) of repel.jl:
```julia
compute_force(force_model, r / s) * (xi - xj) / r
```

When two points coincide (`r = 0`):
- `compute_force(model, 0)` = finite (25.0 for SpacingEquilibriumForce)
- `(xi - xj)` = zero vector (same position)
- `/ 0.0` → **NaN**
- Position becomes NaN → boundary projection snaps back to original → stuck forever

**Fix:** Added `_safe_direction(xi, xj, r)` that returns a random unit vector when
`r == 0` instead of the NaN-producing `0/0`. The random perturbation breaks the
coincidence; the boundary projection then re-snaps the point to the surface,
maintaining the boundary constraint.

Also required `r > zero(r)` instead of `r > 0` for Unitful dimensional compatibility.

**Verified:** 5 trials with boundary-projected repel — all have **zero coincident
points** and healthy min separation (0.018-0.031). Before fix: always exactly 4
coincident points.

### 3. Stencil conditioning is actually OK
Despite the separation=0 issue, **no singular stencils** were detected
(σ_min/σ_max > 1e-8 for all 11450 points). The median σ=1.45e-2 is healthy.
This is because the coincident points are on the boundary (projected), and the
k=50 stencil for a boundary point draws mostly from interior neighbors which
are well-separated. The singular-stencil failure mode in Macchiato requires
coincident points in the INTERIOR where the RBF-FD operator is assembled.

## Implications for Macchiato

1. **The NaN singularity fix is essential for shape optimization** — the boundary
   projection is mandatory when the surface moves. The `_safe_direction` function
   ensures coincident boundary points always get a well-defined repulsion force.
2. **Boundary-projected repel now works correctly** — 0 coincident points, healthy
   separation (0.018-0.031), and the boundary constraint is maintained via
   re-projection after the random perturbation step.
3. **The stencil conditioning is encouraging** — even before the fix, the interior
   stencils were well-conditioned for poly_deg=3. The Macchiato `SingularException`
   was caused by near-surface near-duplicates from the lattice, which the octree +
   repel pipeline avoids.

## Update (2026-06-07, session close) — cull added + root cause re-located

Changes landed this session (uncommitted, `shape_optimization_utils`):
- `src/repel.jl`: `_near_duplicate_keep_mask` helper + `cull_ratio` kwarg on both `repel`
  methods (default `0.0` = off). When >0, removes points closer than `cull_ratio·spacing`
  to a kept point (lower index kept). **Symptom treatment / guarded safety net — not the
  real fix.** Tests added in `test/repel.jl` (helper mask + cull invariant).
- `validate_cavity.jl`: added **spacing-fidelity** readout (`d_NN/h` mean + CV +
  percentiles, coordination number) and a before/after `repel` vs `repel+cull` column;
  verdict now gates on the culled cloud with spacing-CV as the primary metric.

**Root-cause re-location (`/tmp/wtp_locate_stuck.jl`, 300-iter repel, no cull):** the
residual close pairs are **NOT boundary points and NOT at the poles** — all 91 pairs with
NN < 0.25·Δ (worst 0.061·Δ) are **interior VOLUME points**, scattered (radii 0.48–0.99),
with ~0 pole accumulation. So:
- The original "4 coincident boundary points" (snap-collapse) defect was real but is
  **fixed** by `_safe_direction`; it is no longer the active defect.
- The current defect is **volume-repel dynamics** (balanced standoff and/or fixed-α
  overshoot limit-cycle), so the fix is the **force/step dynamics** — **adaptive
  per-point step (§2a)** is now the primary quality lever — **not** the boundary
  projection (on hold) and **not** the cull. See `repel_convergence_ideas.md` §0
  "Locator finding" for the full reasoning and the next diagnostic (per-pair tracer).

## Next steps

`validate_cavity.jl` with the fix has been run (numbers above). The remaining work is
**convergence speed and uniformity**, so that repel can run *inside* the optimization
loop (where boundary-point count and connectivity change every design step). In
priority order — see the roadmap in `NODEGEN_FINDINGS.md` for the full framing:

1. **Active/adaptive repel force (Lever 1)** — replace fixed `α` damping with a
   residual-aware adaptive step (and/or an active force law that strengthens where
   spacing error is large) to reach `tol=1e-4` in far fewer than 300 iters and close
   the mesh_ratio/void gap. This is the main blocker to PASS.
2. **Octree-based NN search (Lever 2)** — implement `plan_octree_nn_search.md` (O(N)
   in-place rebuild + spatially-local k-NN on the existing `SpatialOctree`) to make
   each iteration cheap, so re-relaxing on every changed-boundary step is affordable.
3. **`:jittered` octree placement** — lower the initial `fill`/mesh_ratio so Lever 1
   starts from a low-discrepancy cloud instead of Poisson.

## Files

- `validate_cavity.jl` — the validation script (repo root)
- `NODEGEN_FINDINGS.md` — master findings + roadmap
- `plan_octree_nn_search.md` — Lever 2 design
- This document
