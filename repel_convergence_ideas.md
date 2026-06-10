# Repel convergence & initial placement — improvement ideas

## 0. Evidence & re-tiering (2026-06-07, updated 2026-06-09)

A staged diagnostic (cavity Δ=0.08, 11450 pts) settled the key question — **is
mesh_ratio 15 a convergence problem or an equilibrium problem?**

| cum. iters | sep/Δ | fill/Δ | mesh_ratio | residual |
|---|---|---|---|---|
| 0 (raw)    | 0.009 | 1.07 | 117  | —     |
| 100        | **0.0550** | 0.96 | 17.5 | 0.052 |
| 300        | **0.0550** | 0.89 | 16.1 | 0.039 |
| 600        | **0.0550** | 0.89 | 16.1 | 0.79  |
| 1000       | **0.0550** | 0.88 | 16.0 | 0.039 |
| 1500       | **0.0550** | 0.92 | 16.8 | 0.035 |

**Findings that re-tiered this document (2026-06-07):**

1. **The bulk equilibrium is already excellent.** `fill/Δ ≈ 0.88–0.96` means even the
   worst covering gap has a neighbor at ~0.9·Δ.
2. **`separation` is a frozen equilibrium defect, not iteration starvation.** `sep/Δ`
   sits at **exactly 0.0550 for 1400 iterations** and never moves.
3. **`mesh_ratio` is the wrong gate.** 16 = fill/sep = 0.9/0.055 is ~entirely the one
   stuck pair — a brittle min/max ratio dominated by a handful of outliers.

### Locator finding (2026-06-07)

Located where the stuck points actually are:
- **91 pairs with NN < 0.25·Δ; worst pair at 0.061·Δ.**
- **All of them are VOLUME points**, scattered through the interior, **not** on boundary.
- **Pole accumulation ≈ 0** — no points pile at lat-long poles.

**Root cause confirmed (2026-06-09):** balanced standoff — the closest pair freezes
at a fixed `r/s` for hundreds of iterations. Force trace shows `r/s` constant (not
oscillating), so it's a genuine equilibrium where neighbor forces cancel, not an
overshoot limit-cycle.

**Fix applied:** stochastic frozen-pair kick (`kick_after=10`). Tracks the closest
pair; if frozen for 10 iterations, applies a small random displacement (0.1·s) to
the volume point. Breaks the symmetry. Result: separation 0.14·Δ → 0.50·Δ.

---

## 1. Initial placement: the biggest lever (SPEED — deferred)

The current `:random` placement draws Poisson-uniform points per octree leaf.
This guarantees arbitrarily close pairs AND voids.

### 1a. Bridson Poisson-disk sampling (best option)
O(N), guarantees min separation ≥ h, produces blue-noise. Adapt to graded spacing
by using `h_min / √3` for the background grid.

### 1b. Centroidal Voronoi pre-relaxation (CVT)
One Lloyd step removes ~80% of Poisson clustering. O(N log N) per step.

### 1c. Octree-grid seeding
Place one point per leaf at a jittered position. Trivial, guarantees min separation
≈ leaf size.

### 1d. Recommendation
1. Octree-grid seeding (replace `:random` default) — trivial, big win
2. Bridson Poisson-disk (`:bridson`) — best quality, moderate effort
3. CVT pre-relaxation — diminishing returns if Bridson is done

## 2. Convergence acceleration

### 2a. Adaptive per-point step size — DONE (2026-06-09)
`α_i = clamp(1/(|F_i|+ε), α_min, α_max)`. Points with strong forces take the
maximum safe step; points near equilibrium take tiny steps. Eliminates oscillation.

### 2b. Force-norm stopping criterion — DONE (2026-06-09)
`convergence = max_i(|F_i| * s_i)`. At equilibrium forces vanish. Detects true
convergence earlier than displacement-based metrics.

### 2c. Over-relaxation / momentum (DEFERRED)
`v_i ← γ * v_i + α * F_i`, γ ≈ 0.5–0.8. Accelerates smooth modes by ~2–3×.
Danger: can diverge if α too large.

### 2d. Multi-grid repel (DEFERRED)
Coarse pass (k=8, 50 iters) + fine pass (k=21, 50 iters). Total ~60% wall-clock
of a single k=21 pass for 100 iters.

### 2e. Near-duplicate pre-cull — DONE (2026-06-07)
`cull_ratio` parameter. Greedy mask removes points closer than `ratio * spacing`.
Off by default; guarded safety net only.

### 2f. Stochastic frozen-pair kick — DONE (2026-06-09)
`kick_after` parameter. Tracks closest pair; if frozen for N iterations, applies
0.1·s random displacement to the volume point. Breaks balanced standoffs. The key
fix that moved separation from 0.14·Δ to 0.50·Δ.

### 2g. Displacement cap — DONE (2026-06-09)
`|Δp| ≤ s_i`. Prevents runaway from strong forces. Safety harness for stronger
force laws (e.g. `StrongSpacingForce`).

## 3. Combined strategy (updated 2026-06-09)

Current pipeline:
```
1. Octree :random fill → Poisson initial cloud
2. repel(cloud, spacing, octree; kick_after=10) → blue-noise relaxed cloud
3. cull_ratio=0.5 (optional) → remove residual near-duplicates
```

Future optimal pipeline:
```
1. Octree-grid seeding (or Bridson) → blue-noise initial cloud
2. Near-duplicate pre-cull → remove pathological close pairs
3. Coarse repel (k=8, adaptive α, momentum) → bulk equilibration
4. Fine repel (k=21, adaptive α, kick_after) → local polish
5. Force-norm convergence check → stop when truly equilibrated
```

Expected total iterations: ~30-50 (vs current 200-400).
Expected wall-clock improvement: ~5-10×.

## 4. Tier status (updated 2026-06-09)

**Tier A — quality (DONE):**

| # | Change | Status |
|---|--------|--------|
| A1 | Adaptive per-point step α | ✅ Done |
| A2 | Short-range-core force law | ❌ Tested (`StrongSpacingForce` γ=3,4) — makes standoffs *worse*. Not the right lever. |
| A3 | Spacing-fidelity metric (d_NN/h, CV, p05/p95, coordination) | ✅ Done (`spacing_fidelity_metrics`) |
| A4 | Near-duplicate cull (`cull_ratio`) | ✅ Done (guarded, default off) |
| A5 | Stochastic frozen-pair kick (`kick_after`) | ✅ Done — **the key fix** |

**Tier B — convergence hygiene (DONE):**

| # | Change | Status |
|---|--------|--------|
| B1 | Force-norm stopping criterion | ✅ Done |
| B2 | Per-pair tracer (`_trace_closest_pair!`) | ✅ Done |
| B3 | Displacement cap (`|Δp| ≤ s`) | ✅ Done |

**Tier C — speed (DEFERRED):**

| # | Change | Status |
|---|--------|--------|
| C1 | Octree-grid seeding / `:jittered` default | Deferred |
| C2 | Bridson Poisson-disk (`:bridson`) | Deferred |
| C3 | Momentum / over-relaxation | Deferred |
| C4 | Multi-grid repel | Deferred |
| C5 | Octree-based NN search | Deferred |
| C6 | CVT pre-relaxation | Deferred |

## 5. Open questions

1. **Coordination 18.6 vs target 12–14.** Is this a convergence issue (need more
   iters), an initial-placement issue (Poisson start is too dense), or is the 1.4h
   threshold too generous?
2. **Can we reach CV < 0.10?** Currently 0.14. Might need better initial placement
   (Bridson) or more repel iterations.
3. **Wall-clock for shape-opt loop.** The current 300-iters repel takes ~30s on
   11k points. Needs to be <5s to live inside the optimization loop. Tier C items
   (octree NN, momentum, multi-grid) are the path.
