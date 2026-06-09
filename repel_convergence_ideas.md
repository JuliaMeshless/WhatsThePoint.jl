# Repel convergence & initial placement — improvement ideas

## 0. Evidence & re-tiering (2026-06-07)

A staged diagnostic (`/tmp/wtp_repel_diag.jl`, cavity Δ=0.08, 11450 pts) settled the
key question — **is mesh_ratio 15 a convergence problem or an equilibrium problem?**

| cum. iters | sep/Δ | fill/Δ | mesh_ratio | residual |
|---|---|---|---|---|
| 0 (raw)    | 0.009 | 1.07 | 117  | —     |
| 100        | **0.0550** | 0.96 | 17.5 | 0.052 |
| 300        | **0.0550** | 0.89 | 16.1 | 0.039 |
| 600        | **0.0550** | 0.89 | 16.1 | 0.79  |
| 1000       | **0.0550** | 0.88 | 16.0 | 0.039 |
| 1500       | **0.0550** | 0.92 | 16.8 | 0.035 |

**Findings that re-tier this whole document:**

1. **The bulk equilibrium is already excellent.** `fill/Δ ≈ 0.88–0.96` means even the
   worst covering gap has a neighbor at ~0.9·Δ — the interior already respects the
   prescribed spacing. So node *quality* is not a convergence-budget problem.
2. **`separation` is a frozen equilibrium defect, not iteration starvation.** `sep/Δ`
   sits at **exactly 0.0550 for 1400 iterations** and never moves. This is a stuck
   boundary pair: the repel force pushes it apart, `closest_point_on_triangle`
   (deterministic) snaps both points back to the same shared edge/vertex, standoff.
3. **`mesh_ratio` is the wrong gate.** 16 = fill/sep = 0.9/0.055 is ~entirely the one
   stuck pair — a brittle min/max ratio dominated by a handful of outliers.

**Consequence — separate the levers by what they actually move:**

- **Quality levers (raise/realize the equilibrium ceiling — the "best placement" goal):**
  fixing the **boundary-projection stuck pair** (§2e near-duplicate cull + a projection
  that doesn't re-snap collapsed pairs). This is the *only* thing on the list that
  changes the final node positions here. **Do this first.**
- **Measurement (so the campaign is trustworthy):** **spacing fidelity** = distribution
  of `d_NN/h` (mean + **CV**), plus coordination number to confirm blue-noise vs
  lattice; and **force-norm stopping (§2b)** + **adaptive per-point step (§2a)** so
  convergence is monotone and measurable. Cheap, do alongside.
- **Speed levers (reach the *same* equilibrium faster — DEFER to the wall-clock phase):**
  all of §1 initial placement (Bridson/CVT/octree-seeding), §2c momentum, §2d
  multi-grid, and the octree-based NN search. With true convergence these do **not**
  change the result — only iteration count / per-iteration cost.

**Target definition (for the record):** the achievable optimum is a *maximally-random-
jammed blue-noise packing* at the prescribed density — `d_NN/h` tightly peaked (low CV),
~12–14 first-shell neighbors (3D dense-liquid coordination; the 3D kissing number is 12,
**not** the 2D hexagon's 6), no voids, well-conditioned stencils — **not** a crystal. A
perfect FCC/simple-cubic lattice is both unreachable on a curved/bounded domain and
*worse* conditioned for RBF-FD (symmetry degenerates the polynomial Vandermonde).

### Locator finding (2026-06-07, later — REVISES the root-cause attribution)

Concern raised: the near-duplicate **cull** (added this session as `cull_ratio`, Tier A1)
is a *symptom* treatment. If the boundary projection has an attractor (e.g. a high-valence
mesh vertex / a lat-long pole), points could funnel into it and the cull would destroy
them one after another — a silent drain. So we located *where* the stuck points actually
are (`/tmp/wtp_locate_stuck.jl`, cavity, 300-iter repel, **no cull**):

- **91 pairs with NN < 0.25·Δ; worst pair at 0.061·Δ.**
- **All of them are VOLUME points** (`is_bnd = no`), scattered through the interior
  (radii 0.48–0.99), **not** on the boundary.
- **Pole accumulation ≈ 0** — essentially no points pile at the lat-long poles.

This **refutes two hypotheses**:
1. The lat-long pole degeneracy is *not* the cause (no pole pile-up).
2. The boundary-projection snap is *not* the current cause either — the original
   "4 coincident boundary points" defect was real *before* `_safe_direction`, but after
   the live-graph + `_safe_direction` fixes the residual close pairs are **interior /
   volume**, where there is no triangle projection at all (volume points are only
   `isinside`-filtered).

**Revised root cause (candidate, to confirm):** the residual close pairs are a
**volume-repel dynamics** problem — either a *balanced standoff* (a symmetric local
config where neighbor forces net to ~zero, consistent with the staged run's separation
frozen at *exactly* 0.0550·Δ for 1500 iters) or a *fixed-α overshoot limit-cycle* (a
high-force near-pair at r/s≈0.06 gets a large `s·α·F` step, overshoots, re-approaches).
Either way the lever is the **force/step dynamics**, *not* the boundary projection:
- **Adaptive per-point step (§2a / Tier B3) is promoted to the primary quality fix** — it
  removes the overshoot mode.
- If pairs are a genuine balanced standoff, a **short-range-core force law** (force → ∞ as
  r → 0 fast enough that no neighbor sum can balance it) or a small stochastic kick is the
  fix, *not* smaller steps. **Next diagnostic:** track one stuck pair's `r`, `|F|`, and
  displacement across iterations to distinguish standoff vs overshoot.
- **The boundary projection is NOT implicated by this run.** Keep any projection rework
  (old Tier A2) on hold until evidence points back to it (e.g. on a real STL with sharp
  edges).

**Cull status:** keep `cull_ratio` **off by default**, as a *guarded* safety net only.
Planned guard (not yet implemented): if a cull would remove more than a small fraction
(~1%) of points, warn/abort — a large cull is itself the signal of an upstream drain to
fix, exactly the failure mode this finding was checking for.

---

## 1. Initial placement: the biggest lever

The current `:random` placement draws Poisson-uniform points per octree leaf.
This guarantees arbitrarily close pairs AND voids — the worst possible starting
point for repel. The `:lattice` alternative is the opposite failure (axis-aligned,
anisotropic). `:jittered` (stratified) is better but still not blue-noise.

**The initial placement quality directly determines how many repel iterations
are needed.** A blue-noise initial cloud might need 10-20 fine-tuning iterations;
a Poisson cloud needs 100-400 to reach the same quality.

### 1a. Bridson Poisson-disk sampling (best option)

Algorithm (Bridson 2007): maintain an active list and a background grid with
cell size `h / √3`. Pick a random active point, generate up to 30 candidates
in its annulus `[h, 2h]`, accept the first that doesn't collide with existing
points. If none, deactivate the point. Repeat until the active list is empty.

**Properties:**
- O(N) time and space — single pass, no iteration
- Guarantees minimum separation ≥ h (the target spacing)
- Produces blue-noise distribution (Fourier spectrum has a clean gap below the
  Nyquist frequency)
- Exactly the distribution that repel converges to — so repel would only need
  to fine-tune near boundaries and spacing transitions

**Adaptation to graded spacing:** the background grid cell size must be
`h_min / √3` where `h_min` is the minimum spacing over the domain. Candidates
are accepted if `distance ≥ h(x)` where `h(x)` is the local target spacing.
This handles graded spacing (BoundaryLayerSpacing) naturally.

**Where to implement:** in `_generate_points_in_box` (octree.jl), replacing
the `:random` branch. Each leaf generates Bridson points within its bounding
box, with a shared rejection grid across leaves to prevent inter-leaf collisions.

**Expected impact:** reduces repel iterations by 5-10× for the same final quality.
The Poisson close pairs that dominate early repel work are eliminated at the source.

### 1b. Centroidal Voronoi pre-relaxation (CVT)

Instead of repelling points iteratively, compute a single Voronoi tessellation
and move each point to its cell centroid. One Lloyd relaxation step removes
~80% of the Poisson clustering. Two steps remove ~95%.

**Pros:** deterministic, O(N log N) per step, very fast convergence
**Cons:** requires Voronoi tessellation (DelaunayTriangulation.jl is already a
dependency), may not handle graded spacing well, boundary treatment is tricky

**Use case:** as a fast pre-pass before repel, not a replacement. One Lloyd
step + 20 repel iterations should match 200 repel iterations from Poisson.

### 1c. Octree-grid seeding

Use the octree grid itself as the placement grid: place one point per leaf at
a jittered position (center + random offset within the cell). This guarantees
minimum separation ≈ leaf size (which is calibrated to the local spacing).

**Pros:** trivial to implement, deterministic spacing guarantee, no additional
data structures
**Cons:** produces a slightly anisotropic pattern (octree grid has preferred
directions), not true blue noise

**Use case:** as a replacement for `:random` that's strictly better. The
anisotropy is mild and repel will smooth it out in far fewer iterations than
it takes to fix Poisson close pairs.

### 1d. Recommendation

Implement in order of impact/effort:
1. **Octree-grid seeding** (replace `:random` default) — trivial change, big win
2. **Bridson Poisson-disk** (new `:bridson` placement mode) — best quality, moderate effort
3. **CVT pre-relaxation** (optional pre-pass) — diminishing returns if Bridson is done

## 2. Convergence acceleration

### 2a. Adaptive per-point step size

The current fixed `α = 0.05 * min(spacing)` is too aggressive for some points
(causing oscillation) and too conservative for others (slow crawl).

Replace with a per-point step:
```
α_i = clamp(s_i / (|F_i| + ε), α_min, α_max)
```

Points with strong forces (near-duplicates, voids) take the maximum safe step;
points near equilibrium take tiny steps. This is a per-point line search that
kills oscillation without slowing down the fast movers.

**Expected impact:** 2-3× fewer iterations, elimination of the oscillation
pattern visible in the convergence history (0.18 → 0.31 → 0.13 → 0.15).

### 2b. Force-norm stopping criterion

The current convergence metric watches displacement `max |Δp_i| / s_i`. This
can be small while forces are still large (two points oscillating around
equilibrium with small amplitude but large restoring forces).

Switch to force-norm monitoring:
```
convergence = max_i (|F_i| * s_i)
```

At equilibrium, forces vanish. This detects true convergence earlier and avoids
the false convergence signals from oscillating points.

### 2c. Over-relaxation / momentum

The Jacobi-style sweep (compute all forces from snapshot, then move) is stable
but slow. Adding momentum:
```
v_i ← γ * v_i + α * F_i
p_i += v_i
```

with γ ≈ 0.5-0.8 accelerates convergence by ~2-3× for smooth interior modes.
The boundary projection naturally damps boundary oscillation.

**Danger:** momentum can cause divergence if α is too large. Start with γ = 0.5
and increase if convergence is monotone.

### 2d. Multi-grid repel

Start with coarse parameters (k=8, large α, few iterations) to get bulk density
right, then refine (k=21, smaller α, more iterations) for final polish.

- **Coarse pass:** k=8, α=0.1, 50 iterations — removes large-scale density
  gradients, cheap per-iteration (small k)
- **Fine pass:** k=21, α=0.02, 50 iterations — polishes local regularity

Total cost: ~100 iterations but the coarse pass is ~3× cheaper per iteration
than the fine pass, so wall-clock is ~60% of a single k=21 pass for 100 iters.

### 2e. Near-duplicate pre-cull

The `_safe_direction` fix handles the NaN singularity, but near-duplicates
(r ≈ 0.001·Δ) still take many iterations to separate because the distance is
tiny and the force is finite. A pre-pass that merges points closer than 0.1·Δ
removes these pathological cases before repel starts.

The octree naturally groups near-duplicates in the same leaf — a simple scan
of each leaf's element list detects them in O(N).

## 3. Combined strategy

The optimal pipeline would be:

```
1. Octree-grid seeding (or Bridson) → blue-noise initial cloud
2. Near-duplicate pre-cull → remove pathological close pairs
3. Coarse repel (k=8, adaptive α, momentum) → bulk equilibration
4. Fine repel (k=21, adaptive α) → local polish
5. Force-norm convergence check → stop when truly equilibrated
```

Expected total iterations: ~30-50 (vs current 200-400).
Expected wall-clock improvement: ~5-10×.

## 4. Priority order (re-tiered per §0 evidence)

**Tier A — quality (changes the final node positions; do now). Re-ordered per the
Locator finding: the residual close pairs are VOLUME points, so the fix is in the
force/step dynamics, not the boundary projection.**

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| A1 | Adaptive per-point step α (§2a) — removes the volume overshoot/limit-cycle | Low | **High — the promoted primary quality fix** |
| A2 | Short-range-core force law (if pairs are a balanced standoff, not overshoot) | Med | High — conditional on the next diagnostic |
| A3 | Spacing-fidelity metric (`d_NN/h` mean + CV, coordination) — the right gate | Low | High — done this session (in `validate_cavity.jl`) |
| A4 | Near-duplicate cull (§2e, `cull_ratio`) — **guarded safety net only, default off** | Low | Low — symptom treatment; a large cull = upstream drain to fix |
| — | Boundary projection rework (old A2) — **ON HOLD**, not implicated by the Locator run | — | revisit only if a real STL with sharp edges shows boundary pile-up |

**Tier B — convergence hygiene (do alongside A, cheap):**

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| B1 | Force-norm stopping criterion (§2b) | Low | Medium — true convergence, no false stops |
| B2 | Per-pair force/displacement tracer (next diagnostic: standoff vs overshoot) | Low | High — decides A1 vs A2 |

**Tier C — speed only (same equilibrium, faster; DEFER to wall-clock phase):**

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| C1 | Octree-grid seeding / `:jittered` default (§1c) | Low | Speed — fewer iters, no quality change |
| C2 | Bridson Poisson-disk (`:bridson`, §1a) | Med | Speed — best init, fewer iters |
| C3 | Momentum / over-relaxation (§2c) | Med | Speed — 2-3× faster smooth modes |
| C4 | Multi-grid repel (§2d) | Med | Speed — fewer total iters |
| C5 | Octree-based NN search | Med | Speed — cheaper per-iteration (see `plan_octree_nn_search.md`) |
| C6 | CVT pre-relaxation (§1b) | High | Speed — low marginal value if Bridson done |

Rationale: §0 showed the bulk is already converged and the only quality defect is the
frozen stuck pair, so Tier A is the lever for "best placement." Tier C items only buy
iteration-count / per-iteration speed and belong in the explicitly-deferred wall-clock
optimization phase — they will not move the spacing-fidelity numbers.
