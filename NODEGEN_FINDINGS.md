# Node generation — findings & roadmap

Consolidated 2026-06-11 from `NODEGEN_FINDINGS.md`, `cavity_validation_findings.md`,
`octree_nn_assessment.md`, `plan_octree_nn_search.md`, and
`repel_convergence_ideas.md` (all retired; full text in git history).

## Goal

Perfect WhatsThePoint node generation (Octree fill + repel refinement) for
downstream RBF-FD meshless solving (Macchiato shape-opt). Strategic target:
bring repel *inside* the shape-optimization loop — boundary-point count and
connectivity change every design step, so the cloud must re-relax fast and
predictably. The gating requirement is **convergence speed and robustness**,
not just final quality.

Validation target: annular cavity (R=1, r=0.547), judged by
`validate_cavity.jl` (separation, spacing CV, coordination, per-stencil
degree-3 Vandermonde σ_min/σ_max).

## Current status (2026-06-09 gate, branch `shape_optimization_utils`)

Quality gate: **PASS** (cavity, Δ=0.08, 11450 pts, 300 iters):

| Metric | value | target |
|---|---|---|
| separation / Δ | 0.500 | > 0.1 ✓ |
| spacing CV (d_NN/h) | 0.140 | < 0.15 ✓ |
| p05 / p95 (d_NN/h) | 0.541 / 0.863 | — |
| coordination (≤1.4h) | 18.6 | 12–14 ✗ |
| singular stencils (k=50, deg 3) | 0 / 11403 | 0 ✓ |
| min / median σ_min/σ_max | 3.8e-4 / 1.49e-2 | — |

Speed (re-measured 2026-06-11, after the `knn!` buffer fix): **0.47 s /
10 iters at 45.7k pts on 8 threads, 0.096 GB allocated** (was 0.53 s /
0.758 GB; single-threaded 2.66 s). ~300 iterations needed from a `:random`
start — iteration count is now the dominant cost, and CV relaxation is what
sets it (see Session 1 findings).

**Honest SOTA position:**
- *Quality*: competitive on the internal gate, but no head-to-head vs the
  reference direct generators (Medusa / Slak–Kosec PNP) has been run.
- *Speed*: behind direct generation — PNP gets sep ≥ 0.5·h *by construction*
  in one O(N log N) pass; we need ~300 iterations + kick.
- *Differentiator*: incremental re-relaxation in the shape-opt loop +
  deposition (emergent boundary sampling; no published equivalent known).

## Architecture verdict

The Octree→repel split is the right decomposition:
- **Octree** (`src/discretization/algorithms/octree.jl`): graded *density*
  (node count follows `h(x)`), arbitrary STL. Placement options: `:random`
  (Poisson per leaf, default), `:jittered` (stratified grid per leaf),
  `:lattice`.
- **Repel** (`src/repel.jl`): local *regularity* (blue-noise spacing) — what
  conditions RBF-FD stencils and removes close-pair `SingularException`s.

## Quality indicators

`mesh_ratio` (fill/separation) was dropped as a primary metric — a min/max
ratio dominated by a handful of outliers. Primary metrics
(`spacing_fidelity_metrics` in `src/metrics.jl`):

| Indicator | Definition | Target |
|---|---|---|
| separation/Δ | min(d_NN) / Δ | > 0.1 |
| spacing CV | std(d_NN/h) / mean(d_NN/h) | < 0.15 |
| p05 / p95 (d_NN/h) | spacing-fidelity percentiles | — |
| coordination | mean neighbors within 1.4h | 12–14 |
| singular stencils | count of σ_min/σ_max < 1e-8 | 0 |

## Established findings

Defects found and fixed (all in `src/repel.jl`):
1. **Frozen neighbor graph** — kd-tree built once. Fixed: `rebuild_every=1`.
2. **Wrong default force** — `InverseDistanceForce` has no root, cloud drifts.
   Fixed: `SpacingEquilibriumForce` default.
3. **Coincident-point NaN** — `0/0` direction at r=0; point sticks forever.
   Fixed: `_safe_direction` returns a random unit vector.
4. **Balanced standoff** (the key fix, sep 0.14 → 0.50·Δ) — the closest pair
   freezes at a fixed `r/s` for hundreds of iterations; neighbor forces cancel
   exactly. Only a stochastic perturbation breaks it: `kick_after` kicks the
   volume point 0.1·s after N frozen iterations. Locator evidence: all stuck
   pairs were interior **volume** points — not boundary, not poles.

Negative results and principles (don't re-derive these):
- **More iterations don't help.** 1500-iter staged diagnostic: separation
  frozen at exactly 0.0550·Δ from iter 100 to 1500 while bulk fill was already
  0.88–0.96·Δ. Residual defects are *equilibrium* defects, not iteration
  starvation — the levers are dynamics (step/kick) and seeding, not budget.
- **`StrongSpacingForce` (γ=3,4) makes standoffs worse** — a stronger core
  moves the equilibrium closer, not farther. Kept as an option, not default.
- **Cull = defect signal** (design principle, Davide): the `cull_ratio` safety
  net should NEVER fire in healthy generation; both `repel` methods `@warn`
  when it removes anything. Activation means an upstream generation bug.
- **Check spacing vs geometry resolution before trusting a failing test** —
  box.stl is a mm-authored STL read as meters; a prescribed h 24× the boundary
  tessellation once sent a session chasing the wrong bug.
- **Deposition works and is pressure-driven** (`deposit_ratio`, octree method):
  escaped volume points project to the nearest triangle and convert to
  boundary points, accepted only when no boundary point is within
  `deposit_ratio·s` (self-limiting). Sparse box: 234 → 408 boundary points;
  an under-filled volume deposits nothing.

Open defects:
1. **Boundary-projection collisions** — the cavity culls ~42–59 points/run
   (projection parks pairs on shared edges/vertices). The one thing keeping
   the cull warning loud.
2. **Coordination 18.6 vs 12–14** — cloud denser than ideal blue-noise;
   likely a seeding artifact (Poisson start), possibly a too-generous 1.4h
   threshold.
3. **Float32 STL → Octree Float64 CRS mismatch** — workaround: promote mesh.
   Worth fixing in `octree.jl`.
4. **Deposition validated on one 834-pt box only** — curved geometry,
   variable h, boundary-free start untested.

## Roadmap (assessed 2026-06-11)

Framing: total cost = (iterations to converge) × (cost per iteration). The
factors multiply and have independent levers:

| Lever | Attacks | Expected gain | Effort to first result |
|---|---|---|---|
| `:jittered` validation | iteration count | **measured 2026-06-11: none** | done |
| Bridson graded Poisson-disk | iteration count | 300 → ~30–80 iters | ~1 session |
| k-NN buffer reuse (`knn!`) | cost/iter | **measured: −87% alloc, GC 8.8→0.7%, −11% threaded wall** | done |
| `rebuild_every` > 1 | cost/iter | **measured: ~5% — not a lever** (rebuild is 3% of loop) | done |
| Octree NN search | cost/iter | ≤ ~25% of loop (query-locality only; rebuild premise measured void) | 1–2 sessions (design done) |
| Momentum | iteration count | ~2–3× | hours (cap guards it) |

### Session 1 — cheap experiments, high information

0. ✅ **Done 2026-06-11.** STL-based gate: cavity exported to
   `test/data/cavity.stl` (4608 binary facets), `validate_cavity.jl` now
   imports it via GeoIO with the Float64-promotion workaround and grew
   `--placement=`/`--save-vtk` flags plus per-run iteration/wall-clock
   reporting. Gate reproduces the baseline exactly (PASS, sep/Δ=0.500,
   CV=0.139, coord 18.6, 0 singular, 56 culled) — the STL path changes
   nothing. Runs were single-threaded (16 cores idle): wall-clock numbers
   below are 1-thread.
1. ✅ **Done 2026-06-11.** `:jittered` vs `:random`: **no end-to-end gain** —
   final clouds statistically identical (sep 0.500 both, CV 0.140 vs 0.139,
   coord 18.5 vs 18.6, 57 vs 56 culled), both burn all 300 iters. Staged runs
   (gate metrics at cum. 25/50/100/200/300 iters) showed why:
   - **CV is the binding gate metric.** Separation clears 0.1·Δ within 25
     iters from either start; the rest of the budget grinds the CV tail
     (0.18 → 0.15 takes ~175 iters).
   - **CV drops 0.42 → 0.18 in the first 25 iterations** (the cliff), then
     crawls. Per-leaf jitter has no cross-leaf separation constraint, so its
     raw CV (~0.40) is no better than Poisson — it enters the same tail.
   - Sharpened Bridson bet: a seeding only pays if its *raw* CV lands below
     ~0.18, i.e. below the post-cliff level — true blue-noise, not stratified
     jitter.
2. ✅ **Done 2026-06-11.** Profile (45.7k pts, 10 iters, boundary-projected):
   total 2.89 s / 0.757 GB single-threaded. Breakdown: **kd rebuild 2.6% of
   time / 4% of alloc** — the "per-iteration rebuild cost" assumption was
   wrong; `searchdists` queries were 48% of time / 66% of alloc. Fix landed
   in `src/repel.jl`: raw `KDTree` + in-place `knn!` into `TaskLocalValue`
   buffers (also drops the `norm.()` temporary; `_deposit_escaped!` queries
   the same tree). Result: **0.757 → 0.095 GB (−87%)**; wall-clock −8%
   single-threaded, and on 8 threads **0.53 → 0.47 s with GC 8.8% → 0.7%**.
   Threading scales 5.7× on 8 threads. Remaining loop time is the kd
   traversal itself (~50%) + force/projection (~50%). Verified: full
   `Pkg.test()` green (142,791 pass) and the cavity gate reproduces on the
   new code (PASS, sep/Δ=0.500, CV=0.142, coord 18.6, 0 singular, 56 culled).
3. ✅ **Done 2026-06-11.** `rebuild_every ∈ {1,2,3,5}`, same start, 300 iters:
   gate quality completely insensitive (sep 0.500, CV 0.139–0.140, PASS all),
   but only ~5% faster at 5 — consistent with the rebuild being 3% of loop
   time. Keep default 1 (free freshness); the knob is not a speed lever.

Exit criterion met. Consequences for later sessions:
- **Octree NN search (Session 4) is demoted further**: its O(N)-rebuild
  premise attacks a measured 3% cost. Its only remaining upside is query
  locality (26-neighbor direct lookup vs kd traversal), bounded at ~2× of the
  ~50% query share ≈ 25% of the loop. Build it only if the shape-opt loop's
  measured budget demands it after Bridson + momentum.
- Iteration count (Bridson, momentum) is now confirmed as the dominant
  remaining lever, exactly as the staged-gate data implied.

### Session 2 — `:bridson` graded Poisson-disk seeding

Bridson with graded h is essentially the reference PNP algorithm, so it pays
twice:
- A start that satisfies the separation gate *by construction*; repel reduces
  to a short CV/coordination polish. The most plausible path to coordination
  12–14 and the CV < 0.10 stretch goal (iterations demonstrably don't fix CV).
- An in-house PNP-equivalent **baseline for the SOTA benchmark** — no need to
  stand up Medusa (C++) first; keep it as stretch validation.

Implementation notes: O(N); min separation ≥ h by construction; adapt to
graded spacing with an `h_min/√3` background grid. (CVT one-Lloyd-step
pre-relaxation removes ~80% of Poisson clustering but has diminishing returns
once Bridson exists.)

Acceptance criterion from the Session-1 staged data: the seeding only pays if
its **raw CV lands below ~0.18** (the post-cliff level) — true blue-noise
does, stratified jitter doesn't. Measure raw `spacing_fidelity_metrics`
before any repel as the first checkpoint.

### Session 3 — benchmark harness + cull silence

- Harness: Bridson-direct (≈ PNP) vs full pipeline on identical geometries;
  all gate metrics + wall-clock. Turns "SOTA" from belief into a number.
  Geometry ladder: `cavity.stl` (calibrated baseline) → `box.stl` (sharp
  edges/flat faces) → `bifurcation.stl` (thin curved branches, realism — the
  generality claim).
- Fix boundary-projection collisions (tangential separation at the landing
  site, or an acceptance-style check against existing boundary points) →
  cull permanently silent.

### Session 4 (conditional) — octree NN search + momentum

Trigger (updated after Session 1): the original premise — O(N) rebuild vs
O(N log N) — is void; the rebuild is a measured 3% of loop time. The only
remaining upside is query locality (kd traversal is ~50% of the loop), so
build it only if the shape-opt loop's measured budget still demands ≤25%
more speed after Bridson + momentum land. Design summary (full plan retired
to git history, `plan_octree_nn_search.md`):

- **Option A**: `OctreeNN` wrapper around the existing `SpatialOctree`
  implementing the `searchdists` interface — repel logic unchanged, pure
  performance change, node positions identical.
- Steps: (1) extend `find_neighbor` from 6 face directions to all 26
  (`(di,dj,dk) ∈ {-1,0,1}³`, trivial in the integer coordinate system);
  (2) `OctreeNN` + `searchdists` + in-place `rebuild!` (clear
  `element_lists`, re-insert each point — O(N), no allocation);
  (3) benchmark vs kd-tree on the cavity; (4) `search_method` kwarg in
  `repel`.
- Key risk: too few points per leaf if leaf ≈ Δ → use a coarser NN octree
  (leaf ≈ 2–3Δ gives 8–27 points/leaf, enough for k=21 from the 3×3×3
  neighborhood).

Momentum as a cheap add-on: `v_i ← γ·v_i + α·F_i`, γ ≈ 0.5–0.8, ~2–3× on
smooth modes; the displacement cap is the divergence guard. Multi-grid
(coarse k=8 pass → fine k=21, ≈60% wall-clock of a single fine pass) only if
still needed after the above.

### After the speed push — deposition (the novelty)

1. Cavity with decimated boundary (~h or sparser) + `deposit_ratio=0.5`;
   success = gate PASS **and cull silent**.
2. Boundary-free entry point: discretize from `TriangleOctree` + spacing
   alone, deposition grows the boundary from nothing. Needs
   empty-`PointBoundary` plumbing.

### Target end state

Seeded start (jittered/Bridson) → coarse repel (k=8, momentum) → fine repel
(k=21, `kick_after`) → force-norm stop. Expected ~30–50 total iterations
(vs current 200–400), ~5–10× wall-clock. Stretch: spacing CV < 0.10
(currently 0.14).

## Validation assets

- `validate_cavity.jl` (repo root): annular cavity through
  Octree→repel→kick→cull. PASS as of 2026-06-10 (sep/Δ=0.500, CV=0.140,
  0 singular). Rerun: `jlrun validate_cavity.jl [Δ]`
- `validate_repel.jl` (repo root): box.stl frozen-vs-live comparison.
- Full `Pkg.test()` green (142k+ assertions) as of 2026-06-10.

## Housekeeping backlog

- Update `CLAUDE.md` repel docs (`α_min`, `kick_after`, `cull_ratio` warn,
  `deposit_ratio`, `trace`).
- Float32/Float64 type-mismatch fix in `octree.jl`.
