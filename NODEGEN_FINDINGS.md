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

Scale expectation (Davide, 2026-06-11): production runs will generate **many
millions of volume points** in real geometries — the reference case is
`examples/octree_boundary_layer.jl`: the Octree algorithm on `bunny.stl`
(≈70k-facet STL, ~86 m across) with `BoundaryLayerSpacing` (0.8 m wall /
4.0 m bulk) at `max_points = 600_000`, i.e. graded spacing at large N
(currently `:jittered` placement). Algorithmic choices must be judged at
10⁶–10⁷ points with boundary-layer grading, not at the 10⁴-point
constant-spacing gate.

## ⚠ Geometry corruption notice (discovered 2026-06-11, Session 2)

**Every cavity-gate number recorded before 2026-06-11 PM was measured on a
corrupt geometry.** `_make_sphere_mesh` in `validate_cavity.jl` built its
point list with a column-major comprehension while the connectivity assumed
row-major indexing: the resulting "cavity" had a wrinkled non-watertight
surface, a missing azimuthal wedge, 608 chord triangles through the interior
(6× the true surface area), and an `isinside` that misclassified ~30% of the
annulus. The bug is as old as the gate itself (first commit of
`validate_cavity.jl`, 2026-06-09, verified via `git log -S`), so **all**
historical cavity numbers — including the original PASS baseline and the
repel-fix improvements — were measured on the corrupt domain. It went
undetected because **every gate metric is d_NN-based (separation, spacing
CV, coordination, stencil σ) — all local measures, structurally blind to
coverage voids and wrong domains.** The
"STL gate reproduces the programmatic baseline" check was self-consistent
garbage (both sides used the same scrambled generator), and the one-time
roundtrip check compared face centers — which survive scrambling.

Caught only when the surface Poisson-disk sampler placed 784 samples
off-shell (it samples the actual surface; face centers of chord triangles had
just blended in as interior junk). Fixes: generator index order corrected,
degenerate pole slivers no longer emitted (they failed the manifold
orientation check and planted near-duplicate face centers at the poles),
`cavity.stl` regenerated (4416 facets, area 16.268 vs analytic 16.326), and
the gate now asserts total mesh area against the analytic value at export
*and* after re-import. **Lesson (reinforces the existing principle): validate
the geometry itself (area, isinside probes) before trusting distribution
metrics — d_NN statistics cannot see a broken domain.**

## Current status (2026-06-11 clean-geometry gate, branch `shape_optimization_utils`)

Cavity gate, Δ=0.08, **1 thread**, clean `cavity.stl` (4416 facets):

| Pipeline | verdict | CV (post-cull) | sep/Δ | singular | culled | repel iters |
|---|---|---|---|---|---|---|
| `:random` + face centers (old default) | MARGINAL | 0.151 | 0.500 | 0 | **626** | 300 |
| `:bridson` + face centers | MARGINAL | 0.151 | 0.500 | 0 | 589 | 300 |
| **`:bridson` + Poisson-disk boundary** | **PASS, raw** | **0.071** | **0.750** | **0** | **0** | **0** |

*(Updated 2026-06-11 Session 3, new `ClippedSpacingForce` repel default: the
seeded path after 300 repel iters is now PASS CV 0.044 / sep 0.694 / 0 culled
— repel adds quality on top of the raw seed; the `:random` legacy path
measures MARGINAL CV 0.184 / 650 culled under the new default — see Session 3
verification notes.)*

The direct-generation pipeline (`sample_surface` boundary + `:bridson` volume)
satisfies the full gate **by construction in ~3 s single-threaded, zero repel
iterations**, with the cull warning silent for the first time. The previous
"PASS, CV 0.140, 56 culled" baseline was an artifact of the corrupt domain;
on the true cavity the repel-from-random pipeline never passes pre-cull and
leaves 626 near-duplicates.

Repel speed (measured 2026-06-11 on the corrupt geometry, but a
geometry-independent code property): 0.47 s / 10 iters at 45.7k pts on 8
threads, 0.096 GB allocated (single-threaded 2.66 s) after the `knn!` buffer
fix.

**Honest SOTA position (updated 2026-06-11):**
- *Quality*: the direct pipeline now meets the gate by construction
  (sep ≥ 0.75·r, CV 0.07, 0 singular raw) — same construction class as the
  reference direct generators (Medusa / Slak–Kosec PNP); a head-to-head on
  identical geometries is still unrun (Session 3).
- *Speed*: in the direct-generation class now — one O(N) advancing-front pass
  per stage (~3 s total on the cavity, 1 thread); the 300-iteration repel
  budget is gone for static geometry.
- *Differentiator*: incremental re-relaxation in the shape-opt loop +
  deposition (emergent boundary sampling; no published equivalent known) —
  both gated on fixing repel's blue-noise degradation (open defect 2).

## Architecture verdict (revised 2026-06-11)

Generation is now a two-stage *direct* pipeline; repel is an optional
relaxation tool, no longer the quality workhorse:
- **Boundary**: `sample_surface(mesh, spacing)` / `PointBoundary(mesh, spacing)`
  (`src/surface_sampling.jl`) — graded surface Poisson-disk; supersedes
  face-center import for quality-sensitive use.
- **Octree** (`src/discretization/algorithms/octree.jl`): graded *density*
  (node count follows `h(x)`), arbitrary STL. Placement options: `:random`
  (Poisson per leaf, default), `:jittered` (stratified grid per leaf),
  `:lattice`, `:bridson` (global graded Poisson-disk, Session 2 — the only
  mode with a cross-leaf separation guarantee; `bridson_factor` sets the disk
  radius relative to `h`, default 0.75). With a Poisson-disk boundary,
  `:bridson` output passes the gate raw.
- **Repel** (`src/repel.jl`): retained for incremental re-relaxation under
  boundary change (shape-opt) and deposition. Measured 2026-06-11: it
  *degrades* an already-blue-noise cloud (open defect 2), so it should not
  follow the direct pipeline on static geometry.

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

*(Caveat added 2026-06-11: items below were measured on the corrupt cavity.
The repel defects and their fixes are real dynamics mechanisms — frozen graph,
force root, NaN direction, standoff — and remain in place, but their quoted
metric improvements (e.g. sep 0.14 → 0.50) await clean re-measurement if they
ever matter again; with the direct pipeline they mostly don't.)*

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
1. **Imported face-center boundaries are themselves a defect** (resolved for
   generation by `sample_surface`, still open for face-center users): on the
   clean cavity the tessellation's pole rings (min d_NN/h = 0.078) and uneven
   face spacing floor the boundary CV at ~0.24 after 300 repel iterations and
   leave 589–626 culled near-duplicates. The Poisson-disk boundary measures
   CV 0.061 raw and culls zero. Boundary-projection collisions during repel
   (the original defect framing) remain plausible contributors when repel is
   used, but the dominant term was the import sampling all along.
2. **Repel degrades constructed blue-noise** — **RESOLVED 2026-06-11 Session 3**
   (see "Session 3 — repel quality refinement" below). Root cause: the
   attractive branch of `SpacingEquilibriumForce` is a condensation
   instability. Fix: `ClippedSpacingForce` (repulsion-only, compact support)
   is the new `repel` default; 300 iters from the raw-PASS cloud now
   *improve* CV 0.072 → 0.044 with sep/Δ 0.728. Verified: seeded gate PASS
   improved (CV 0.044, cull silent), `Pkg.test()` green, legacy `:random`
   path mildly worse but accepted (see Session 3 notes).
3. **Coordination**: with the clean geometry and the resampled pipeline,
   coordination is 10.5 raw / 14.3 after repel — at or near the 12–14 ideal
   band; the historical 18.6 was a corrupt-domain artifact. Consider the
   defect closed pending re-measurement on other geometries.
4. **Float32 STL → Octree Float64 CRS mismatch** — workaround: promote mesh.
   Worth fixing in `octree.jl`.
5. **Deposition validated on one 834-pt box only** — curved geometry,
   variable h, boundary-free start untested.

## Roadmap (assessed 2026-06-11)

### Planned session sequence (Davide, 2026-06-11)

The next three sessions, in order:

1. ✅ **Done 2026-06-11 (Session 3, code landed; gate/Pkg.test re-runs
   pending — see Session 3 notes).** **Refine repel for quality** — make repel produce *higher* quality than it
   does today; concretely, resolve open defect 2 (repel currently degrades a
   constructed blue-noise cloud: CV 0.071 → 0.093, sep 0.75 → 0.55 over 300
   iters; residual plateaus ~0.2 without converging). Levers identified: a
   force model whose equilibrium matches Poisson-disk statistics, step/decay
   schedule, and a principled stopping criterion. Repel's mandate is the
   shape-opt inner loop, so "quality" means: applied to an already-good
   cloud, it must improve or at worst preserve it.
2. **Octree gap-tracking sampler acceleration, tested against `bunny.stl`** —
   build the design recorded below ("Sampler scaling — octree gap tracking")
   and validate on the production rung: the `octree_boundary_layer.jl`
   configuration pushed to ≥10⁶ points, graded spacing, stated thread count.
3. **Remove the proven-inferior generation strategies** in favour of the
   direct pipeline (Poisson-disk surface placement + `:bridson` volume).
   Davide's assessment: if the octree pipeline proves robust enough, the
   older algorithms are not needed at all — SlakKosec / VanDerSandeFornberg
   are "totally incapable of discretizing bunny.stl". Scope: the legacy 3D
   algorithms, and plausibly the per-leaf placements (`:jittered` measured
   useless; `:lattice` untested in anger; `:random` perhaps kept as a cheap
   debug mode). **Caveat to resolve before deleting:** `FornbergFlyer` is the
   *only* 2D algorithm and `Octree` is 3D-only — removing the legacy set
   either keeps FornbergFlyer for 2D, adds a quadtree variant, or explicitly
   drops 2D support. Removal also touches CLAUDE.md, docs, examples, and the
   test suite.

Framing for the levers below: total cost = (iterations to converge) × (cost
per iteration). The factors multiply and have independent levers:

| Lever | Attacks | Expected gain | Effort to first result |
|---|---|---|---|
| `:jittered` validation | iteration count | **measured 2026-06-11: none** | done |
| Bridson graded Poisson-disk (volume) | iteration count | **measured 2026-06-11: vol CV 0.074 raw; gate stays boundary-bound with face centers** | done |
| Surface Poisson-disk (boundary) | iteration count | **measured 2026-06-11: with `:bridson` volume the gate passes raw — iterations-to-PASS = 0** | done |
| k-NN buffer reuse (`knn!`) | cost/iter | **measured: −87% alloc, GC 8.8→0.7%, −11% threaded wall** | done |
| `rebuild_every` > 1 | cost/iter | **measured: ~5% — not a lever** (rebuild is 3% of loop) | done |
| Octree NN search | cost/iter | demoted again: static generation no longer iterates; relevant only if repel-in-loop survives | only on measured need |
| Momentum | iteration count | demoted: static pipeline needs 0 iterations; reconsider for shape-opt re-relaxation | only on measured need |
| Octree gap-tracking sampler | sampler wall-clock at production N (10⁶–10⁷ pts) | ~5–20× fewer candidates, + parallel phase groups; maximality proof (design recorded below) | 1–2 sessions, build on trigger |

### Session 1 — cheap experiments, high information

*(2026-06-11 caveat: all Session-1 gate numbers were measured on the corrupt
cavity. The profiling/allocation results (items 2–3) are geometry-independent
and stand. Item 0's "STL gate reproduces the programmatic baseline" was
self-consistent garbage — both sides shared the scrambled generator. Item 1's
jittered-vs-random null result is mechanistically geometry-independent
(per-leaf jitter has no cross-leaf constraint) and was reconfirmed in spirit
by the clean-geometry decomposition.)*

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

### Session 2 — `:bridson` volume seeding + `sample_surface` boundary seeding ✅ (2026-06-11)

Session 2 grew beyond its plan: it delivered the volume Bridson placement,
then (after the user asked how the boundary projection equilibrium works) the
surface Poisson-disk sampler — which immediately exposed the geometry
corruption (see notice at top), forced a re-baseline, and ended with the
direct-generation pipeline passing the gate raw. All numbers below:
**1 thread**, clean `cavity.stl` (4416 facets), Δ=0.08 unless noted.

**Implemented:**

- `placement = :bridson` (`src/discretization/algorithms/octree.jl`): single
  global advancing-front pass (Bridson 2007) seeded from the boundary points,
  disk radius `r(x) = bridson_factor·h(x)`, domain-spanning background bucket
  grid (cell `= r_min/√3`, linked-list buckets because boundary seeds violate
  the disk criterion among themselves). Candidates are domain-tested through
  the node-octree leaf classification (interior leaves trusted, boundary
  leaves get exact `isinside`) and separation-tested globally:
  `‖xᵢ−xⱼ‖ ≥ min(rᵢ, rⱼ)` against *all* points, boundary included, by
  construction. The front saturates at the disk-packing density, so
  `max_points` is a **cap, not a target** (warn on truncation); per-leaf
  allocation and the deficit fill are bypassed. ~1 s for 6842 points.
  Placement validation added to both `Octree` constructors (the mesh-based one
  previously validated nothing).
- `sample_surface(mesh, spacing; factor=0.75)` + `PointBoundary(mesh, spacing)`
  (`src/surface_sampling.jl`): graded Poisson-disk dart throwing on the
  continuous triangle surface (area-weighted triangle pick, uniform-in-triangle
  sample, same `min(rᵢ,rⱼ)` criterion and grid machinery). Samples carry the
  parent triangle's normal; per-point areas preserve total mesh area ∝ r².
  Continuous sampling (not face-center thinning) removes tessellation
  artifacts *and* fills regions coarser than the target spacing. ~0.6 s for
  2.8k points. Caveat noted in code: 3D-distance blocking means sub-`h` thin
  sheets would suppress one side.
- Testitems for both (separation guarantees brute-forced, constant + graded
  spacing, area preservation, normals).

**Findings:**

1. **`bridson_factor = 1.0` (strict d_NN ≥ h) is a measured dead end.** A
   saturated 3D dart-throwing front packs only η ≈ 0.39 points per `r³`
   (measured), so r = h yields ~45% fewer points than the prescribed `1/h³`.
   That starts `SpacingEquilibriumForce` in its **attractive branch**
   (`u ≈ 1.05 > 1`): repel compacts the cloud toward its denser equilibrium
   and destroys the seeding (corrupt-geometry run: CV 0.50 → 0.22 at 300
   iters, FAIL — direction of the effect is geometry-independent). Default is
   therefore **`bridson_factor = 0.75`** (≈ η^(1/3) = 0.73 density-matching
   point): the saturated front lands at the prescribed density and at repel's
   own equilibrium spacing.

2. **Volume seeding works perfectly; the gate was boundary-bound.** Clean
   staged decomposition (chained repel, kick_after=10, face-center boundary):

   | cum iters | bridson vol CV | bridson bnd CV | random vol CV | random bnd CV |
   |---|---|---|---|---|
   | 0 | 0.074 | 0.398 | 0.388 | 0.406 |
   | 25 | 0.065 | 0.294 | 0.129 | 0.286 |
   | 300 | 0.083 | 0.246 | 0.094 | 0.238 |

   The Bridson volume at **zero iterations** beats the random volume at 300.
   But with face centers the boundary CV floors at ~0.24 and the full-cloud
   gate stalls at CV ≈ 0.18 pre-cull from either start — **the entire
   iteration budget was always boundary redistribution.** Face-center culls on
   the clean geometry: 626 (random) / 589 (bridson) — the old "56 culled" was
   a corrupt-domain artifact; per the cull-is-a-defect-signal principle, the
   imported face-center boundary is itself the defect.

3. **The headline: direct generation passes the gate raw.**
   `sample_surface` boundary (2758 pts, CV 0.061) + `:bridson` volume
   (6842 pts, CV 0.075): the raw cloud measures **CV 0.071, sep/Δ 0.750,
   0 singular stencils, coordination 10.5** — full gate criteria met by
   construction in ~3 s single-threaded, **zero repel iterations**
   (baseline budget: 300 iterations and still MARGINAL). Official run:
   `validate_cavity.jl --placement=bridson --resample-boundary` → **PASS**
   (sep/Δ 0.616, CV 0.094, 0 singular, **cull removed 0** — silent for the
   first time). Iterations-to-PASS: **0**.

4. **Repel *degrades* a constructed blue-noise cloud.** From the raw-PASS
   start, 300 repel iterations drift CV 0.071 → 0.093, sep 0.75 → 0.55, while
   coordination rises 10.5 → 14.3; residual hovers at ~0.2 without
   converging. Repel's force equilibrium (k=21, SpacingEquilibriumForce) is
   simply a *worse* configuration than saturated Poisson-disk. Consequences:
   (a) for static clouds, repel after good seeding is unnecessary-to-harmful;
   (b) repel's remaining role is incremental re-relaxation under boundary
   *change* (the shape-opt loop) and deposition; (c) if sub-seeding quality
   matters there, the force model/step needs revisiting (a force whose
   equilibrium reproduces PDS statistics, or an early-stop criterion).

5. `Pkg.test()` green: **142,808 pass / 2 broken (pre-existing)**, including
   the `:bridson` and `sample_surface` testitems.

### Session 3 — repel quality refinement (2026-06-11, plan item 1) ✅ landed and verified

All numbers **1 thread**, clean `cavity.stl`, Δ=0.08, seeded cloud = 2770
Poisson-disk boundary + 6842 `:bridson` volume points (9612 total, raw CV
0.072, sep/Δ 0.750). Harness: `diagnose_repel_quality.jl` (repo root;
`preserve` / `recover` / `stallkick` experiments, deterministic seed).

**Diagnosis (preserve experiment, 300 staged iters from raw-PASS):**

| config | CV @300 | sep/Δ @300 | coord @300 | residual @300 |
|---|---|---|---|---|
| A `SpacingEquilibriumForce(0.2)` (old default) | 0.095 ↑ | 0.602 | 14.4 ↑ | ~0.21 plateau |
| B clipped repulsion-only, support u<1 | **0.044 ↓** | **0.728** | 10.1 | 0.069 ↓ |
| C attraction damped ×0.1 | 0.039 ↓ | 0.747 | 10.5 | 0.064 ↓ |
| D clipped, root u0=0.9 | 0.034 ↓ | 0.711 | 10.3 | 0.049 ↓ |
| E old default, α×0.25 | 0.073 (slower ↑) | 0.625 | 12.1 | 0.18 |

- **Root cause of defect 2: the attractive `u>1` branch is a condensation
  instability**, not noise. At prescribed density `1/h³` the force's preferred
  bond length `s` is unreachable (close-packing at bond `s` would need
  ~1.4/h³), so the cloud condenses into locally denser clusters + voids:
  coordination 10.5→14.4, CV up, sep down. E confirms: quarter step → same
  drift, quarter speed (instability marches ∝ α). C confirms dose-response
  (×0.1 attraction → ~10× slower, still improving at 300).
- **Fix chosen: B — `ClippedSpacingForce(β=0.2, u0=1.0)`** (the
  `SpacingEquilibriumForce` repulsive branch, zero for `u ≥ u0`). Equilibrium
  set contains every Poisson-disk configuration (all pairs ≥ u0·s) —
  "improve or preserve" is structural, not tuned. C scored marginally better
  CV but keeps the condensation term at 1/10 strength (long-horizon risk);
  D over-relaxes mean d_NN downward. **B is now the `repel` default** (`β`
  still feeds it; old force selectable via `force_model=SpacingEquilibriumForce(β)`).
- **Recover (all points perturbed by 0.3·h; proxy for one shape-opt step):**
  gate-PASS (sep>0.1, CV<0.15) in **10 iterations** from CV 0.241/sep 0.197;
  CV back at seed level (~0.07) by ~75–90 iters. Old default also reaches
  gate-PASS ~10–20 but stalls at CV 0.092 and re-degrades (coord climbing).
- **Stopping criterion:** force-norm `tol` is the wrong shape — a saturated
  repulsion-only packing is a frustrated glass; residual plateaus ~0.05–0.07,
  never reaches 1e-4. CV however keeps genuinely improving ~hundreds of
  iters, so a pure stall detector doesn't fire either. Landed both, computed
  free from the sweep's nn data (`_dnn_cv`): **`cv_target`** (primary; stop at
  direct-pipeline quality, ≈0.07 on the cavity — relaxing past what a re-seed
  gives is wasted budget) and **`stall_after`** (backstop, ≥0.1% improvement
  window). Defaults 0 = off (backward compatible).
- **Integrated production config** (`kick_after=10, cv_target=0.07,
  stall_after=50`, single call): raw-PASS start stops in **6 iters / 1.6 s**
  (CV 0.068 — recognized as already-good); perturbed start stops in
  **90 iters / 4.2 s** (CV 0.070, sep 0.419, gate-PASS). Repel wall-clock at
  9.6k pts, 1 thread: ~0.5 s / 10 iters.
- ⚠ Caveats recorded honestly: (a) sep is a min-statistic and dips
  transiently (e.g. 0.375 at the 6-iter early stop; 0.54 mid-run at 200) —
  single standoff pairs; `kick_after` frees them (0.335 frozen → 0.557) but a
  sep-aware stop or best-iterate return was *not* built; (b) the preserve
  "clean win" (CV ≤0.075, sep ≥0.70 at 300 fixed iters) is met by B
  (0.044 / 0.728).

**Code landed:** `ClippedSpacingForce` in `src/repel_forces.jl` (+ export);
default flip + `cv_target`/`stall_after` kwargs + `_dnn_cv` in `src/repel.jl`;
tests updated/added in `test/repel.jl` (force values, β-feeds-default now vs
clipped, stall/cv_target stopping); docs updated (`docs/src/repel.md` — was
stale, claimed InverseDistanceForce default — and `docs/src/api.md`).

**Verification (run 2026-06-11, after the session; gates 1 thread):**
1. **Seeded gate (`--placement=bridson --resample-boundary`): PASS, improved**
   — final (repel+kick+cull) CV **0.094 → 0.044**, sep/Δ **0.616 → 0.694**,
   0 singular, **cull removed 0** (still silent). The 300 repel iterations now
   add quality on top of the raw-PASS seed instead of eroding it.
2. **Legacy gate (default `:random` + face centers): MARGINAL as before, but
   mildly worse** — post-cull CV 0.151 → 0.184, culled 626 → 650 (sep 0.500,
   0 singular unchanged). Mechanism: the clipped force no longer pulls the
   uneven face-center boundary toward uniformity (no attraction), so the
   boundary CV floor rises. Accepted: this is the proven-inferior pipeline
   slated for removal (plan item 3); anyone needing the old behavior passes
   `force_model=SpacingEquilibriumForce(β)` explicitly. The verdict class
   (MARGINAL) and the gate-binding sep/singular criteria are unchanged.
3. **`Pkg.test()`: green** — 142,816 pass / 2 broken (pre-existing), 5m10s,
   including the new force-model and `cv_target`/`stall_after` testitems.
4. **`validate_repel.jl` (box.stl): separation 0.0 / mesh_ratio Inf — NOT a
   regression.** A/B on the identical cloud shows both forces end with
   coincident bnd–bnd pairs (old default **381**, new default **349**): it is
   the pre-existing deterministic-projection parking defect (open defect 1
   family) on a scenario with h = diag/18 ≈ 2.4 m vs ~0.22 m face-center
   spacing — the documented box.stl spacing-vs-tessellation trap — and the
   script uses no `cull_ratio`. The script's frozen-vs-live comparison is
   currently meaningless (both Inf); fixing it (resampled boundary or
   `cull_ratio`) added to housekeeping.

### Generality + benchmark harness

*(Ordering superseded by the planned session sequence above; the geometry
ladder folds naturally into plan item 2's bunny work, and the benchmark
harness into plan item 3's removal justification.)*

The Session-2 result reframes the program: the static-generation problem is
solved on the cavity (direct generation passes raw, cull silent). It remains
to establish that this *generalizes* and to quantify the SOTA claim:

- **Geometry ladder for the direct pipeline**
  (`sample_surface` + `:bridson`): `box.stl` (sharp edges/flat faces — watch
  the sampler near feature edges: continuous sampling has no special edge
  handling, and adjacent-face proximity blocking may under-sample creases) →
  `bifurcation.stl` (thin curved branches — watch the 3D-distance blocking
  caveat if branch thickness approaches `h`) → the
  `examples/octree_boundary_layer.jl` configuration (`bunny.stl` +
  `BoundaryLayerSpacing`), pushed to ≥10⁶ volume points — the
  production-scale rung: graded spacing exercises both the `min(rᵢ,rⱼ)`
  criterion at a real h-contrast and the background-grid memory behavior,
  and its wall-clock is the measurement that triggers — or retires — the
  gap-tracking sampler below. Success = raw gate-style PASS + silent cull on
  all rungs. Re-validate geometry sanity first (area vs reference, isinside
  probes) — lesson learned.
- **Benchmark harness**: direct pipeline (≈ PNP-equivalent) vs the old
  repel pipeline on identical geometries; all gate metrics + wall-clock,
  stated thread counts. The raw Bridson-direct numbers are now meaningful
  (0 singular on the cavity raw cloud with the resampled boundary).
- **Repel's remaining mandate** (feeds Session 4 decision): incremental
  re-relaxation under boundary change (shape-opt) and deposition. Before
  building on it, address open defect 2 (repel degrades constructed
  blue-noise — force equilibrium vs PDS statistics, early stopping).

### Conditional — octree NN search + momentum (repel speed)

*(Feeds plan item 1 if repel-quality work also needs speed; momentum and
force-model work belong to that session, the NN search stays conditional.)*

Trigger (updated after Session 1): the original premise — O(N) rebuild vs
O(N log N) — is void; the rebuild is a measured 3% of loop time. The only
remaining upside is query locality (kd traversal is ~50% of the loop), so
build it only if the shape-opt loop's measured budget still demands ≤25%
more speed after the quality work lands. Design summary (full plan retired
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

### Sampler scaling — octree gap tracking (recorded 2026-06-11, design only)

The direct pipeline is the new hot path, so its scaling matters at the
production target (millions of points). Measured profile of the current
dart thrower (cavity, 1 thread): **~3.3 µs/candidate, ~2.3% acceptance**
(~290k candidates for 6.8k accepted — each retired active point burns its
30 attempts mostly into covered or exterior space). Extrapolated serial cost
at 10M points: **~20–25 min** — fine at gate scale, unacceptable at
production scale. Two structural weaknesses appear at scale:

1. **Wasted darts** dominate (the per-candidate work — `find_leaf` + bucket
   scan — is already cheap; the count is the problem).
2. **The uniform background grid breaks under graded spacing**: cell count
   ∝ `(L/h_min)³` independent of N, so a boundary-layer `h_min` explodes
   memory; the current `1<<27`-cell guard then coarsens cells and widens
   scans.

The fix for both is the same structure we already have: **octree gap
tracking** (maximal Poisson-disk sampling, Ebeida/Gamito family), where the
node octree replaces both the candidate generator and, at scale, the
proximity buckets:

- Maintain a pool of **uncovered cells**, seeded from the node-octree
  interior/boundary leaves (containment becomes free by construction —
  `SpacingCriterion` already subdivides to `≈ alpha·h(x)`, exactly the right
  granularity for graded spacing, the awkward case in the literature).
- Draw darts volume-weighted from the pool only; on acceptance mark covered
  cells, subdivide partially-covered ones, keep uncovered children.
- **Pool empty = maximality proof** — `stall_limit` and the fuzzy truncation
  warning become exact statements ("saturated" vs "cap hit with gaps left").
- Expected ~5–20× fewer candidate evaluations; composes with **parallel
  phase groups** (leaves ≥ 2r apart are independent; 8-coloring on the
  2:1-balanced tree) for another ~×n_threads. Order of magnitude at 10M
  points, 8 threads: tens of seconds to ~2 min.
- Same idea applies to `sample_surface` with uncovered triangle *fragments*.

**Trigger to build it** (don't before): (a) the shape-opt loop chooses
re-seed-per-design-step over incremental repel — a faster sampler then
sidesteps open defect 2 entirely; or (b) the Session-3 benchmark at
production N (e.g. bunny at h giving ≥10⁶ points) shows sampler wall-clock
is material. Until then the simple dart thrower is correct and adequate at
gate scale.

### After the speed push — deposition (the novelty)

1. Cavity with decimated boundary (~h or sparser) + `deposit_ratio=0.5`;
   success = gate PASS **and cull silent**.
2. Boundary-free entry point: discretize from `TriangleOctree` + spacing
   alone, deposition grows the boundary from nothing. Needs
   empty-`PointBoundary` plumbing.

### Target end state

**Reached for static generation (2026-06-11), better than the original
target:** Poisson-disk boundary + graded Bridson volume → gate PASS **by
construction, zero repel iterations**, ~3 s single-threaded on the cavity
(the original target was ~30–50 iterations). CV 0.071 raw also beats the
< 0.10 stretch goal. Remaining end-state work is the *dynamic* case: fast
re-relaxation when the boundary changes inside the shape-opt loop —
re-seeding from scratch each design step (3 s) vs incremental repel
(0.47 s / 10 iters at 45.7k pts, 8 threads) is now a measurable trade-off,
and repel must first stop degrading seeded clouds (open defect 2).

## Validation assets

- `validate_cavity.jl` (repo root): annular cavity through
  Octree→repel→kick→cull. Geometry regenerated clean 2026-06-11 (4416
  facets, area-asserted at export and re-import; the corrupt original is
  parked untracked at `test/data/cavity_corrupt_backup.stl`). Flags:
  `--placement=random|jittered|lattice|bridson`, `--resample-boundary`,
  `--save-vtk`. Current verdicts (1 thread): default → MARGINAL (CV 0.151,
  626 culled); `--placement=bridson --resample-boundary` → **PASS**
  (sep/Δ=0.616, CV=0.094, 0 singular, 0 culled).
- `validate_repel.jl` (repo root): box.stl frozen-vs-live comparison.
- Full `Pkg.test()` green (142k+ assertions), including `:bridson` and
  `sample_surface` testitems.

## Housekeeping backlog

- Update `CLAUDE.md` repel docs (`α_min`, `kick_after`, `cull_ratio` warn,
  `deposit_ratio`, `trace`).
- Float32/Float64 type-mismatch fix in `octree.jl`.
- Bridson truncation warning fires when the cap sits exactly at the
  saturation count (the cavity gate case) — benign there (scattered
  single-dart holes, repel closes them), but consider only warning when the
  active front is still large relative to the generated count.
- `validate_repel.jl` is currently meaningless: both frozen and live runs end
  at separation 0 / mesh_ratio Inf (face-center boundary + projection parking
  on box edges at h ≫ tessellation; pre-existing, not Session-3). Rework it
  with a resampled boundary and/or `cull_ratio`, or retire it — the live-tree
  fix it demonstrates is long since default.
