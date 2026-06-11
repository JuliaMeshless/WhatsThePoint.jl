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
2. **Repel degrades constructed blue-noise** (new, Session 2): from a raw-PASS
   cloud, 300 iterations drift CV 0.071 → 0.093 and sep 0.75 → 0.55, residual
   plateaus ~0.2. Harmless for the gate (stays PASS) but disqualifying for
   "repel as polisher." Needs a force model whose equilibrium matches PDS
   statistics, or an early-stop/convergence criterion, before repel-in-loop
   work (its actual remaining use case) is built.
3. **Coordination**: with the clean geometry and the resampled pipeline,
   coordination is 10.5 raw / 14.3 after repel — at or near the 12–14 ideal
   band; the historical 18.6 was a corrupt-domain artifact. Consider the
   defect closed pending re-measurement on other geometries.
4. **Float32 STL → Octree Float64 CRS mismatch** — workaround: promote mesh.
   Worth fixing in `octree.jl`.
5. **Deposition validated on one 834-pt box only** — curved geometry,
   variable h, boundary-free start untested.

## Roadmap (assessed 2026-06-11)

Framing: total cost = (iterations to converge) × (cost per iteration). The
factors multiply and have independent levers:

| Lever | Attacks | Expected gain | Effort to first result |
|---|---|---|---|
| `:jittered` validation | iteration count | **measured 2026-06-11: none** | done |
| Bridson graded Poisson-disk (volume) | iteration count | **measured 2026-06-11: vol CV 0.074 raw; gate stays boundary-bound with face centers** | done |
| Surface Poisson-disk (boundary) | iteration count | **measured 2026-06-11: with `:bridson` volume the gate passes raw — iterations-to-PASS = 0** | done |
| k-NN buffer reuse (`knn!`) | cost/iter | **measured: −87% alloc, GC 8.8→0.7%, −11% threaded wall** | done |
| `rebuild_every` > 1 | cost/iter | **measured: ~5% — not a lever** (rebuild is 3% of loop) | done |
| Octree NN search | cost/iter | demoted again: static generation no longer iterates; relevant only if repel-in-loop survives | only on measured need |
| Momentum | iteration count | demoted: static pipeline needs 0 iterations; reconsider for shape-opt re-relaxation | only on measured need |

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

### Session 3 — generality + benchmark harness

The Session-2 result reframes the program: the static-generation problem is
solved on the cavity (direct generation passes raw, cull silent). Session 3
should establish that this *generalizes* and quantify the SOTA claim:

- **Geometry ladder for the direct pipeline**
  (`sample_surface` + `:bridson`): `box.stl` (sharp edges/flat faces — watch
  the sampler near feature edges: continuous sampling has no special edge
  handling, and adjacent-face proximity blocking may under-sample creases) →
  `bifurcation.stl` (thin curved branches — watch the 3D-distance blocking
  caveat if branch thickness approaches `h`). Success = raw gate-style PASS +
  silent cull on all three. Also re-validate geometry sanity first (area vs
  reference, isinside probes) — lesson learned.
- **Benchmark harness**: direct pipeline (≈ PNP-equivalent) vs the old
  repel pipeline on identical geometries; all gate metrics + wall-clock,
  stated thread counts. The raw Bridson-direct numbers are now meaningful
  (0 singular on the cavity raw cloud with the resampled boundary).
- **Repel's remaining mandate** (feeds Session 4 decision): incremental
  re-relaxation under boundary change (shape-opt) and deposition. Before
  building on it, address open defect 2 (repel degrades constructed
  blue-noise — force equilibrium vs PDS statistics, early stopping).

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
