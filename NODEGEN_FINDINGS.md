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

## ⚠ Geometry corruption notice #2 (discovered 2026-06-11, Session 4)

**The 2026-06-11 "clean geometry" was inside-out.** The corrected
`_make_sphere_mesh` wound its triangles **inward**, so the regenerated cavity
(and `test/data/cavity.stl`) had outer-shell normals pointing in and
inner-shell normals pointing out: the signed-distance classification put the
solid annulus OUTSIDE and the complement (inner hole + bbox-corner region)
INSIDE. Direct check on the Session-2/3 seeded configuration: **0 of 6842
volume points were in the annulus** (1297 in the hole, 5545 between the outer
sphere and the bbox). Every Session 2/3 cavity number — the raw-PASS headline,
the Session-3 force-model table, the production-config timings — was measured
on a blue-noise cloud filling the *complement* domain. The d_NN gate is
structurally blind to point *location*, so everything still PASSed; the
Bridson truncation warning that Session 3 rationalized as "benign mild
truncation" was a symptom (the complement is larger than the annulus). The
area assertion added after corruption #1 cannot see orientation.

A second, independent src defect was exposed while diagnosing this:
`_local_sign_vote` returned "ambiguous" whenever `find_leaf` on the nearest
surface point tie-broke into a leaf holding no triangles, and `isinside` maps
ambiguous (sd = 0) to EXTERIOR — **11.6% of true annulus-interior points
misclassified as outside** even after the winding fix (0% in the opposite
direction). Fixed in `src/octree/triangle_octree.jl`, then the whole vote was
replaced by the exact angle-weighted pseudonormal sign and an inside-out
construction guard was added (Session 4b below). Post-fix scan: 0 / 300k
misclassifications.

Fixes landed (Session 4): generator winding corrected in `validate_cavity.jl`
(and the experiment's copy), orientation guard added to the gate (mid-annulus
+ hole isinside probes that **error** on an inside-out mesh), sign-vote fix in
src. **`test/data/cavity.stl` on disk is still the inside-out artifact** — the
gate now refuses it with instructions to delete and regenerate (left to
Davide; no test in `test/` uses the file). Lesson, now twice paid: validate
the domain with isinside probes *and* check where generated points actually
land (radius histogram / outside-count / fill distance) — area and d_NN
metrics each have a blind spot that these close.

Dynamics-level conclusions from Sessions 2–3 (condensation instability,
clipped force preserving Poisson-disk statistics, cv_target/stall stopping,
kick mechanics) are force-model properties measured on a *legitimate (if
unintended) domain* and were re-confirmed in spirit by Session 4's true-annulus
runs; the specific quoted metrics are superseded by the Session-4 baseline
below.

## Current status (2026-06-11 clean-geometry gate, branch `shape_optimization_utils`)

*(Superseded 2026-06-11 Session 4: numbers below were measured on the
inside-out domain — see corruption notice #2. True-annulus baseline, 1 thread,
final code: direct pipeline **PASS raw, sep/Δ 0.750, CV 0.056, coord 12.1,
0 singular, fill max 1.18Δ, cull silent**, 9633 pts in ~5.4 s
boundary+volume + ~4 s TriangleOctree.)*

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

**Honest SOTA position (updated 2026-06-11, revised 2026-06-16):**
- *Quality*: the direct pipeline now meets the gate by construction
  (sep ≥ 0.75·r, CV 0.07, 0 singular raw) — same construction class as the
  reference direct generators (Medusa / Slak–Kosec PNP); a head-to-head on
  identical geometries is still unrun (Session 3).
- *Speed*: in the direct-generation class now — one O(N) advancing-front pass
  per stage (~3 s total on the cavity, 1 thread); the 300-iteration repel
  budget is gone for static geometry. **⚠ BUT: Bridson does not scale —
  bunny.stl (70k facets, 129k m³) timed out at every spacing finer than the
  coarse original parameters (Session 5). The "direct pipeline passes raw"
  claim holds only at cavity scale (~10k points). Production scale
  (10⁵–10⁷ points) requires the gap-tracking sampler.**
- *Differentiator*: incremental re-relaxation in the shape-opt loop +
  deposition (emergent boundary sampling; no published equivalent known) —
  both gated on fixing repel's blue-noise degradation (open defect 2).

## Architecture verdict (revised 2026-06-11, revised 2026-06-16)

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
  `:bridson` output passes the gate raw. **⚠ Scaling caveat (Session 5):
  `:bridson` does not scale to production geometry — bunny.stl timed out at
  every spacing finer than the coarse original parameters. The gap-tracking
  sampler (plan item 2) must replace the dart thrower before `:bridson` is
  usable at production N.**
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
6. **`_local_sign_vote` empty-vote-leaf misclassification** — **RESOLVED
   2026-06-11 Session 4** (see corruption notice #2): nearest surface point
   could land in a triangle-free leaf via `find_leaf` tie-break → ambiguous →
   EXTERIOR; 11.6% of true cavity-interior points misclassified. Initial fix
   (nearest triangle always votes) was then superseded the same day by
   removing the vote entirely: exact angle-weighted pseudonormal sign
   (Session 4b). Verified: 0/300k misclassified, `Pkg.test()` green.
7. **`test/data/cavity.stl` is the inside-out artifact** (corruption #2) —
   **RESOLVED 2026-06-16**: deleted the old artifact and regenerated with
   corrected winding. Verified: signed volume +3.478 (positive = outward),
   consistent normals, mid-annulus isinside = true, inner-hole isinside =
   false. ParaView inspection confirmed by Davide.
8. **TriangleOctree construction cost**: ~3.6–5 s on the 4416-facet cavity,
   ~12 s on 70k-facet bunny (1 thread, with `classify_leaves`; post-4b
   numbers — the vote-era cavity build was ~6 s), paid once per design step
   by *both* arms (A inside the `Octree` alg, B for repel's wall test). At
   production N this is a first-order in-loop cost — fold into plan item 2.
9. **Bridson dart thrower does not scale to production geometry** — **OPEN
   (Session 5, 2026-06-16)**. bunny.stl (70k facets, 129k m³): timed out at
   every spacing finer than `at_wall=0.8m, bulk=4.0m, layer_thickness=3.0m`
   (which itself produced only 14k volume points). Bottlenecks: `_bridson_h_min`
   O(leaves) scan + dart acceptance rate collapse at fine spacing. Fix: gap-
   tracking sampler (plan item 2, design already recorded). This is now the
   critical path — the direct pipeline cannot fill production geometries
   without it.

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
   **(Trigger fired in Session 4: re-seed-per-design-step won the shape-opt
   trade-off; include octree-build cost in scope — see Session 4 section.)**
   **⚠ CRITICAL PATH (Session 5, 2026-06-16): Bridson timed out on bunny.stl
   at every spacing finer than the coarse original parameters. The direct
   pipeline cannot fill production geometries without this. See Session 5.**
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
| Octree gap-tracking sampler | sampler wall-clock at production N (10⁶–10⁷ pts) | ~5–20× fewer candidates, + parallel phase groups; maximality proof (design recorded below) | **CRITICAL PATH — Bridson timed out on bunny.stl (Session 5)** |

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

### Session 4 — shape-opt re-relaxation trade-off (2026-06-11, pre-step for plan item 2) ✅

Question: per design step of the shape-opt loop, is it cheaper/better to
**re-seed from scratch (A)** or to **warm-start repel from the previous
cloud (B)**? Decides repel's in-loop role and is trigger (a) for the
gap-tracking sampler. Harness: `shape_opt_tradeoff.jl` (repo root), all runs
**1 thread**, Δ=0.08, deterministic seeds, both arms from the same base state
(true-annulus direct-pipeline cloud, 2791 bnd + 6842 vol). Every deformed
geometry validated before measuring (area vs analytic incl. Thomsen ellipsoid
area, 42 isinside probes) — which is how corruption #2 was caught (see notice).

Deformation ladder: r_inner 0.547 → {0.550, 0.560, 0.600} (wall moves
0.04/0.16/0.66·Δ; domain shrinks → old points strand outside) + 5% ellipsoidal
x-stretch of the outer shell (domain grows → fresh near-wall underdensity).
Arms: **A** = Poisson-disk boundary + `:bridson` volume from scratch;
**B** = same new boundary + old volume points, production repel
(`kick_after=10, cv_target=0.07, stall_after=50, cull_ratio=0.5`);
**B2** = B with stranded-outside old points dropped first (one isinside pass,
~0.07 s). Beyond the gate, each final cloud got an **outside-point count** and
a **fill-distance probe** (20k interior probes; d_NN metrics see neither).

Numbers below are from the final code state (after the Session-4b
pseudonormal/octree changes, which cut all walls ~35%); the pre-4b run
reached identical verdicts and conclusions on every rung.

| rung | arm | wall | iters | sep/Δ | CV | fill max/Δ | outside | culled | verdict |
|---|---|---|---|---|---|---|---|---|---|
| +0.5% | A | 6.2 s | — | 0.750 | 0.057 | 1.25 | 0 | 0 | PASS |
| +0.5% | B | 5.4 s | 13 | 0.511 | 0.069 | 1.18 | 0 | **1** | PASS |
| +2.4% | A | 5.1 s | — | 0.750 | 0.056 | 1.23 | 0 | 0 | PASS |
| +2.4% | B | 5.6 s | 19 | 0.530 | 0.069 | 1.18 | **1** | **1** | PASS |
| +2.4% | B2 | 5.6 s | 19 | 0.530 | 0.069 | 1.18 | 0 | **1** | PASS |
| +9.7% | A | 5.1 s | — | 0.750 | 0.057 | 1.08 | 0 | 0 | PASS |
| +9.7% | B | 14.5 s | 116 | 0.558 | 0.070 | 1.17 | **154** | 0 | PASS* |
| +9.7% | B2 | 7.8 s | 48 | 0.557 | 0.069 | 1.18 | 0 | **1** | PASS |
| stretch 1.05 | A | 5.4 s | — | 0.750 | 0.057 | 1.12 | 0 | 0 | PASS |
| stretch 1.05 | B | 5.7 s | 7 | 0.509 | 0.069 | 1.20 | 0 | **1** | PASS |

Correspondence (B's qualitative edge for warm-starting RBF-FD weights —
fraction of surviving volume points displaced < 0.25Δ): +0.5% **100%**
(median 0.022Δ), +2.4% **100%** (0.032Δ), +9.7% B2 **97%** (0.067Δ, 80%
< 0.1Δ); surviving fraction 1.000 / 1.000 / 0.977. A has zero correspondence
by construction.

**Decision: re-seed per design step (arm A) wins; repel-in-loop is retired as
the default.** The evidence:

1. **A is deformation-size-invariant and strictly higher quality everywhere**:
   sep/Δ 0.750 vs 0.51–0.56, CV 0.056 vs 0.069, cull always silent, zero
   stranded points, wall-clock flat at ~5–6 s regardless of step size.
2. **B-unfiltered is disqualified outright**: old points outside the deformed
   domain *cannot re-enter* — the octree wall rule reverts any outside
   proposal, and in-domain neighbors push escapees further out. At +9.7%,
   154 of 156 stranded points remained in the hole at stop while the d_NN
   gate still said PASS (PASS* above) — exactly the metric blind spot of
   corruption #1/#2 reproduced inside a healthy run. Any warm start MUST
   pre-filter with an isinside pass (B2; ~0.05 s — negligible).
3. **The cull defect signal fired in 5 of 6 B-arm runs** (1 point each): the
   fresh Poisson-disk boundary lands points within 0.5Δ of old volume points
   and the `cv_target` stop fires before repel separates the pair. This is
   the Session-3 caveat (sep is a min-statistic; no sep-aware stop was built)
   materializing — warm-start + early stop structurally risks near-duplicate
   wall pairs. A never culled.
4. **Wall-clock is octree-dominated, not sampler-dominated, at this scale**:
   of A's ~5–6 s, the `:bridson` fill itself is ~0.9 s and the boundary
   sample 0.06 s; ~3.5–4 s is `TriangleOctree` construction + ~1.2 s
   node-octree build/classify (B pays the same TriangleOctree for repel's
   wall test). The marginal cost of A over B2's repel only favors B at small
   steps (B2 repel ~1.4 s at ≤2.4% vs A's ~2.1 s octree+fill), and that
   advantage is bought with lower quality.
5. The ellipsoid (growth) rung is the one *positive* repel surprise:
   repulsion-only expansion did fill the new 0.6Δ-deep near-wall layer
   (fill max 1.20Δ, no void) in 7 iterations — warm-start is not
   coverage-broken under growth, just lower-quality (near-wall CV 0.072 vs
   0.055) and 6% under-dense globally.

**Consequences:**
- **Trigger (a) for the gap-tracking sampler FIRES**: re-seed-per-design-step
  is the chosen mode, so sampler + octree-build wall-clock at production N is
  the binding in-loop cost. Plan item 2 rises in priority, and its scope
  should include the **TriangleOctree/node-octree build cost** (~7.8 s of the
  ~8 s step at a mere 4.4k facets / 9.6k points; production is 70k facets /
  10⁶–10⁷ points) — at minimum incremental octree reuse under small boundary
  deformation, alongside the sampler acceleration.
- **Repel's remaining mandates**: deposition (unchanged), and *optional*
  correspondence-preserving refinement at small steps (≤2.4%), where B2
  keeps ~100% of points within 0.25Δ — valuable only if Macchiato's RBF-FD
  weight reuse is worth more than the sep 0.75→0.53 quality loss; that is a
  solver-side trade-off to be measured there, not assumed here. If it ever
  revives, repel needs the sep-aware stop (finding 3).
- The Session-3 production config (`cv_target=0.07`) behaves as designed on
  warm starts (stops at direct-pipeline CV), but `cv_target` alone is not a
  sufficient acceptance test: pair it with the cull warning (fired) and an
  outside-count (new) in any future in-loop use.

### Session 4b — isinside hardening: pseudonormal signed distance (2026-06-11)

Follow-up to the sign-vote defect (corruption notice #2): the distance-weighted
local sign vote was replaced wholesale by the **angle-weighted pseudonormal
test** (Bærentzen & Aanæs 2005) in `src/octree/triangle_octree.jl` +
`geometric_utils.jl`:

- `closest_point_on_triangle` now has a `_feature` variant returning which
  triangle feature (face / edge / vertex) the closest point lies on — a free
  by-product of the Ericson region tests, no tolerance involved.
- `MeshPseudonormals` (built once per `TriangleOctree`, O(n), coordinate-keyed
  so STL triangle soup needs no topology cleanup): face normals, per-edge
  normal sums, angle-weighted per-vertex normal sums.
- `_compute_signed_distance_octree` signs by
  `dot(p − cp, pseudonormal(closest feature))` — **provably exact for
  watertight, consistently outward-oriented meshes**; the vote's failure
  modes (ambiguous → EXTERIOR bias; mixed votes across thin sheets — the
  bifurcation caveat; the empty-vote-leaf bug) are eliminated structurally,
  not patched. The nearest-triangle traversal caches the winning feature, so
  the exact query does strictly less work than one vote iteration.
- New construction guard: **signed-volume check** (`Σ dot(v1, v2×v3)/6`,
  divergence theorem) under `verify_orientation && classify_leaves` — an O(n)
  exact witness that a *consistently wound but globally inside-out* closed
  mesh is rejected at construction with a clear error. This is the check that
  would have caught corruption #2 the moment the first `TriangleOctree` was
  built. (`has_consistent_normals` cannot see global inversion. The rejection
  threshold is scale-relative — clearly negative volume only — so open/flat
  surfaces with meaningless ≈0 signed volume still construct; they remain
  fine for distance-only use, and `classify_leaves=false` skips the check
  entirely. Verified: inverted cube vol −1 and the inside-out cavity vol
  −3.48 rejected; open square and all authored STLs accepted.)
- Testitems added (`test/octree_isinside.jl`): pseudonormal sign correctness
  for queries whose closest feature is a cube corner / edge / face (the cases
  a face-normal sign test gets wrong), and the signed-volume guard contract
  (inverted cube rejected, distance-only allowed, open square allowed).

Measured (1 thread, `bench_isinside.jl`, 200k queries/class, vote baseline
includes the empty-leaf patch):

| mesh | build | uniform-bbox query | near-surface query |
|---|---|---|---|
| cavity 4.4k facets (vote) | 5.4 s | 6.6 µs | 12.9 µs |
| cavity (pseudonormal) | **3.7 s** | **4.0 µs** | **6.9 µs** |
| bunny 70k facets (vote) | 13.1 s | 0.32 µs | 2.45 µs |
| bunny (pseudonormal) | 12.2 s | 0.33 µs | 2.62 µs |

Cavity exact path ~2× faster (coarse mesh → big vote leaves → the vote loop
was the cost); bunny within noise (the octree traversal dominates and is
untouched — that, not the sign, is the next efficiency lever if octree-build/
query cost matters at production scale, plan item 2). Correctness: 0/300k
misclassifications on the corrected cavity (interior and exterior), the
formerly-failing exact-tie probes (mid-annulus on the z-axis/diagonal) now
classify correctly, and the inside-out generator output is rejected at
construction.

**Verification (2026-06-12 AM, end of session): full `Pkg.test()` green —
189,673 pass / 2 broken (pre-existing) / 0 failed, 4m53s — including the new
pseudonormal-feature and signed-volume-guard testitems, after the guard
threshold was made scale-relative (the first run errored on the open
`simple_square_mesh` testitem, signed volume ≡ 0; clearly-negative-only fixed
it). The trade-off benchmark re-run on the final code reproduced every
Session-4 verdict with ~35% lower walls (table above).**

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

**Trigger (a) FIRED 2026-06-11 (Session 4)**: re-seed won the trade-off on
every deformation rung — build the sampler next, and include the
octree-construction cost in its scope (Session 4 finding 4: octree builds,
not the dart thrower, dominate the per-step wall at gate scale).

### After the speed push — deposition (the novelty)

1. Cavity with decimated boundary (~h or sparser) + `deposit_ratio=0.5`;
   success = gate PASS **and cull silent**.
2. Boundary-free entry point: discretize from `TriangleOctree` + spacing
   alone, deposition grows the boundary from nothing. Needs
   empty-`PointBoundary` plumbing.

### Target end state

**Reached for static generation (2026-06-11), better than the original
target:** Poisson-disk boundary + graded Bridson volume → gate PASS **by
construction, zero repel iterations** (true-annulus re-measurement, Session 4:
sep/Δ 0.750, CV 0.056, fill max 1.18Δ; ~5.4 s single-threaded
boundary+volume of which ~3.3 s is octree build/classify, +~4 s
TriangleOctree if wall queries are needed). CV raw beats the < 0.10 stretch
goal. The *dynamic* case is now **decided (Session 4)**: re-seed from scratch
each design step — deformation-size-invariant PASS at ~5–6 s — beats incremental
repel (lower quality everywhere, strands points under shrink without an
isinside pre-filter, trips the cull signal). Remaining end-state work is
making the re-seed step fast at production N: gap-tracking sampler + octree
build cost (plan item 2). **⚠ Session 5 (2026-06-16): confirmed this is the
critical path — Bridson timed out on bunny.stl at production-relevant
spacings. The "reached" claim above holds only at cavity scale (~10k
points); production geometry (10⁵–10⁷ points) requires the gap-tracking
sampler before the direct pipeline is usable.**

## Validation assets

- `validate_cavity.jl` (repo root): annular cavity through
  Octree→repel→kick→cull. Geometry regenerated clean 2026-06-11 (4416
  facets, area-asserted at export and re-import; the corrupt original is
  parked untracked at `test/data/cavity_corrupt_backup.stl`). **Session 4:
  generator winding fixed (was inside-out — corruption #2) and an
  orientation guard added (mid-annulus/hole isinside probes); the gate now
  errors on the stale on-disk `cavity.stl` until it is deleted and
  regenerated. Pre-Session-4 verdict lines below were measured on the
  inside-out domain.** Flags:
  `--placement=random|jittered|lattice|bridson`, `--resample-boundary`,
  `--save-vtk`. Verdicts (1 thread, inside-out domain): default → MARGINAL
  (CV 0.151, 626 culled); `--placement=bridson --resample-boundary` → PASS
  (sep/Δ=0.616, CV=0.094, 0 singular, 0 culled). True-annulus direct
  pipeline (Session 4 harness): **PASS raw — sep/Δ 0.750, CV 0.056,
  coord 12.1, fill max 1.18Δ, 0 singular, 0 culled.**
- `shape_opt_tradeoff.jl` (repo root, Session 4): deformation-ladder
  re-seed-vs-warm-start benchmark with programmatic (corrected) cavity
  generator, per-rung geometry validation, outside-count, fill-distance and
  displacement-correspondence metrics.
- `bench_isinside.jl` (repo root, Session 4b): TriangleOctree construction +
  isinside query throughput on the cavity and `bunny.stl` (uniform-bbox and
  near-surface query mixes).
- `validate_repel.jl` (repo root): box.stl geometry-ladder rung 1 — direct
  pipeline seeding + repel improve-or-preserve verdict (reworked 2026-06-11;
  PASS, see housekeeping notes).
- Full `Pkg.test()` green (142k+ assertions), including `:bridson` and
  `sample_surface` testitems.

## Housekeeping backlog

All four items closed in the 2026-06-11 housekeeping sweep (post-Session 3,
pre-shape-opt-trade-off-test); `Pkg.test()` green after the sweep (189,671
pass / 2 pre-existing broken — count grew vs 142k because several testitems
assert per generated point/pair and point counts shifted with RNG state):

- ✅ `CLAUDE.md` repel docs updated (default `ClippedSpacingForce`,
  `cv_target`/`stall_after`, `kick_after`, cull-warn principle,
  `deposit_ratio`; `repel_forces.jl` listed in components).
- ✅ Float32/Float64 fix: `discretize` now promotes Float32 (binary-STL)
  boundaries to Float64 itself when `alg isa Octree`
  (`_ensure_float64_boundary` in `octree.jl`, preserves surface
  names/normals/areas; testitem in `test/octree.jl`). Scripts no longer need
  the manual mesh-promotion workaround; `validate_repel.jl` already dropped
  it, `validate_cavity.jl` still carries it (harmless — remove on next gate
  touch).
- ✅ Bridson truncation warning is now evidence-based: on cap-hit with a
  non-empty front, a bounded probe (200 darts, nothing inserted) checks
  whether uncovered volume actually remains; warn only then. Outcome on the
  cavity seed: the warning **still fires and is correct** — the analytic-count
  cap (6842) sits slightly below true front saturation, a genuine mild
  truncation (benign at gate scale). The silenced case is cap == exact
  saturation.
- ✅ `validate_repel.jl` reworked into geometry-ladder rung 1 (box.stl, sharp
  edges): direct pipeline (Poisson-disk boundary 713 + `:bridson` volume
  1244) + repel improve-or-preserve verdict. Measured (1 thread): raw
  CV 0.048 / sep 0.750; **300 repel iters → CV 0.020, sep/Δ 0.870** (PASS —
  flat faces let repel approach near-crystal order); production stopping
  (`cv_target=0.07`) correctly recognizes the already-at-target cloud and
  returns it **unchanged** in 1 iteration (PASS). The old frozen-vs-live
  comparison is retired (live tree long since default; its face-center
  scenario ended in projection-parked coincident pairs, sep 0 both arms).
  Bonus rung-1 datum: the surface sampler handled box.stl's sharp edges
  without under-sampling artifacts visible in the metrics (boundary CV
  folded into full-cloud CV 0.048 raw).
- Also landed during the sweep: `cv_target` stop now reverts to the
  pre-sweep configuration it actually measured (`p .= p_old` before break) —
  an already-at-target cloud is returned bit-identical instead of
  one-sweep-moved (the sweep had nudged box CV 0.048 → 0.053 before the fix);
  testitem asserts identity.

### Session 5 — bunny.stl production-scale validation (2026-06-16) ✅ findings

Attempted to run the direct pipeline (`sample_surface` boundary + `:bridson`
volume) on `bunny.stl` (69,664 facets, ~129k m³ interior volume, ~86 m
across) — the production-scale geometry referenced in the roadmap.

**Mesh properties (verified):**
- Consistent normals: yes
- Signed volume: 128,807 m³ (solid closed surface, not a thin shell)
- Bbox: 86 × 67 × 86 m

**Run 1 — original parameters** (`at_wall=0.8m, bulk=4.0m,
layer_thickness=3.0m`): completed in ~3.5 min. Boundary: 19k Poisson-disk
points. Volume: **14k Bridson points**. Metrics: sep/Δ 0.753, CV 0.158,
coordination 14.2, cull silent. Plot inspected — no points flying outside,
correct graded density near walls.

**Problem identified:** 14k volume points is far below the expected count.
Root cause: `BoundaryLayerSpacing` with `bulk=4.0m` and `layer_thickness=3.0m`
puts the bunny's entire interior at 4.0m spacing. The boundary layer is a thin
shell (3 m deep), and the bunny's interior beyond that is nearly empty at 4 m
resolution. Expected volume points at 4 m: 128,807/64 ≈ 2,000. The 14k count
is mostly boundary-layer fill.

**Run 2 — tighter spacing** (`at_wall=0.4m, bulk=2.0m, layer_thickness=6.0m`):
boundary: 80k points (227 s). Node octree: 345k leaves (18 s). Bridson front
**timed out** after 20 min — `_bridson_h_min` iterating 345k leaves + dart
throwing acceptance rate collapsed at the finer spacing.

**Run 3 — very fine spacing** (`at_wall=0.15m, bulk=1.5m,
layer_thickness=8.0m`): node octree: 10.5M leaves (470 s). Bridson front
timed out during `_bridson_h_min` alone.

**Conclusion: the Bridson dart thrower does not scale to production geometry.**
The two bottlenecks are:

1. **`_bridson_h_min`**: iterates over *all* octree leaves to find the minimum
   spacing — O(leaves). At 345k–10.5M leaves this dominates wall-clock before
   the front even starts.
2. **Dart throwing acceptance rate**: at fine spacing the fraction of candidates
   that pass the separation + isinside tests drops below ~2%, so the front burns
   hundreds of millions of rejected darts. Each rejection is cheap (bucket scan
   + isinside), but the count is the problem.

The gap-tracking sampler (plan item 2, design recorded 2026-06-11) directly
addresses both: the octree pool replaces `_bridson_h_min` (spacing is known per
pool cell) and eliminates wasted darts (draw only from uncovered cells). This
is now **the critical path** for the node generation pipeline — correctness and
quality are solved, but the direct pipeline cannot fill production geometries
in reasonable time without it.

**Action:** implement the octree gap-tracking sampler. Scope includes the
TriangleOctree/node-octree build cost (measured ~7.8 s at gate scale, first-order
at production N).
