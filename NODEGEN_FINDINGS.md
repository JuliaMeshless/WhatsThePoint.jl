# Node generation — state of development & roadmap

**Rewritten 2026-06-20** to reflect the *actual committed code* at HEAD
(`08dd8eb`, branch `shape_optimization_utils`). The previous revision had
drifted out of sync with git: its closing sessions (8 and 9) narrated
working-tree experiments that were never committed and contradicted the
landed code (see "Doc-vs-code reconciliation" below). This version is the
ground truth for opening a pull request. The full blow-by-blow session log
of how we got here lives in git history (`NODEGEN_FINDINGS.md` at the
listed commits, plus the retired scratch markdowns).

---

## 1. Goal & ecosystem context

Bring WhatsThePoint node generation to a state of practical SOTA for
downstream RBF-FD meshless solving. WhatsThePoint is the **geometry stage**
of the JuliaMeshless ecosystem:

| Package | Role | Consumes |
|---|---|---|
| **WhatsThePoint.jl** | point-cloud generation (this repo) | STL / `Mesh` |
| **RadialBasisFunctions.jl** | RBF interpolation + differential operators | point cloud + topology |
| **Macchiato.jl** | physics + shape-optimization framework | the above two |

Macchiato pins `WhatsThePoint = {rev = "main"}`, compat `"0.2"`. **All of
this branch's work reaches the solver only once merged to `main`** — that is
the immediate motivation for landing a clean PR now.

Strategic target: node generation that is good enough *and* fast enough to
sit **inside the Macchiato shape-optimization loop**. Boundary point count
and connectivity change every design step, so the cloud must be regenerated
fast, deterministically, and at production scale. The gating requirement is
**generation speed + robustness at 10⁶–10⁷ points with graded spacing**, not
just final quality on a small gate.

Two validation rungs are used throughout:
- **Cavity gate** (small): annular cavity (R=1, r=0.547), `validate_cavity.jl`
  — separation, spacing CV, coordination, per-stencil degree-3 Vandermonde
  σ_min/σ_max. Cheap, sharp on local quality.
- **Production rung** (large): `examples/octree_boundary_layer.jl` —
  `bunny.stl` (69,664 facets, ~129k m³, ~86 m across) with
  `BoundaryLayerSpacing`, pushed to ≥10⁶ volume points. This is where
  algorithmic choices are actually judged.

---

## ✅ RESOLVED (2026-06-21) — coarse-spacing usability: `suggest_spacing` + bridson clamp-and-warn

`:bridson` is the default placement (correct — it must be). Making it the
default exposed a **step-0 usability failure**: on a **1×1×1 unit cube with
`ConstantSpacing(1m)`**, `:bridson` returned **0 volume points, silently** — the
entire interior lies within one disk radius (`bridson_factor·h = 0.75·1 =
0.75 m`) of the boundary seeds, so every candidate is rejected. Unit-/small-
domain geometries are extremely common, so a default sampler that yields nothing
on "h ≈ domain size" was unusable. This is now fixed (landed on
`shape_optimization_utils`); the previously-RED coarse-spacing tests are green
again *without* pinning to `:random`.

**What landed** (`src/discretization/spacing_guidance.jl`,
wired into `src/discretization/algorithms/octree.jl`):

1. **`suggest_spacing(mesh | boundary | "model.stl")`** — the "step 0" probe,
   exported. A quick `du -sh`-style report of domain extent, enclosed volume,
   facet count, and three spacing landmarks:
   - `h_ceiling = L_min/(2·bridson_factor)` — at/above this the Poisson-disk
     interior is empty (the hard limit);
   - `h_baseline` — recommended start: ≈10 points across the shortest axis, or
     `cbrt(volume/n_points)` capped below the ceiling when a budget is given;
   - `h_fine = h_baseline/2`.
   Returns a `NamedTuple` so it composes directly into `discretize`.
2. **Bridson never fails silently** — before the node octree is built (its
   resolution is spacing-driven, so the check must precede subdivision),
   `_guard_coarse_spacing` probes the *interior* min spacing on a grid. When it
   is at/above `h_ceiling`, it emits a loud `@warn` (reporting the ceiling and
   the clamp target) and wraps the spacing in `_ClampedSpacing` capped at
   `L_min/10`, so generation still yields a usable cloud. Spacings that are
   already fillable are passed through untouched. A post-bridson
   `isempty → throw ArgumentError` is the belt-and-suspenders for degenerate /
   non-watertight domains the interior probe can't catch.
3. **Contract chosen: clamp + loud warn** (over warn+refuse). Rationale:
   "bridson must work out-of-the-box on everything" + the "ok-ish first
   estimate, then refine" workflow both favor producing *something* usable with
   a loud diagnostic over throwing and forcing a retry. The interior-grid probe
   (not the field's boundary minimum `at_wall`) is deliberate: a boundary-layer
   field bottoms out only in an infinitesimal wall shell that can't host a
   packing, so the interior sample is the honest "is this fillable?" test.

The coarse-spacing tests in `test/octree.jl` ("Octree with different spacing
types", "points are inside", "#76 high aspect ratio", "promotes Float32 STL
boundaries") now pass via the clamp; `test/spacing_guidance.jl` covers the new
helper and the warn path. Everything else in this document (direct pipeline,
gradient limiter, smart cap, etc.) stands unchanged.

---

## 2. Current state — what is committed and works (HEAD `08dd8eb`)

The branch is **+3918 / −318 lines over `main`** across 29 files. Generation
is now a **two-stage direct pipeline**; repel is a demoted, optional
relaxation tool. The following are landed in git and exercised by tests:

| Capability | Where | Status |
|---|---|---|
| **Poisson-disk boundary sampling** (`sample_surface`, `PointBoundary(mesh, spacing)`) — graded surface dart-throwing, replaces face-center import | `src/surface_sampling.jl` | ✅ landed, tested |
| **Bridson global Poisson-disk volume** (`placement = :bridson`) — single advancing front, cross-leaf separation guarantee | `src/discretization/algorithms/octree.jl` | ✅ landed, tested |
| **KDTree-accelerated spacing** — `BoundaryLayerSpacing` / `LogLike` build a KDTree in the constructor; `_min_distance` is O(log n) not O(n) | `src/discretization/spacings.jl` | ✅ landed (the key production-scale fix) |
| **Parallel leaf classification** (`tmap`) | `src/octree/spatial_octree.jl` | ✅ landed `08dd8eb` (12.5× on classification) |
| **Parallel `_bridson_h_min`** (`tmapreduce`) | `octree.jl` | ✅ landed 2026-06-20 (was still sequential; doc had over-claimed) |
| **Smart cap** — `max_points = nothing` default; Octree auto-estimates `⌈1.1·∑ box_vol/h³⌉` (`_estimate_volume_points`) | `octree.jl`, `discretization.jl` | ✅ landed 2026-06-20 (see §3 — `08dd8eb` shipped only the `nothing` default, which crashed) |
| **Gradient-limited spacing** (`max_growth`) — Lipschitz envelope of `h(x)` on octree leaves + node refinement: steep-but-smooth grading for CFD/RBF-FD | `octree.jl` | ✅ landed 2026-06-21 (see §4a) |
| **Exact pseudonormal isinside** (Bærentzen & Aanæs) + signed-volume inside-out guard | `src/octree/triangle_octree.jl`, `geometric_utils.jl` | ✅ landed, tested |
| **`ClippedSpacingForce` repel default** (repulsion-only) + `cv_target`/`stall_after` stopping | `src/repel.jl`, `repel_forces.jl` | ✅ landed, tested |
| **Spacing-fidelity metrics** (`spacing_metrics`, `spacing_fidelity_metrics`) | `src/metrics.jl` | ✅ landed |
| Clean regenerated `cavity.stl` (positive signed volume, consistent normals) | `test/data/cavity.stl` | ✅ landed |
| Production example + legacy variant | `examples/octree_boundary_layer*.jl` | ✅ landed |

### Production-scale result (the headline)

`bunny.stl`, `BoundaryLayerSpacing(at_wall=0.25m, bulk=1.0m,
layer_thickness=10.0m)`, smart cap, **1 thread**:

| Phase | Wall-clock | Share |
|---|---|---|
| Mesh load | 2.1 s | — |
| Boundary Poisson-disk sample | 10 s | 6% |
| TriangleOctree build | 7.5 s | 4% |
| Node octree build (2.79M leaves) | 23.6 s | 14% |
| Classification | 11.7 s | 7% |
| **Bridson fill (1.92M volume pts)** | **131.6 s** | **77%** |
| **Total** | **~172 s** | |

Quality at 1.92M volume points: **sep/h_min 0.751, spacing CV 0.051,
coordination 14.3, 0 singular stencils**. Smart cap auto-estimated 1.92M
points from the spacing integral. This is excellent blue noise at production
scale — the static-generation problem is solved for the bunny.

### Cavity gate (small rung, clean true-annulus geometry)

Direct pipeline (`sample_surface` boundary + `:bridson` volume) passes the
full gate **by construction, zero repel iterations**: sep/Δ 0.750, CV 0.056,
coordination 12.1, fill max 1.18Δ, 0 singular, cull silent, ~5–6 s
single-threaded (≈half of which is TriangleOctree construction).

---

## 3. Doc-vs-code reconciliation & the gap_tracked removal (2026-06-20)

The previous doc's tail claimed Session 7–8 changes were "not committed to
git" and that the gap-tracked sampler had been removed. **Both claims were
false against HEAD.** Git reality:

- `c70b824` *"implement gap tracking disk sampling"* — committed the
  `:gap_tracked` placement **and** the KDTree spacing acceleration.
- `08dd8eb` *"parallel classify_leaves & smart cap for n of nodes"* —
  committed parallel classification and the `max_points=nothing` default, but
  **not** the `_estimate_volume_points` estimator the default depends on, and
  **not** the `_bridson_h_min` parallelization the message implied; deleted the
  scratch scripts.

So parallel classification was in `main`-bound code, **the `:gap_tracked`
sampler was still present** (the "Session 8 removal" never landed), and **the
"smart cap" was half-wired** — see the next subsection.

**Resolved 2026-06-20: `:gap_tracked` removed.** Every benchmark said it was
*slower* than `:bridson`, not faster (cavity: comparable; bunny 1M config:
`:gap_tracked` timed out at 600s vs `:bridson`'s 132s — pool-management
overhead, `_draw_leaf`/`_refresh_pool!` Dict ops, and the per-candidate grid
check outweigh targeted dart throwing). The original "5–20× fewer candidates"
prediction did not hold, so the measured-inferior path was deleted:
`_GapTracker`, `_generate_gap_tracked`, `_draw_leaf`, `_refresh_pool!`, the
`:gap_tracked` validation branches in both `Octree` constructors, the
docstring section, and the `_discretize_volume` dispatch (≈225 lines).
**`:bridson` is now the sole global Poisson-disk sampler.** Verified: no
`gap`/`GapTracker` symbols remain in `src/`, package loads, full `Pkg.test()`
green (see §10).

**Resolved 2026-06-20: the broken "smart cap" default.** `08dd8eb` shipped
`max_points = nothing` as the `discretize` default but never committed the
`_estimate_volume_points` estimator that gives `nothing` meaning (it did not
exist anywhere in `src/`). The result was a latent crash: the recommended
default invocation `discretize(boundary, spacing; alg=Octree(...))` reached
`_generate_bridson(..., max_points=nothing)` and raised a `MethodError`. The
full suite stayed green only because **every test and the example passed an
explicit `max_points`**, so the default path was never exercised — the same
metric-blind-spot lesson (§5) in CI form. Fixes landed:
- `_estimate_volume_points(node_tree, classification, spacing)` =
  `⌈1.1·∑ box_volume/h(x)³⌉` over non-exterior leaves (parallel `tmapreduce`),
  resolved into `max_points` in the Octree `_discretize_volume` when `nothing`.
- `_bridson_h_min` parallelized with `tmapreduce` (was still the sequential
  scan the doc claimed had been parallelized).
- Extracted a shared `_non_exterior_leaves` helper (both integral helpers
  iterate the same leaf set) — a small de-duplication.
- New regression testitem (`test/octree.jl`, "Octree auto-estimates max_points
  when unset") **exercises the default path** so this cannot regress silently.
- Internal `_discretize_volume(::Octree)` default aligned to `nothing`.
Verified on the cavity mesh: default path estimates 2162, saturated front
fills 1068 (≈0.39× density, as documented); full `Pkg.test()` green (§10).

The `speedup_octree_nodegen.md` scratch note at repo root is superseded by the
KDTree fix it proposed (now landed) and the removals/fixes above — delete it
before the PR.

---

## 4. Architecture verdict

Two-stage **direct** pipeline; repel is optional relaxation, not the quality
workhorse:

- **Boundary**: `sample_surface(mesh, spacing)` / `PointBoundary(mesh, spacing)`
  — graded surface Poisson-disk. Supersedes face-center import for
  quality-sensitive use (face-center boundaries are themselves a defect, see
  §6). Caveat: 3D-distance blocking means sub-`h` thin sheets can under-sample
  one side.
- **Volume — `Octree` algorithm** (`placement = :bridson`): graded *density*
  (node count follows `h(x)`), arbitrary STL, single global advancing front
  with a cross-leaf separation guarantee `‖xᵢ−xⱼ‖ ≥ min(rᵢ,rⱼ)` enforced by a
  background bucket grid against *all* points (boundary included).
  `bridson_factor` (default 0.75 ≈ η^(1/3)) matches the saturated front to the
  prescribed density. `max_points` is a **cap, not a target**.
  - Note: the `Octree` struct's *default* placement is still `:random`;
    `:bridson` must be requested. Consider making `:bridson` the default as
    part of the PR (or at least in the examples/docs).
- **Repel** (`src/repel.jl`): retained for deposition and *optional*
  correspondence-preserving refinement at small deformation steps. On static
  geometry it should not follow a good direct seed — its force equilibrium is
  a worse configuration than a saturated Poisson-disk (see §6 defect 2, now
  mitigated by the clipped force).

**Legacy 3D algorithms** (`SlakKosec`, `VanDerSandeFornberg`) and
**`FornbergFlyer`** (2D-only) remain. They are "totally incapable of
discretizing bunny.stl" (Davide) and are slated for eventual removal in favour
of the direct pipeline — but removal touches the 2D story (`FornbergFlyer` is
the *only* 2D path), CLAUDE.md, docs, examples, and tests. **Out of scope for
this PR**; do it as a focused follow-up.

---

## 4a. Steep spacing gradients & the gradient limiter (`max_growth`, 2026-06-21)

Steep spacing variation (boundary layers, wakes, shocks) is the crucial CFD
case. How the pipeline handles it, across the three independent layers:

1. **Separation correctness — exact at any steepness.** The Bridson grid scans
   radius `r_c = factor·h(c)` around each candidate and rejects within
   `min(r_c, r_j)`; whichever radius is smaller, every true violator lies within
   `r_c`, so the scan is sufficient regardless of gradient. (The old §"cross-leaf
   separation at gradients" worry is moot — the global grid is exact.)
2. **Octree subdivision — center-only criterion + 2:1 balance.** Adjacent leaf
   *sizes* grade ≤2×, but this bounds the leaf structure, **not** the point
   density.
3. **Sampler density — followed exact `h(c)` with no grading limit (the gap).**
   A steep prescribed `h` produced an abrupt density jump: points pack at the
   fine spacing right up to the transition, then switch — a non-smooth `d_NN`
   field that distorts RBF-FD stencils straddling it. Measured on the cavity
   with a steep BL: worst adjacent-point `d_NN` ratio **4.25×**.

Practical ceilings (unchanged): proximity-grid memory `(L/h_min)³` (coarsens
past 134M cells); `BoundaryLayerSpacing` is a fixed-shape sigmoid (smooth,
bounded, but *not* geometric BL growth — can't express "wall 1e-5, grow 1.1×/
layer"); `LogLike` →0 at the wall (needs a floor).

**Landed fix — gradient-limited spacing (`Octree(...; max_growth=g)`).** Replaces
the prescribed `h₀(x)` with its `g`-Lipschitz envelope
`h(x) = minᵧ(h₀(y) + g·‖x−y‖)` — the steepest field ≤ `h₀` that grows no faster
than `g` (CFD growth ratios 1.1–1.2 ⇒ `g ≈ 0.1–0.2`). Implementation
(`src/discretization/algorithms/octree.jl`):
- `_gradient_limit_field` — multi-source min-plus (Bellman/Jacobi) relaxation
  over a k-NN graph of leaf centres, swept to a fixpoint, parallel `tmap`.
- `_apply_gradient_limit` — because limiting can make the field *finer* than
  `h₀` in a transition band, it **refines** any leaf the new field out-resolves
  (`box_size > alpha·h`) and re-limits, to a fixpoint. This is the
  "node-refinement-for-steep-but-smooth" mechanism.
- `_LeafSpacing` — serves the limited field to the sampler, grid sizing, and
  point-count estimate via `find_leaf` lookup, so the whole pipeline is
  consistent.
- No-op (no refinement, no cost) when `h₀` is already `g`-smooth — which the
  current gentle bunny BL configs (`g≈0.1`) already are.

**Measured (cavity, steep BL at_wall 0.03 / bulk 0.18 / layer 0.12, `g=0.15`):**

| metric | `max_growth=0` | `max_growth=0.15` |
|---|---|---|
| worst adjacent `d_NN` ratio | **4.25×** | **1.67×** |
| mean adjacent `d_NN` ratio | 1.16 | 1.055 |
| node-octree leaves | 36,989 | 100,878 (auto-refined) |
| volume points | 3,961 | 21,591 (finer band) |

Tested (`test/octree.jl`: "max_growth smooths steep spacing gradients",
"invalid max_growth throws").

**Known limitations / next levers (honest):**
- The limited field is **piecewise-constant per leaf** (a `find_leaf` lookup),
  so a point near a leaf face sees a stepped `h` — the residual source of the
  1.67× worst-case vs the ideal ~1.15×. Sub-leaf interpolation (store the
  source gradient, evaluate `h_leaf + g·dist`) would tighten it; not yet built.
- **Cost**: the refinement loop re-runs the *full* `classify_node_octree` each
  round (~10.7s on the cavity at 100k leaves). Incremental classification of
  only the new leaves is the obvious optimization before production-scale steep
  runs. Each Jacobi sweep is O(N·k) parallel; #sweeps ≈ band width in leaves.
- k-NN propagation crosses only the leaf graph; a thin exterior wall between two
  interior regions is bridged if centres are k-NN close (a geodesic limiter
  would be exact — not needed yet).
- Still missing the *modeling* side: a **geometric `BoundaryLayerSpacing`** mode
  (`first_height` + `growth_ratio` + `n_layers`) to *prescribe* true CFD layers,
  which the limiter then guarantees smooth into the bulk. Natural follow-up.

---

## 5. Quality gate & metrics

`mesh_ratio` (fill/separation) is *not* a primary metric — a min/max ratio
dominated by outliers (it reads as "elevated" under benign truncation). The
primary metrics (`spacing_fidelity_metrics`, `src/metrics.jl`):

| Indicator | Definition | Target |
|---|---|---|
| separation/Δ | min(d_NN) / Δ | > 0.1 (good ≈ 0.75) |
| spacing CV | std(d_NN/h) / mean(d_NN/h) | < 0.15 (good ≈ 0.05) |
| p05 / p95 (d_NN/h) | spacing-fidelity percentiles | tight around ~0.78 |
| coordination | mean neighbors within 1.4h | 12–14 |
| singular stencils | σ_min/σ_max < 1e-8 count | 0 |

**Hard-won lesson — every gate metric above is d_NN-based and therefore
structurally blind to *coverage voids*, *wrong domains*, and *point
location*.** Two separate geometry corruptions (a scrambled sphere generator,
then an inside-out winding) passed the full gate while points filled the
*complement* domain. Always validate the geometry itself first — mesh area
vs. analytic, `isinside` probes at known interior/exterior points, and a
radius/fill histogram of where points actually land — *before* trusting
distribution metrics. The cavity gate now asserts mesh area at export and
re-import and errors on an inside-out mesh; the `TriangleOctree` rejects a
globally inverted closed mesh at construction (signed-volume guard).

---

## 6. Established findings & lessons (don't re-derive)

**Repel dynamics** (all in `src/repel.jl` / `repel_forces.jl`):
- Defects found and fixed: frozen kd-tree (→ `rebuild_every=1`), rootless
  `InverseDistanceForce` default (→ equilibrium force), coincident-point NaN
  direction (→ `_safe_direction` random unit vector), and the key **balanced
  standoff** (closest pair freezes; only a stochastic `kick_after` breaks it).
- **Condensation instability** was open defect 2: the *attractive* branch of
  `SpacingEquilibriumForce` slowly condenses a good cloud into clusters +
  voids (coordination ↑, CV ↑, sep ↓). **Fixed** by making
  `ClippedSpacingForce` (repulsion-only, compact support, zero for `u ≥ u0`)
  the default — its equilibrium set contains every Poisson-disk configuration,
  so "improve or preserve" is structural. `SpacingEquilibriumForce` remains
  selectable via `force_model=`.
- **Stopping**: force-norm `tol` is the wrong shape (a saturated
  repulsion-only packing is a frustrated glass; residual plateaus ~0.05–0.07).
  Use `cv_target ≈ 0.07` (stop at direct-pipeline quality) as primary,
  `stall_after` as backstop. Both default to off (0).
- **More iterations don't help** an equilibrium defect; the levers are
  dynamics (step/kick) and seeding, not budget. `StrongSpacingForce` makes
  standoffs *worse*.

**Generation principles:**
- **Cull = defect signal** (Davide): `cull_ratio` must NEVER fire in healthy
  generation; both `repel` methods `@warn` when it removes anything.
  Activation ⇒ upstream generation bug.
- **Check spacing vs geometry resolution before trusting a failing test** —
  `box.stl` is mm-authored read as meters; a prescribed `h` 24× the boundary
  tessellation once sent a session chasing the wrong bug.
- **Face-center imported boundaries are a defect** for quality: tessellation
  pole rings and uneven face spacing floor the boundary CV (~0.24 after 300
  repel iters) and leave hundreds of near-duplicates. The Poisson-disk
  boundary measures CV ~0.06 raw and culls zero. Resolved for generation by
  `sample_surface`; still a caveat for face-center users.

**Shape-opt re-relaxation (Session 4 decision — drives Macchiato integration):**
- Per design step, **re-seed from scratch beats warm-start repel.** Re-seed is
  deformation-size-invariant and strictly higher quality (sep/Δ 0.75 vs
  0.51–0.56, CV 0.056 vs 0.069, cull silent, zero stranded points). Warm-start
  *must* pre-filter stranded points with an `isinside` pass or they sit
  permanently outside the deformed domain while the d_NN gate still says PASS.
- Repel's surviving mandate: **deposition** (escaped volume points project to
  the nearest triangle and become boundary points — emergent surface sampling,
  no known published equivalent; `deposit_ratio`, self-limiting) and *optional*
  correspondence-preserving refinement at small steps (≤2.4%, ~100% of points
  stay < 0.25Δ — valuable only if Macchiato's RBF-FD weight reuse is worth the
  sep 0.75→0.53 quality loss; a solver-side measurement, not assumed here).

---

## 7. Open items & known limitations

1. ~~`:gap_tracked` decision~~ — **resolved 2026-06-20 (removed, §3).**
2. **Bridson fill is now the dominant cost** (77% of the bunny wall, 132s for
   1.92M pts). The acceptance-rate collapse at fine graded spacing (~2% at
   fine `h`) is the real remaining scaling lever — *not* the proximity
   structure (the octree-native proximity map was tried and **failed**:
   >600s vs the grid's 22.6s, catastrophic per-candidate traversal cost; the
   `_BridsonGrid` is retained and adequate at current parameters,
   162 MB at h_min=0.25m).
3. **Background-grid memory under very fine graded spacing**: cell count ∝
   `(L/h_min)³`. Fine at h_min=0.25m; would explode below ~0.1m. A hash-grid
   (Dict-based, O(N) memory, O(1) lookup) is the right fix *if* that regime is
   ever needed — octree traversal was the wrong fix.
4. **TriangleOctree / node-octree build cost** (~7.5 s + ~24 s on bunny; ~3.7 s
   on cavity) is paid once per design step in the shape-opt loop. At small
   deformations, incremental octree reuse is the obvious win but is **large
   effort**; defer until the Macchiato loop's measured budget demands it.
5. **Default placement is `:random`** in the `Octree` struct — surprising
   given `:bridson` is the recommended production mode. Consider flipping.
6. ~~Stale `discretize` docstring (`max_points=10_000_000`)~~ — **fixed
   2026-06-20**: both `discretize` docstrings now document
   `max_points=nothing` (auto-estimate for Octree, 10M fallback otherwise).
7. **Cross-leaf separation at steep spacing gradients** is enforced by the
   global grid, not by coverage tracking (a code comment claiming neighbor
   coverage propagation is aspirational). Correctness holds via the grid; a
   nicety, not a quality issue at `alpha=1.0`.

---

## 8. PR scope — getting to "ready to open"

Recommended contents of the PR off `shape_optimization_utils` → `main`:

**Must do:**
1. ✅ Resolve `:gap_tracked` — **done 2026-06-20 (removed, §3)** so the
   committed code tells one coherent story.
2. ✅ Fix the stale `discretize` docstring (`max_points` default) — **done
   2026-06-20**.
3. Delete/retire the repo-root scratch artifacts that shouldn't ship:
   `speedup_octree_nodegen.md` (superseded), and decide whether
   `validate_cavity.jl` belongs in the repo or under `examples/` / `test/`.
4. ✅ Trim this document of the duplicated history and confirm it matches the
   final committed code (this rewrite is that trim).
5. Confirm `Pkg.test()` green after the gap_tracked removal (last full run
   before removal: 189,673 pass / 2 pre-existing broken / 0 failed).

**Nice to have in the same PR:**
- Flip the `Octree` default placement to `:bridson` (or document loudly).
- A short `docs/` page describing the direct pipeline as the recommended path.

**Explicitly out of scope** (follow-up PRs): removing the legacy
SlakKosec/VanDerSandeFornberg/FornbergFlyer algorithms (2D-support decision),
incremental octree reuse, and the hash-grid for sub-0.1m spacing.

---

## 9. Where we stand vs SOTA, and toward Macchiato / RBF

**Honest SOTA position:**
- *Quality* — the direct pipeline meets the gate by construction (sep ≈ 0.75,
  CV ≈ 0.05, 0 singular raw): the same construction class as the reference
  direct generators (Medusa / Slak–Kosec PNP). A documented head-to-head on
  identical geometries against Medusa is **still unrun** — that is the
  measurement that would let us *state* parity rather than claim it.
- *Speed* — production-scale generation works: 1.92M points on bunny.stl in
  172 s single-threaded, with classification no longer the bottleneck. Bridson
  fill (77%) is the next lever (acceptance-rate collapse at fine spacing), and
  the whole pipeline is largely single-threaded (boundary sampling and the
  fill front are the remaining serial phases).
- *Differentiators* — (a) **re-seed-per-design-step** integration with a
  shape-opt loop, deformation-size-invariant; (b) **deposition** (emergent
  boundary sampling) with no known published equivalent. Both are unique to
  this pipeline.

**What Macchiato needs from us next** (so the PR unblocks the other fronts):
1. **A merged `main`** that Macchiato can pin — it already tracks `rev=main`,
   compat `0.2`. This PR is that unblock.
2. **Deterministic, fast re-seeding** at the production N of the target
   shape-opt problem (e.g. the plate-with-hole / Helmholtz cases in
   Macchiato's plans). The re-seed decision (§6) is made; the remaining gate is
   the Bridson-fill wall-clock at the loop's actual N and thread count —
   measure it on the real Macchiato geometry, not just the bunny.
3. **AD-compatibility awareness**: Macchiato's shape-opt is going through AD
   (manual adjoint + AD pipeline). Node *positions* from a dart-thrower are
   non-differentiable by construction — the shape-opt adjoint must treat the
   cloud as re-generated (discrete) per step, with sensitivities flowing
   through the *solver* on a fixed cloud, not through the sampler. Confirm this
   boundary with the Macchiato side before assuming repel's
   correspondence-preserving mode is needed.

**For RadialBasisFunctions.jl**: the relevant lever is stencil quality
(coordination 12–14, 0 singular Vandermonde) and topology. Those are already
in good shape from the direct pipeline; the natural next collaboration is a
WTP→RBF benchmark that scores generated clouds by *operator* accuracy
(Laplacian/gradient error on a manufactured solution), closing the loop from
"good blue noise" to "good RBF-FD weights".

---

## 10. Validation assets

- `validate_cavity.jl` (repo root): annular cavity through
  Octree→repel→kick→cull. Clean geometry (4416 facets, area-asserted at export
  and re-import, orientation-guarded). Flags:
  `--placement=random|jittered|lattice|bridson`, `--resample-boundary`,
  `--save-vtk`. True-annulus direct pipeline: PASS raw (sep/Δ 0.750, CV 0.056,
  coord 12.1, fill max 1.18Δ, 0 singular, 0 culled).
- `examples/octree_boundary_layer.jl` (+ `_legacy.jl`): the production rung —
  bunny.stl + `BoundaryLayerSpacing`, the harness behind the §2 numbers.
- `test/` — `surface_sampling.jl`, `octree.jl` (`:bridson`, smart cap,
  Float32→Float64 promotion), `octree_isinside.jl` (pseudonormal feature sign,
  signed-volume guard), `repel.jl` (clipped force, cv_target/stall stopping).
- Last full `Pkg.test()`: **189,673 pass / 2 broken (pre-existing) / 0 failed**
  — re-run after the gap_tracked decision before opening the PR.

> Retired scratch harnesses (in git history): `bench_isinside.jl`,
> `diagnose_repel_quality.jl`, `profile_repel.jl`, `shape_opt_tradeoff.jl`,
> `validate_repel.jl`, and the consolidated markdowns
> (`cavity_validation_findings.md`, `octree_nn_assessment.md`,
> `plan_octree_nn_search.md`, `repel_convergence_ideas.md`).
