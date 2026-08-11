---
slug: rbf-implicit-surface-sdf
created: 2026-07-26-0859
status: open
---

# Handoff: RBF implicit surface + SDF representation for WhatsThePoint.jl

## Goal / why this matters

WhatsThePoint represents geometry two ways today, both discrete: a triangle mesh
(`SimpleMesh` / `TriangleOctree`) and a point sample (`PointSurface` / `PointBoundary`).
Every geometric query — projection, inside/outside, normals, curvature — is answered
against facets or a local PCA fit of neighbouring points. Both are piecewise, so projected
points snap to facets, normals are noisy, and curvature is a finite-difference estimate.

Build a third representation: a continuous, differentiable scalar field `f: ℝ³ → ℝ` whose
zero level set *is* the surface, fitted with RadialBasisFunctions.jl. That single object
answers all of the above analytically, and it is simultaneously the SDF — so "RBF
parameterization of a surface mesh" (the NURBS analogy) and "SDF representation" are the
same deliverable, not two.

**This increment builds the object and a validation example. It rewires nothing.**

## Background & current state

Nothing implemented yet. This is a design handoff from a brainstorming session — no code
was written, no files touched. The decisions below were made with the user and should be
treated as settled, not relitigated:

- **Implicit first, deformation later.** Shape-deformation parameterization (the actual
  NURBS-for-optimization use case) gets layered on afterwards — you deform the field, not
  the points, so the field must exist first.
- **Home: a package extension.** No new hard dependency on WhatsThePoint.
- **Fit strategy: greedy center selection** (Carr et al. 2001), behind an abstract API so a
  partition-of-unity backend can slot in later without touching consumers.
- **Scope stops at the object + a runnable example.** Do NOT touch `repel`, `isinside`,
  `sample_surface`, or spacing. Those get rewired in a later increment, once there are
  measured fidelity numbers to justify it.

## The math

**Construction** — Carr et al., *Reconstruction and Representation of 3D Objects with
Radial Basis Functions*, SIGGRAPH 2001. Given surface samples `sᵢ` with outward unit
normals `n̂ᵢ`, build three constraints per selected sample:

```
f(sᵢ)          =  0        on-surface
f(sᵢ + δᵢ n̂ᵢ)  = +δᵢ       outside
f(sᵢ − δᵢ n̂ᵢ)  = −δᵢ       inside
```

The off-surface pairs prevent the trivial solution `f ≡ 0`. Giving them the values `±δᵢ`
rather than `±1` makes `f` approximately distance-valued near the surface for free.
`δᵢ` starts at 1% of the bbox diagonal and halves until the nearest surface sample to
`sᵢ ± δᵢ n̂ᵢ` is still `sᵢ` (Carr's validity test) — one KD-tree query per candidate.

Fit is `Interpolator(constraint_points, constraint_values, PHS(1; poly_deg=1))`.
`φ(r) = r` is the biharmonic kernel in ℝ³ and the one Carr used; its linear growth away
from the surface is what makes the field behave like a distance function. `PHS3`'s `r³`
grows cubically and degrades the far field. `PHS3` is one keyword away and the example
compares them.

**Greedy center selection** — a surface has far fewer degrees of freedom than samples, so
put centers where the error actually is:

```
centers ← Poisson-disk subset of the surface samples (~500)
loop:
    fit                                       O(M³) dense, M = 3·|centers| + 4
    r ← |f(sᵢ)| at ALL N surface samples      O(N·M), matrix-free, tmap
    max(r) ≤ tol·L  →  stop                   L = bbox diagonal
    centers ← centers ∪ {K worst offenders}   min-spacing filtered
```

Centers migrate to creases and high curvature, stay sparse on flat regions — the direct
analog of NURBS knot insertion. Fidelity is a tolerance you set, not a thing you hope for.

**Derived quantities** (Goldman, *Curvature formulas for implicit curves and surfaces*,
CAGD 2005 for the curvature forms):

```
n̂(x)   = ∇f/‖∇f‖
sdf(x)  = f/‖∇f‖                                    eikonal normalization
project: x ← x − f∇f/‖∇f‖²                          Newton on the level set, 2–3 iters
κ_mean  = (∇fᵀH∇f − ‖∇f‖²·tr H) / (2‖∇f‖³)
κ_gauss = (∇fᵀ·adj(H)·∇f) / ‖∇f‖⁴
```

Sanity check for the curvature formulas, `f = ‖x‖ − R`: `κ_gauss = 1/R²` and
`|κ_mean| = 1/R`. Sign convention on `κ_mean` follows the `+∇f` normal direction —
pin it empirically against the torus in the example and document it.

## Key files / locations

### Verified API facts (already checked — do not re-derive)

- `~/dev/RadialBasisFunctions` (v0.6.0, registered). `Interpolator` at
  `src/interpolation.jl` is a **global dense solve** — `bunchkaufman!` on an
  `(N + npoly)²` symmetric-indefinite matrix. Stores `x`, `rbf_weights`,
  `monomial_weights`, `rbf_basis`, `monomial_basis` as public fields.
- **`Interpolator` has no gradient/Hessian evaluation API.** The extension must assemble
  `∇f` and `∇²f` itself from those stored fields.
- The pieces to assemble them already exist and are tested upstream:
  - `RadialBasisFunctions.∇{<:AbstractRadialBasis}` and `H{...}` functors, evaluated as
    `op(x, xᵢ)` → gradient vector / Hessian `SMatrix`. See
    `src/basis/polyharmonic_spline.jl` (PHS1 implements `∂`, `∇`, `H` explicitly).
  - `∂(::MonomialBasis, dim)` and `H(::MonomialBasis)` factories returning `ℒMonomialBasis`
    (`src/operators/monomial/monomial.jl`). Note the two differentiation protocols are
    deliberately different: functor **structs** for radial bases, factory **functions** for
    monomials — the module docstring says so, don't try to unify them.
  - `_get_monomial_basis(::Val{3}, ::Val{1})` exists — 3D linear poly is fine.
- `WhatsThePoint._compute_signed_distance_octree` (`src/octree/triangle_octree.jl:522`) is
  an **exact** pseudonormal signed distance (Bærentzen & Aanæs 2005). This is the validation
  oracle — free ground truth on real geometry, no need to validate only against analytic shapes.
- `PointSurface` stores `geoms::StructVector` with `.point`, `.normal`, `.area`
  (`src/surface.jl`) — the fit input already exists in the right shape.
- Extension pattern to copy: `ext/WhatsThePointMakieExt.jl` (defines methods on
  `WhatsThePoint.visualize`, a stub declared in the parent).
- Test harness: TestItemRunner, `@testitem` blocks, each self-contained with its own
  `using`. `test/testsetup.jl` has `CommonImports` / `TestData` / `OctreeTestData` snippets.
  `test/Project.toml` holds test deps. No way to run a single test file in isolation.

### Files to create / change

| File | Change |
|---|---|
| `src/implicit.jl` | new — abstract type, generic API, docstrings, stub, octree `sdf` method |
| `src/WhatsThePoint.jl` | add `include("implicit.jl")` + exports |
| `ext/WhatsThePointRadialBasisFunctionsExt/*.jl` | new — 4 files, see layout below |
| `Project.toml` | `RadialBasisFunctions` → `[weakdeps]`, `[extensions]`, `[compat] = "0.6"` |
| `test/implicit_surface.jl` | new — `@testitem`s |
| `test/Project.toml` | add `RadialBasisFunctions` |
| `examples/rbf_implicit_surface.jl` | new — the exploration script |

### Architecture

```
src/implicit.jl                                  ← generic, no RBF dependency
  AbstractImplicitSurface{M,C}
  implicit_surface(...)      stub → informative error if RBF.jl not loaded
  sdf, implicit, gradient, normal, curvature, project, isinside   ← generic API
  sdf(::TriangleOctree, p)   exact oracle, wraps _compute_signed_distance_octree

ext/WhatsThePointRadialBasisFunctionsExt/
  WhatsThePointRadialBasisFunctionsExt.jl   module + includes
  constraints.jl   δ selection (Carr validity test) + constraint assembly
  greedy.jl        center selection loop, residual sweep, convergence/cap
  field.jl         f, ∇f, ∇²f from a fitted Interpolator
  surface.jl       RBFImplicitSurface <: AbstractImplicitSurface + API methods
```

Four files, not one — each stays under ~150 lines with one job. `RBFImplicitSurface` is
concrete in the ext, so it cannot be exported from the parent; users reach it through the
`implicit_surface(...)` constructor declared in `src/implicit.jl`.

## Decisions & conclusions

- **Coordinate normalization.** RBF conditioning is scale-dependent. Fit in a normalized
  frame `x̃ = (x − c)/L` (`c` = bbox center, `L` = bbox diagonal); store `(c, L, unit)` on
  the type and map back on evaluation. Makes conditioning independent of mm vs. km.
- **Units.** RBF.jl operates on bare `SVector`s. Strip Unitful at the extension boundary
  (`ustrip`), re-attach on output, so `sdf` returns a `Quantity` in the surface's length
  unit. Preserve machine type (`Float32`/`Float64`) — match the `test/float32_pipeline.jl`
  precedent.
- **Orientation guard.** After fitting, evaluate `f` outside the bbox; if `f < 0`, negate
  the weights. `orient_normals!` (Hoppe MST+DFS) gives *consistent* orientation, not
  necessarily outward, so this is a real failure mode, not a hypothetical.
- **Closed vs. open.** `implicit_surface(::PointBoundary)` fits the union of all named
  surfaces — closed, signed, `isinside` meaningful. `implicit_surface(::PointSurface)` fits
  a single open patch — projection and normals valid, sign is **not**; document that and do
  not define `isinside` for it.
- **`sdf(::TriangleOctree, p)` is a deliberate small addition**, not scope creep: the
  example needs the oracle, and making `sdf` a coherent generic beats reaching into a
  private function from a public example.

## What's left / next steps

Build in this order — each step is testable before the next exists.

1. `src/implicit.jl` — API and docstrings **first**, so the contract is fixed before the fit.
2. `constraints.jl` + `field.jl` — δ selection and field evaluation. Testable against a
   hand-fitted sphere before any greedy logic exists.
3. `greedy.jl` — the selection loop.
4. `surface.jl` — assemble `RBFImplicitSurface`, wire the API methods.
5. `examples/rbf_implicit_surface.jl` — the exploration artifact:
   - **Sphere**, Fibonacci-sampled with exact analytic normals. Plot
     `max|sdf − (‖x‖−R)|` vs. center count. Isolates fit error from normal error.
   - **Torus** — analytic mean and Gaussian curvature. Validates the Goldman formulas and
     pins the sign convention.
   - **`bunny.stl` / `test/data/bifurcation.stl`** — report fit wall time, final center
     count, `max|f|` on surface samples, `sdf` error vs. `sdf(::TriangleOctree, ·)` on a
     random probe cloud, and sign agreement with `isinside`.
   - **Projection** — Newton-project a perturbed cloud; report Hausdorff distance to the
     mesh and iteration counts. Directly measures the payoff for `repel` later.
   - **`PHS1` vs `PHS3`** — same geometry, both bases: near-surface accuracy, far-field SDF
     quality, curvature smoothness. Settles the default with data.
   - **Resample** — project a coarse point set onto the zero level set and `visualize` it.
6. `test/implicit_surface.jl`:
   - Sphere: `sdf` within tolerance of analytic at on- and off-surface probes
   - Sphere/torus: `normal` agrees with analytic to a fixed angle tolerance
   - Torus: mean and Gaussian curvature within tolerance of analytic
   - `project` returns a point with `|f| < tol`; idempotent on an already-projected point
   - Greedy loop reaches the requested tolerance, returns fewer centers than input samples
   - Inward-oriented normals still produce an outward-positive field (orientation guard)
   - `isinside` agrees with the `TriangleOctree` result on `box.stl` probes
   - Float32 input yields a Float32 field
   - `@test_throws` on an open `PointSurface` passed to `isinside`
   - Informative error from `implicit_surface` when RadialBasisFunctions isn't loaded

Per the repo convention: **write the tests, but do not run them unless asked.**

## Gotchas / constraints

- **Dense-solve ceiling.** The solve is `3·|centers| + 4` square. ~2–3k selected centers
  (~9k constraints, ~650 MB, tens of seconds) is the practical laptop ceiling.
  `bifurcation.stl` (24,780 samples) should converge well inside that. The greedy loop
  must `@warn` and stop at a `max_centers` cap rather than silently returning a bad fit.
- **`PHS1` is singular at a center.** `φ = r` has a kink at `r = 0`, so the Hessian blows
  up *exactly at* a center. Curvature queries need a guard. `PHS3` is C² everywhere —
  hence the basis comparison in the example.
- **Duplicate centers make the system singular.** The greedy loop must not re-add a point
  already in the center set, and must enforce a minimum spacing among newly added points
  (otherwise a crease dumps 50 near-coincident centers in one iteration).
- **No secrets involved in this work** — nothing to redact.
- **Out of scope, do not drift into it:** rewiring `repel` / `isinside` / `sample_surface` /
  curvature-driven spacing; a partition-of-unity backend; fast-multipole acceleration of
  evaluation (currently O(M) per query); the deformation/NURBS-for-optimization layer.
- **Two upstream opportunities for RadialBasisFunctions.jl** — note them, don't act on them
  here: (1) `Interpolator` has no gradient/Hessian evaluation API; (2) Hermite/gradient
  constraints (`∇f(sᵢ) = n̂ᵢ`, Macêdo et al. 2011) would eliminate the off-surface points
  and the δ heuristic entirely, but RBF.jl's Hermite machinery currently lives in the
  operator path, not the interpolator.

## References

- Carr et al. (2001), *Reconstruction and Representation of 3D Objects with Radial Basis
  Functions*, SIGGRAPH — the construction and the greedy algorithm (§4.3)
- Goldman (2005), *Curvature formulas for implicit curves and surfaces*, CAGD — the
  curvature forms
- Bærentzen & Aanæs (2005) — the pseudonormal SDF already implemented in `TriangleOctree`
- Ohtake et al. (2003), *Multi-level Partition of Unity Implicits* — the POU escape hatch
- Macêdo et al. (2011), *Hermite Radial Basis Functions Implicits* — the gradient-constraint
  alternative
