# Direct point-cloud generation: assess → generate → export

This PR brings node generation closer to maturity: a single call produces a
good quality point cloud from an STL.

---

## How it works now (the practical view)

### 1. Assess the geometry and pick a spacing — `suggest_spacing`
You no longer have to guess a node spacing and try it blindly.
`suggest_spacing` probes the geometry (extent, enclosed volume,
shortest feature) and recommends a sensible baseline:

```julia
g = suggest_spacing("model.stl")
# geometry: model.stl
#   extent     86 × 65 × 63 m   (min axis 63 m)
#   volume     ≈ 129000 m³
#   h_ceiling  42 m   (coarser ⇒ EMPTY interior — stay below)
#   h_baseline 6.3 m  (≈ 520k vol pts) ← start here
#   h_fine     3.2 m  (≈ 4.1M vol pts)
```

It returns those values so they drop straight into generation, and it also acts
as a guard: a spacing too coarse for the domain to host an interior is caught
(loudly), never silently producing an empty cloud.

### 2. Generate the cloud — `discretize` (Bridson, zero refinement)
A single call produces the final cloud. The Bridson sampler (below) places
points at the requested spacing by construction, so there is no relaxation or
refinement pass afterward— what comes out is what you use. The point budget
is auto-sized from the spacing, so the interior fills completely (no caps to set).

```julia
mesh    = GeoIO.load("model.stl").geometry
spacing = ConstantSpacing(g.h_baseline)              # or a graded BoundaryLayerSpacing
bnd     = PointBoundary(mesh, spacing)               # Poisson-disk surface sampling
cloud   = discretize(bnd, spacing; alg = Octree(mesh))   # Bridson volume fill — final cloud
```

Spacing can be constant or **graded** — e.g. fine at the walls, coarse in the
bulk (`BoundaryLayerSpacing`), with an optional smoothness limiter (`max_growth`)
so the density transitions gently instead of jumping.

### 3. Export for ParaView — `export_vtk`
Write a `.vtu` you open directly in ParaView (Representation → *Point Gaussian*):

```julia
export_vtk("cloud", cloud)                                  # geometry
export_vtk("sol", cloud; fields = ("T" => temp, "U" => u))  # after solving
```

Every point carries `point_type` (boundary vs volume) and `surface_id` (colour
by named surface). After a solve, pass your result arrays as `fields` and view
them on the points like any CAE result.

---

## What the pieces actually do

### Poisson-disk sampling (and why)
A **Poisson-disk** point set is one where points are placed as densely as
possible while keeping every pair at least a minimum distance apart — "blue
noise". The result covers space evenly with **no clumps and no gaps**, which is
exactly what a meshless solver (RBF-FD) needs: clumped points give
near-duplicate, ill-conditioned stencils; gaps leave the domain under-resolved.

This PR applies it in two places:

- **Boundary (surface) sampling** — `PointBoundary(mesh, spacing)` throws darts
  *on the triangle surface*, producing evenly spaced wall points at the target
  spacing. This replaces the old "use the triangle face centres" import, whose
  spacing was dictated by the mesh, not by you.
- **Volume filling (Bridson)** — a single global advancing front (Bridson 2007)
  fills the interior, **seeded from the boundary points** so volume points also
  keep their distance from the wall. It is *graded*: the minimum distance follows
  the spacing field `h(x)`, so you get a fine boundary layer flowing smoothly
  into a coarse bulk. The front stops on its own once the domain is saturated —
  that is why no refinement step is needed and why the point count is automatic.

### What it's optimised for (the quality metrics)
The cloud is tuned for the properties that keep a meshless PDE solver accurate
and stable. In plain terms:

- **Separation** — the closest any two points get. If it's tiny, two points are
  nearly on top of each other and the local stencil goes singular. Target: a
  fixed fraction (~0.75) of the requested spacing.
- **Spacing fidelity (CV)** — how closely the *actual* local spacing matches the
  spacing you asked for. Reported as a coefficient of variation; ~0.05 means "the
  points really are at the density requested, very uniformly" (lower is better).
- **Coordination** — how many close neighbours each point has on average
  (~12–15 is healthy) — the meshless analogue of how many cells touch a node; it
  sets how well-supported each stencil is.
- **Mesh ratio** (largest gap ÷ smallest gap) — a single uniformity number;
  close to 1 means blue-noise-even, large means clusters and voids coexist.

Together these say: *points are evenly spaced, none too close or too far, at the
density you requested.* `metrics`, `spacing_metrics`, and
`spacing_fidelity_metrics` report them so you can verify a cloud at a glance.

### Repulsion (`repel`) — now a bonus, not a requirement
Previously, generation produced a rough cloud and then **node repulsion** pushed
points apart over many iterations to reach acceptable quality. 
Now Bridson delivers that quality directly, so `repel` becomes optional: 
nice to have for specific applications, never needed to get a usable cloud. 
Its default force was also switched to repulsion-only
(`ClippedSpacingForce`), it was found that this *preserves* a good cloud instead of slowly
condensing it into clusters and voids.

---

## Notable fixes in this PR
- Bridson never returns a silent empty cloud — too-coarse spacing is clamped and warned.
- Auto point-budget no longer truncates the inward-advancing front, which had left the deep interior empty (cap headroom widened).
- Gradient-limited spacing (`max_growth`) no longer crashes under multithreading.

## Result & validation
- Stanford bunny (69,664 facets, ~86 m), steep boundary layer: **~1.10M graded volume points**, spacing CV ≈ 0.05, separation/spacing ≈ 0.75, coordination ≈ 15 — threaded, full interior, no tuning.
- New tests for spacing guidance, `export_vtk`, surface sampling, octree and isinside; full `Pkg.test()` green; Runic-clean.

