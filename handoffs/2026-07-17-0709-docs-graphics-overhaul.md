---
slug: docs-graphics-overhaul
created: 2026-07-17-0709
status: done
---

# Handoff: README/docs graphics overhaul — design approved, implementation not started

## Goal / why this matters
Kyle finds the current README/docs images boring and story-less (same red/blue
bunny from the same camera angle everywhere). The design for a replacement set
is **already approved** (see Decisions). The task is to implement it: extend
`docs/generate_images.jl` with new figure generators, render the assets, and
wire them into the README and four doc pages.

## Background & current state
Branch: `docs-beautification` (docs recently moved to VitePress via
DocumenterVitepress). All doc/README images are **static, committed** assets
generated manually (never in CI) by `docs/generate_images.jl` (CairoMakie —
NOT the package's `visualize` Makie extension, which exists in
`ext/WhatsThePointMakieExt.jl` but is unused by the doc pipeline).

Current images:
- README: `docs/src/assets/hero-banner.png` (3-panel bunny: surface → cutaway →
  stencil), `docs/src/assets/turntable.gif` (2.4 MB rotating bunny), logo.
- Docs: `docs/src/public/hero.png` (index hero), `assets/bunny-boundary.png`,
  `assets/bunny-discretized.png` (index Quick Example), `assets/2d-discretization.png`
  (quickstart), `assets/algorithm-comparison.png` (discretization.md),
  `assets/repel-comparison.png` (repel.md).
- Pages with NO figures: `octree.md`, `isinside_octree.md`, `boundary_normals.md`,
  `concepts.md`, `guide.md`.
- Orphaned assets (candidates for deletion): `docs/src/assets/bunny.jpeg`,
  `docs/src/assets/logo.png` (the referenced copy is `public/logo.png`).

`docs/generate_images.jl` reusable helpers: `build_scene`, `boundary_panel!`,
`cutaway_panel!`, `stencil_panel!`, `bunny_silhouette`/`silhouette_cloud` (2D),
`scatter_cloud2d!`, `bare_ax3`/`bare_ax2`, `depth_sorted_meshscatter!`,
`save_atomic` (iCloud-safe write), and an ffmpeg palette-quantized GIF pipeline
with a ~5 MB budget. Shared palette: boundary red `#CB3C33`, volume slate blue
`#4F7CAC`, stencil purple `#9558B2`, gray context.

Geometry assets: `docs/src/assets/bunny.stl` (17.6 MB, committed),
`test/data/bifurcation.stl` (1.24 MB), `test/data/cavity.stl`, `test/data/box.stl`,
plus `dev/vessel.stl` (3.0 MB) and `dev/bifurcation.stl` — **check whether
`dev/` is gitignored** before referencing it; prefer the committed
`test/data/bifurcation.stl` if it's the same geometry.

## Decisions & conclusions
Approved by Kyle via interactive Q&A (do not relitigate):

1. **Geometry anchor — mix.** Bunny stays as the recognizable README brand;
   the vessel/bifurcation geometry carries the applied CFD/meshless story in
   docs figures.
2. **Field coloring — honest derived quantities only.** Color points only by
   quantities the package itself computes (local spacing h(x), d_NN/h,
   distance-to-boundary). NO PDE-solution renders, real or fabricated.
3. **Scope — README + 4 flagship doc figures:**
   - spacing-graded bifurcation fill (`BoundaryLayerSpacing`, points colored by
     local spacing or d_NN) → `discretization.md`
   - octree subdivision visualization (leaf wireframes over geometry) → `octree.md`
   - normal orientation before/after (Hoppe MST+DFS) → `boundary_normals.md`
   - point-in-volume classification (inside/outside sample points) → `isinside_octree.md`
   Explicitly NOT in scope: figures for `concepts.md`/`guide.md`, metrics
   histograms (that was the rejected "full sweep" option).
4. **README animation — replace `turntable.gif`** with a "pipeline build-up"
   GIF: surface points sample on → interior fills → repel visibly relaxes →
   stencil connectivity flashes on. Also refresh `hero-banner.png` into an
   annotated pipeline strip.

## What's left / next steps
1. **Verify package APIs first** (a Plan agent attempting this stalled and died
   — none of this is verified; read the source, don't trust guessed names):
   - `src/octree/spatial_octree.jl`, `src/octree/triangle_octree.jl`: how to
     iterate leaves and get their boxes + inside/outside/intersecting
     classification for wireframe plotting.
   - `src/normals.jl`: how to get unoriented (post-PCA, pre-orientation)
     normals vs oriented ones, and per-point normal access for quiver arrows.
   - `src/isinside.jl`: signatures, incl. the TriangleOctree-accelerated path.
   - `src/spacings.jl`: evaluating `BoundaryLayerSpacing` at arbitrary points.
   - `src/repel.jl`: getting intermediate clouds for GIF frames (likely chunked
     calls with small `max_iters`, feeding output back in; note `kick_after`
     randomness).
   - `src/metrics.jl` + `src/neighbors.jl`: per-point d_NN (metrics may only
     return summaries; if so use the search API).
   - `src/topology.jl`: stencil edges for the connectivity flash frames.
2. Extend `docs/generate_images.jl`: one `generate_*` function per new figure,
   following existing naming/helper style; remove `generate_turntable`, add
   `generate_pipeline_gif`; update `main()` ordering.
3. Run `julia --project=docs docs/generate_images.jl` (manual, never CI).
4. Update image references + captions: `README.md` (hero-banner section and the
   turntable spot under "Quick Example"), `docs/src/discretization.md`,
   `docs/src/octree.md`, `docs/src/boundary_normals.md`,
   `docs/src/isinside_octree.md`.
5. Delete replaced/orphaned assets (`turntable.gif`, `bunny.jpeg`, unreferenced
   `assets/logo.png`) and sanity-check total committed image weight.

## Gotchas / constraints
- Repo lives in iCloud Drive: `save_atomic` in the script exists because plain
  saves flake there. In shell commands, always double-quote the absolute path
  (`"…/Mobile Documents/…"`), never backslash-escape spaces.
- If `Pkg` operations error on a nonexistent `/home` dev path, delete the
  gitignored `test/Manifest.toml` and retry (known stale-manifest iCloud issue).
- GIF budget ≈ 5 MB using the script's existing ffmpeg palette pipeline; the
  old turntable was 2.4 MB, and the pipeline build-up GIF has more visual
  change per frame, so watch size (fewer frames / smaller resolution if needed).
- The doc pipeline deliberately does NOT use the package's `visualize`
  extension (bespoke palette/shading instead) — keep it that way for
  consistency.
- Doc pages embed pre-rendered PNGs next to non-executed ```julia fences; keep
  that pattern (no `@example` Makie blocks).
