```@meta
CurrentModule = WhatsThePoint
```

# Octree Algorithm

`Octree` is a 3D discretization algorithm that uses a spacing function (for example `BoundaryLayerSpacing`) to adapt point density:

- finer near walls/boundaries,
- coarser in the bulk interior.

This is useful for CFD and boundary-layer-dominated meshless simulations where you need high resolution close to surfaces without over-resolving the full volume.

## Basic Usage

```julia
using WhatsThePoint
using GeoIO
using Unitful: m

mesh = GeoIO.load("model.stl").geometry
boundary = PointBoundary(mesh)

spacing = BoundaryLayerSpacing(
    points(boundary);
    at_wall=0.6m,
    bulk=4.0m,
    layer_thickness=8.0m,
)

alg = Octree(mesh)
cloud = discretize(boundary, spacing; alg)   # max_points auto-estimated
```

## Bridson Placement (default)

The default placement mode, `:bridson`, runs a single global advancing-front Poisson-disk pass (Bridson 2007) graded to the spacing field `h(x)`: every generated point keeps a distance of at least `min(rᵢ, rⱼ)` with `r = bridson_factor·h(x)` from every other point — including the boundary seeds — by construction. The front saturates on its own, so no refinement or repulsion pass is needed afterward, and `max_points` acts as a non-truncating cap (auto-estimated from the spacing integral when unset; a warning fires if a hand-set cap truncates the front).

With `max_growth > 0`, the prescribed spacing is replaced by its gradient-limited (Lipschitz) envelope before sampling, so steep boundary layers transition smoothly instead of jumping — `0.1`–`0.2` matches typical CFD growth ratios of 1.1–1.2.

A too-coarse spacing (one the domain cannot host an interior at) is clamped with a loud warning instead of silently producing an empty cloud; run [`suggest_spacing`](@ref) first to pick a viable spacing deliberately.

## Parameters

The constructor supports the same octree controls used elsewhere, plus placement options for candidate generation:

- octree refinement controls (`tolerance_relative`, `min_ratio`, `node_min_ratio`, `alpha`)
- placement mode (`:bridson` default, or per-leaf `:random`, `:jittered`, `:lattice`)
- Bridson disk radius relative to spacing (`bridson_factor`, default 0.75)
- gradient-limited spacing (`max_growth`, default 0 = off)
- boundary leaf oversampling (`boundary_oversampling`, per-leaf modes only)
- orientation and safety checks (`verify_orientation`, etc.)

See [`Octree`](@ref) and [`BoundaryLayerSpacing`](@ref) in the API reference for the exact signatures.

## Included Example Script

A full runnable example is included in this repository:

- [examples/octree_boundary_layer.jl](https://github.com/JuliaMeshless/WhatsThePoint.jl/blob/main/examples/octree_boundary_layer.jl)

The script demonstrates:

1. probing the geometry with `suggest_spacing`,
2. Poisson-disk boundary sampling and a steep gradient-limited `BoundaryLayerSpacing`,
3. running `Octree` (Bridson placement, auto point budget),
4. checking quality (`spacing_fidelity_metrics`, `metrics`) and rendering cross-section PNGs with CairoMakie,
5. writing a ParaView `.vtu` via `export_vtk`.

## Notes

- `Octree` is 3D only.
- For uniform spacing, `SlakKosec` is a good alternative.
- If your model scale changes significantly, tune `at_wall`, `bulk`, and `layer_thickness` in physical units.
