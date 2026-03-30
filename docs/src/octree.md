```@meta
CurrentModule = WhatsThePoint
```

# Octree

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
cloud = discretize(boundary, spacing; alg=alg, max_points=200_000)
```

## Parameters

The constructor supports the same octree controls used elsewhere, plus placement options for candidate generation:

- octree refinement controls (`tolerance_relative`, `min_ratio`)
- placement mode (`:random`, `:jittered`, or `:lattice`)
- boundary leaf oversampling (`boundary_oversampling`)
- orientation and safety checks (`verify_orientation`, etc.)

See [`Octree`](@ref) and [`BoundaryLayerSpacing`](@ref) in the API reference for the exact signatures.

## Included Example Script

A full runnable example is included in this repository:

- [examples/octree_boundary_layer.jl](https://github.com/JuliaMeshless/WhatsThePoint.jl/blob/main/examples/octree_boundary_layer.jl)

The script demonstrates:

1. building `BoundaryLayerSpacing`,
2. running `Octree`,
3. visualizing the result with Makie.

## Notes

- `Octree` is 3D only.
- For uniform spacing, `SlakKosec` is a good alternative.
- If your model scale changes significantly, tune `at_wall`, `bulk`, and `layer_thickness` in physical units.
