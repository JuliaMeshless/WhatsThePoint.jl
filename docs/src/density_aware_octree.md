```@meta
CurrentModule = WhatsThePoint
```

# Density-aware Octree

`DensityAwareOctree` is a 3D discretization algorithm that uses a spacing function (for example `BoundaryLayerSpacing`) to adapt point density:

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

alg = DensityAwareOctree(mesh)
cloud = discretize(boundary, spacing; alg=alg, max_points=200_000)
```

## Parameters

The constructor supports the same octree controls used elsewhere, plus placement options for candidate generation:

- octree refinement controls (`tolerance_relative`, `min_ratio`)
- placement mode (`:center` or `:jittered`)
- boundary leaf oversampling (`boundary_oversampling`)
- orientation and safety checks (`verify_orientation`, etc.)

See [`DensityAwareOctree`](@ref) and [`BoundaryLayerSpacing`](@ref) in the API reference for the exact signatures.

## Included Example Script

A full runnable example is included in this repository:

- [examples/density_aware_discretization.jl](https://github.com/JuliaMeshless/WhatsThePoint.jl/blob/main/examples/density_aware_discretization.jl)

The script demonstrates:

1. building `BoundaryLayerSpacing`,
2. running `DensityAwareOctree`,
3. colorizing generated volume points by a density proxy (`log10(1/h^3)`),
4. visualizing the result with Makie.

## Notes

- `DensityAwareOctree` is 3D only.
- For uniform spacing, `SlakKosec` and `OctreeRandom` are still good alternatives.
- If your model scale changes significantly, tune `at_wall`, `bulk`, and `layer_thickness` in physical units.
