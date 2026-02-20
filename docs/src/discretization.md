# Discretization

Volume discretization generates interior points from a boundary surface. The `discretize` function dispatches to dimension-specific algorithms.

```julia
cloud = discretize(boundary, spacing; alg=algorithm, max_points=10_000_000)
```

## Algorithm Overview

| Algorithm | Dimensions | Spacing Required | Description |
|-----------|-----------|-----------------|-------------|
| [`SlakKosec`](@ref) | 3D | Yes | Sphere-based candidate generation |
| [`VanDerSandeFornberg`](@ref) | 3D | Yes (`ConstantSpacing` only) | Grid projection with sphere packing |
| [`FornbergFlyer`](@ref) | 2D | Yes (`ConstantSpacing` only) | 1D projection with height-field fill |
| [`OctreeRandom`](@ref) | 3D | No | Octree-guided random point generation |

## SlakKosec

Default algorithm for 3D discretization. Generates candidate points on spheres around existing points, accepting those that are inside the domain and sufficiently far from existing points.

```julia
# Basic usage
cloud = discretize(boundary, spacing; alg=SlakKosec())

# Custom number of candidates per sphere (default: 10)
cloud = discretize(boundary, spacing; alg=SlakKosec(20))

# With octree acceleration for faster isinside queries
octree = TriangleOctree("model.stl"; h_min=0.5)
cloud = discretize(boundary, spacing; alg=SlakKosec(octree))
cloud = discretize(boundary, spacing; alg=SlakKosec(20, octree))
```

Supports both `ConstantSpacing` and variable spacings (`LogLike`).

## VanDerSandeFornberg

3D algorithm that projects a 2D grid onto the shadow plane and fills the volume layer by layer using sphere packing heights.

```julia
cloud = discretize(boundary, ConstantSpacing(1mm); alg=VanDerSandeFornberg())
```

Requires `ConstantSpacing`. Uses `isinside` (Green's function) for filtering generated points.

## FornbergFlyer

2D-only algorithm. Uses a similar height-field approach as VanDerSandeFornberg, but projects onto the x-axis for 2D domains.

```julia
cloud = discretize(boundary, ConstantSpacing(0.1mm); alg=FornbergFlyer())
```

This is the default (and only) algorithm for 2D boundaries.

## OctreeRandom

Generates volume points directly from an octree decomposition of the domain. The octree classifies leaf nodes as interior, boundary, or exterior. Interior leaves are filled with random points directly (100% acceptance rate), while boundary leaves are oversampled and filtered with the octree-accelerated `isinside` test.

**No spacing parameter is needed** — point density is controlled by the octree resolution (`h_min`).

```julia
# From a mesh file (recommended — auto-computes h_min)
cloud = discretize(boundary, OctreeRandom("model.stl"))

# With explicit h_min
cloud = discretize(boundary, OctreeRandom("model.stl"; h_min=0.5))

# From a pre-built TriangleOctree
octree = TriangleOctree("model.stl"; h_min=0.5)
cloud = discretize(boundary, OctreeRandom(octree))

# With custom boundary oversampling (default: 2.0)
cloud = discretize(boundary, OctreeRandom(octree, 3.0))
```

Parameters:
- `h_min` — Minimum octree box size. Auto-computed from mesh diagonal and triangle count if omitted.
- `max_triangles_per_box` — Maximum triangles per leaf before subdivision (default: 50).
- `boundary_oversampling` — Oversampling factor for boundary leaves (default: 2.0). Higher values improve boundary coverage at the cost of more rejected candidates.
- `verify_interior` — Verify generated interior points with `isinside` (default: `false`). Usually unnecessary since leaf classification is reliable.
- `verify_orientation` — Check mesh normal consistency before building the octree (default: `true`).

## Spacing Types

Spacing controls point density for algorithms that require it (all except `OctreeRandom`).

### ConstantSpacing

Uniform spacing everywhere in the domain:

```julia
spacing = ConstantSpacing(1mm)
```

Works with all algorithms.

### LogLike

Variable spacing that is denser near the boundary and coarser in the interior. Uses a logarithmic-like growth function:

```julia
spacing = LogLike(cloud, base_size, growth_rate)
```

- `base_size` — Spacing at the boundary surface.
- `growth_rate` — Rate at which spacing increases away from the boundary. Values > 1 create coarser interior points.

The spacing at a point is computed as `base_size * x / (a + x)` where `x` is the distance to the nearest boundary point.

Works with `SlakKosec` only.

### Power

!!! warning
    `Power` spacing is declared but not yet functional. Its constructor currently throws an error.
