# Getting Started

This guide walks through the core workflow: importing a surface, generating volume points, optimizing the distribution, and preparing the point cloud for a meshless solver.

## Installation

```julia
] add https://github.com/JuliaMeshless/WhatsThePoint.jl
```

## Importing a Surface

Load a surface mesh (STL, OBJ, or any format supported by [GeoIO.jl](https://github.com/JuliaEarth/GeoIO.jl)) as a `PointBoundary`:

```julia
using WhatsThePoint

boundary = PointBoundary("model.stl")
```

The boundary uses **face centers** as points (not mesh vertices). Each surface element stores a point, normal vector, and area.

## Inspecting the Boundary

```julia
# Number of boundary points
length(boundary)

# Access named surfaces
names(boundary)              # surface names
surf = surfaces(boundary)[1] # first surface

# Surface element data
points(surf)   # point coordinates
normals(surf)  # normal vectors
areas(surf)    # element areas
```

## Surface Operations

Split surfaces at normal angle discontinuities to identify distinct geometric faces:

```julia
split_surface!(boundary, 75°)
```

Combine multiple surfaces back into one:

```julia
combine_surfaces!(boundary, "surface_1", "surface_2")
```

## Generating Volume Points

Generate volume points from a boundary using `discretize`. The algorithm choice depends on the problem dimension.

### 3D Algorithms

```julia
spacing = ConstantSpacing(1mm)

# SlakKosec (default for 3D) — sphere-based candidate generation
cloud = discretize(boundary, spacing; alg=SlakKosec())

# VanDerSandeFornberg — grid projection with sphere packing
cloud = discretize(boundary, spacing; alg=VanDerSandeFornberg(), max_points=100_000)

# OctreeRandom — octree-guided random generation (no spacing needed)
cloud = discretize(boundary, OctreeRandom("model.stl"; min_ratio=1e-6))
```

`SlakKosec` can also accept a `TriangleOctree` for accelerated point-in-volume queries:

```julia
octree = TriangleOctree("model.stl"; min_ratio=1e-6)
cloud = discretize(boundary, spacing; alg=SlakKosec(octree))
```

### 2D Algorithm

```julia
# FornbergFlyer (default and only option for 2D)
cloud = discretize(boundary, spacing; alg=FornbergFlyer())
```

### Spacing Types

Control point density with different spacing strategies:

```julia
# Uniform spacing everywhere
spacing = ConstantSpacing(1mm)

# Variable spacing — denser near boundary, coarser in interior
spacing = LogLike(cloud, 0.5mm, 1.2)  # base_size, growth_rate
```

See the [Discretization](@ref) page for detailed descriptions of each algorithm and spacing type.

## Node Repulsion

Optimize the point distribution using the repulsion algorithm (Miotti 2023). This improves point regularity which is important for meshless PDE accuracy:

```julia
cloud, convergence = repel(cloud, spacing; β=0.2, max_iters=1000)
```

`repel` returns a tuple of `(new_cloud, convergence_vector)`. The new cloud has `NoTopology` since points have moved.

## Topology (Point Connectivity)

Meshless methods require a stencil (set of neighbor points) for each point. Compute this with `set_topology`:

```julia
# k-nearest neighbors
cloud = set_topology(cloud, KNNTopology, 21)

# Or radius-based neighbors
cloud = set_topology(cloud, RadiusTopology, 2mm)

# Access neighbor indices
neighbors(cloud, 5)   # neighbors of point 5
neighbors(cloud)       # all neighborhoods

# Check state
hastopology(cloud)     # true
```

Topology can also be set on individual surfaces and volumes:

```julia
surf = set_topology(surf, KNNTopology, 10)
neighbors(surf, i)  # local indices within the surface
```

## Visualization

Visualize point clouds and boundaries using [Makie.jl](https://github.com/MakieOrg/Makie.jl):

```julia
using GLMakie

visualize(cloud; markersize=0.15)
visualize(boundary; markersize=0.15)
visualize_normals(boundary)
```

## Export

Save a point cloud to VTK format for use in external tools:

```julia
export_cloud("output.vtk", cloud)
```
