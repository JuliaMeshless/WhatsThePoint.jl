```@meta
CurrentModule = WhatsThePoint
```

# Guide

This guide walks through the core workflow: importing a surface, generating volume points, optimizing the distribution, and preparing the point cloud for a meshless solver.

## Importing a Surface

Load a surface mesh (STL, OBJ, or any format supported by [GeoIO.jl](https://github.com/JuliaEarth/GeoIO.jl)) as a `PointBoundary`:

```julia
using WhatsThePoint

boundary = PointBoundary("model.stl")
```

!!! note "Face centers, not vertices"
    When importing a mesh, WhatsThePoint uses **face centers** as boundary points rather than mesh vertices. Each face becomes a [`SurfaceElement`](@ref) storing its center point, outward normal, and area. This gives a more uniform boundary representation than raw vertices.

## Inspecting the Boundary

```julia
# Number of boundary points
length(boundary)

# Access named surfaces
names(boundary)              # surface names
surf = surfaces(boundary)[1] # first surface

# Surface element data
points(surf)   # point coordinates
normal(surf)   # normal vectors
area(surf)     # element areas
```

## Surface Operations

Identify distinct geometric faces (walls, inlets, outlets) so you can apply different boundary conditions to each. Split surfaces at normal angle discontinuities:

```julia
split_surface!(boundary, 75°)
```

This builds a k-nearest neighbor graph on the surface, removes edges where adjacent normals differ by more than the threshold angle, and labels each connected component as a separate named surface. See the [Boundary & Normals](boundary_normals.md) page for details on normals, splitting, shadow points, and more.

Combine multiple surfaces back into one:

```julia
combine_surfaces!(boundary, "surface_1", "surface_2")
```

## Shadow Points (Optional)

Virtual points offset inward from the boundary, used in Hermite-type boundary condition enforcement (e.g., Hermite RBF-FD). Shadow points sit just inside the domain along the inward normal direction.

```julia
shadow = ShadowPoints(0.5mm)
shadow_pts = generate_shadows(surf, shadow)
```

See the [Boundary & Normals](boundary_normals.md) page for more shadow point options including variable offsets.

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
cloud = discretize(boundary, OctreeRandom("model.stl"; h_min=0.5))
```

`SlakKosec` can also accept a `TriangleOctree` for accelerated point-in-volume queries:

```julia
octree = TriangleOctree("model.stl"; h_min=0.5)
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
```

After an initial discretization, you can use variable spacing for a second pass:

```julia
# Variable spacing — denser near boundary, coarser in interior
# Requires an existing PointCloud (uses boundary distances internally)
spacing = LogLike(cloud, 0.5mm, 1.2)  # base_size, growth_rate
cloud = discretize(boundary, spacing; alg=SlakKosec())
```

See the [Discretization](discretization.md) page for detailed descriptions of each algorithm and spacing type.

## Node Repulsion

Discretization gives approximate uniformity; repulsion refines it to minimize interpolation error in the meshless solver.

```julia
cloud, convergence = repel(cloud, spacing; β=0.2, max_iters=1000)
```

`repel` returns a tuple of `(new_cloud, convergence_vector)`. The new cloud has `NoTopology` since points have moved.

!!! note "Only volume points are repelled"
    Boundary points remain fixed — only volume (interior) points are moved during repulsion. This preserves the original boundary geometry.

!!! tip "Tuning repulsion"
    The default parameters (`β=0.2`, `k=21`, `max_iters=1000`) work well for most problems. Check `convergence[end]` to verify the distribution has stabilized. See the [Node Repulsion](repel.md) page for detailed parameter guidance.

### Verifying Distribution Quality

Use `metrics` to inspect the point distribution before and after repulsion:

```julia
metrics(cloud)  # prints distance statistics to k nearest neighbors
```

This prints the average, standard deviation, maximum, and minimum distances to each point's k nearest neighbors — useful for quantifying how uniform the distribution is.

## Topology (Point Connectivity)

Meshless solvers compute derivatives using local neighborhoods. Topology pre-computes these neighborhoods so they are ready when the solver needs them.

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

!!! note "Local vs global indices"
    Topology on a `PointSurface` or `PointVolume` uses **local indices** (1 through `length(surf)`). Topology on a `PointCloud` uses **global indices** (1 through `length(cloud)`) where boundary points come first, followed by volume points. See [Concepts](@ref) for details.

## Visualization

Visualize point clouds and boundaries using [Makie.jl](https://github.com/MakieOrg/Makie.jl):

```julia
using GLMakie

visualize(cloud; markersize=0.15)
visualize(boundary; markersize=0.15)
```

## Export

Save a point cloud to VTK format for use in external tools:

```julia
export_cloud("output.vtk", cloud)
```
