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

To place boundary points at a spacing *you* choose instead of the one the tessellation dictates, Poisson-disk sample the surface:

```julia
mesh = GeoIO.load("model.stl").geometry
boundary = PointBoundary(mesh, spacing)   # blue-noise samples at spacing(x)
```

This throws darts on the continuous triangle surface (see [`sample_surface`](@ref)), producing evenly spaced wall points that also remove tessellation artifacts like near-coincident face centers at sphere poles.

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

Before committing to a spacing, probe the geometry:

```julia
g = suggest_spacing("model.stl")   # extent, volume, h_ceiling / h_baseline / h_fine
spacing = ConstantSpacing(g.h_baseline)
```

### 3D Algorithms

```julia
spacing = ConstantSpacing(1mm)

# SlakKosec (default for 3D) — sphere-based candidate generation
cloud = discretize(boundary, spacing; alg=SlakKosec())

# VanDerSandeFornberg — grid projection with sphere packing
cloud = discretize(boundary, spacing; alg=VanDerSandeFornberg(), max_points=100_000)

# Octree — spacing-driven adaptive fill; the default :bridson placement is a
# global graded Poisson-disk front, and max_points is auto-estimated when unset
bl_spacing = BoundaryLayerSpacing(points(boundary); at_wall=0.6m, bulk=4.0m, layer_thickness=8.0m)
cloud = discretize(boundary, bl_spacing; alg=Octree("model.stl"))
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
```

After an initial discretization, you can use variable spacing for a second pass:

```julia
# Variable spacing — denser near boundary, coarser in interior
# Requires an existing PointCloud (uses boundary distances internally)
spacing = LogLike(cloud, 0.5mm, 1.2)  # base_size, growth_rate
cloud = discretize(boundary, spacing; alg=SlakKosec())
```

See the [Discretization](discretization.md) page for detailed descriptions of each algorithm and spacing type.
For a complete runnable example, see
[examples/octree_boundary_layer.jl](https://github.com/JuliaMeshless/WhatsThePoint.jl/blob/main/examples/octree_boundary_layer.jl).

## Node Repulsion (Optional)

The Octree algorithm's default Bridson placement delivers blue-noise quality by construction, so repulsion is optional polish there; for the other algorithms it refines approximate uniformity to minimize interpolation error in the meshless solver.

```julia
# Volume-only — boundary points stay fixed, escaped volume points are removed
cloud = repel(cloud, spacing; β=0.2, max_iters=1000)

# Boundary-aware (3D) — all points move, escaped points projected back to surface
octree = TriangleOctree("model.stl"; classify_leaves=true)
cloud = repel(cloud, spacing, octree; β=0.2, max_iters=1000)

# Collect convergence history via keyword
conv = Float64[]
cloud = repel(cloud, spacing, octree; β=0.2, max_iters=1000, convergence=conv)
```

The returned cloud has `NoTopology` since points have moved.

!!! tip "Tuning repulsion"
    The default parameters (`β=0.2`, `k=21`, `max_iters=1000`) work well for most problems. By default the relaxation stops once the spacing quality stalls (`stall_after=50`); set a `cv_target` to stop at a chosen quality instead. See the [Node Repulsion](repel.md) page for detailed parameter guidance.

### Verifying Distribution Quality

Use `metrics` and `spacing_fidelity_metrics` to inspect the point distribution before and after repulsion:

```julia
metrics(cloud)                            # neighbor distances, separation, fill, mesh ratio
spacing_fidelity_metrics(cloud, spacing)  # d_NN/h statistics: mean, CV, percentiles, coordination
```

`metrics` reports neighbor-distance statistics plus the global separation (smallest nearest-neighbor distance), fill (largest), and their ratio; `spacing_fidelity_metrics` measures how closely the actual local spacing matches the prescribed `h(x)` — the numbers that matter for meshless stencil conditioning.

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

Write a ParaView-ready `.vtu` (open it and set *Representation* to *Point Gaussian*). Every point carries `point_type` (boundary vs volume), `surface_id` (colour by named surface), and normals; after solving, attach your result arrays as `fields`:

```julia
export_vtk("cloud", cloud)                                  # geometry
export_vtk("sol", cloud; fields = ("T" => temp, "U" => u))  # with solution data
```

`save("output", cloud; format = :vtk)` remains as a thin wrapper, and `format = :jld2` (the default) serializes the cloud itself.
