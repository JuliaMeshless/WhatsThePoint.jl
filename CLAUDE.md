# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

WhatsThePoint.jl is a Julia package providing tools for manipulating point clouds for use in the solution of PDEs via meshless methods. It is part of the JuliaMeshless organization.

**Important:** This package is under heavy development and does not yet have robust testing for all methods.

## Development Commands

```julia
# Activate package environment
] activate .
] instantiate

# Run full test suite
julia --project -e 'using Pkg; Pkg.test()'

# Run individual test file (from test/ directory)
using SafeTestsets
@safetestset "PointCloud" begin
    include("cloud.jl")
end

# Build documentation locally
julia --project=docs docs/make.jl
```

## High-Level Architecture

WhatsThePoint.jl uses a hierarchical type system for representing point clouds:

1. **PointSurface** - Points with normals and areas representing a surface
2. **PointBoundary** - Collection of named surfaces forming a boundary
3. **PointVolume** - Interior volume points
4. **PointCloud** - Combines boundary and volume into a complete point cloud (mutable)

All types inherit from `Domain{M,C}` where `M<:Manifold` and `C<:CRS` (coordinate reference system) from the Meshes.jl ecosystem. This parameterization enables type stability and proper dispatch.

### Key Design Patterns

- **Composition over inheritance:** Types build on each other (surfaces → boundary → cloud)
- **Immutability:** PointSurface is immutable; PointCloud is mutable for iterative construction
- **StructArrays:** Surface elements stored as StructArray for cache-friendly memory layout
- **Heavy parallelization:** Uses OhMyThreads (tmap, tmapreduce) throughout
- **Full unit support:** Unitful.jl integrated across all operations

## Core Components

### Data Structures (`src/`)
- `surface.jl` - PointSurface and SurfaceElement types
- `boundary.jl` - PointBoundary managing named surfaces
- `volume.jl` - PointVolume for interior points
- `cloud.jl` - PointCloud combining boundary and volume

### Geometry Operations (`src/`)
- `normals.jl` - Normal computation using PCA and orientation via MST+DFS (Hoppe 1992)
- `isinside.jl` - Point-in-polygon/volume tests (2D: winding number, 3D: Green's function)
- `shadow.jl` - Shadow point generation
- `surface_operations.jl` - Split, combine, add surfaces to boundaries

### Discretization (`src/discretization/`)
Three algorithms available with **important 2D vs 3D considerations:**

- **SlakKosec** (3D only, default) - `algorithms/slak_kosec.jl`
- **VanDerSandeFornberg** (3D only) - `algorithms/vandersande_fornberg.jl`
- **FornbergFlyer** (2D only) - `algorithms/fornberg_flyer.jl`

Spacing types in `spacings.jl`: ConstantSpacing, LogLike, Power

### Optimization (`src/`)
- `repel.jl` - Node repulsion algorithm (Miotti 2023) for improving point distribution quality

## Important Technical Details

### 2D vs 3D Algorithm Differences
The discretization algorithms are dimension-specific:
- 2D geometries: Must use `FornbergFlyer()`
- 3D geometries: Use `SlakKosec()` (default) or `VanDerSandeFornberg()`

### Normal Orientation Strategy
Normal computation and orientation uses a two-step process (Hoppe 1992):
1. Compute normals via PCA on local neighborhoods
2. Orient consistently using minimum spanning tree with DFS traversal

This approach handles arbitrary surface topologies without requiring manifold assumptions.

### Point-in-Volume Testing
Different algorithms for different dimensions:
- **2D:** Winding number algorithm
- **3D:** Green's function approach

### Surface Import Behavior
When importing meshes (e.g., STL files), the package uses **face centers** as boundary points, not vertices. This is important for understanding point distributions after import.

## Common Workflows

### Creating a Point Cloud from STL

```julia
using WhatsThePoint

# Import boundary from STL
boundary = PointBoundary("path/to/file.stl")

# Define spacing
spacing = ConstantSpacing(1m)

# Discretize volume (choose algorithm based on dimensionality)
cloud = discretize(boundary, spacing;
                   alg=VanDerSandeFornberg(),
                   max_points=100_000)
```

### Surface Splitting by Normal Angle

```julia
# Split surfaces at 75 degree angle threshold
split_surface!(boundary, 75°)

# This modifies the boundary in-place, creating new named surfaces
# based on normal discontinuities
```

### Node Repulsion Optimization

```julia
# Optimize point distribution
repel!(cloud, spacing; β=0.2, max_iters=1000)

# β controls repulsion strength
# This improves point distribution quality iteratively
```

### Visualization

```julia
using GLMakie

# Visualize point cloud
visualize(cloud; markersize=0.15)

# Visualize normals
visualize_normals(boundary)
```

## Key Functions Reference

- `discretize` / `discretize!` - Generate volume points from boundary
- `split_surface!` - Split boundary surfaces by normal angle threshold
- `combine_surfaces!` - Merge multiple surfaces into one
- `compute_normals` / `orient_normals!` - Normal vector handling
- `repel!` - Optimize point distribution via node repulsion
- `isinside` - Test if point is inside domain
- `import_surface` - Load from STL/mesh files (via GeoIO.jl)
- `export_cloud` - Save to VTK format
- `visualize` - Makie-based visualization

## Testing Structure

Tests use `SafeTestsets.jl` for isolated environments:

```
test/
├── runtests.jl          # Main orchestrator
├── points.jl            # Point utilities tests
├── normals.jl           # Normal computation tests
├── surface.jl           # PointSurface tests
├── boundary.jl          # PointBoundary tests
├── cloud.jl             # PointCloud tests
├── isinside.jl          # Point-in-volume tests
└── data/
    ├── bifurcation.stl  # Test data (24,780 points)
    └── box.stl          # Test data
```

## CI/CD

`.github/workflows/CI.yml` runs on Ubuntu, macOS, Windows with Julia 1.10 and 1.11. Tests trigger on pushes to main, tags, and PRs (excluding docs/license changes).

Documentation builds automatically via `documenter.yml` and deploys to GitHub Pages.
