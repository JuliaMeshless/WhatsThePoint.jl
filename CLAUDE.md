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

### Iteration Behavior

Iterating over `PointCloud` and `PointBoundary` yields individual points (as `SurfaceElement` or volume points), not surfaces. To iterate over surfaces, use `surfaces()`:

```julia
for pt in cloud               # iterates over individual points
for surf in surfaces(cloud)   # iterates over PointSurface objects

for pt in boundary            # iterates over individual points
for surf in surfaces(boundary) # iterates over PointSurface objects
```

## Core Components

### Data Structures (`src/`)
- `surface.jl` - PointSurface and SurfaceElement types
- `boundary.jl` - PointBoundary managing named surfaces
- `volume.jl` - PointVolume for interior points
- `cloud.jl` - PointCloud combining boundary, volume, and topology
- `topology.jl` - Point connectivity (KNNTopology, RadiusTopology)

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

### Geometric Assumptions (Euclidean Manifolds Only)

WhatsThePoint.jl currently supports **Euclidean manifolds only** (`𝔼{2}` and `𝔼{3}`). The following functions have explicit Euclidean type constraints:

- `compute_normals` / `orient_normals!` - Uses PCA and Euclidean dot products
- `discretize` algorithms - Euclidean point generation
- `repel!` - Euclidean distance-based repulsion
- `isinside` (Green's function) - Euclidean norms
- `distance` - Explicitly uses Euclidean metric
- `generate_shadows` - Euclidean vector arithmetic

**Coordinate Systems:** Any CRS is supported on Euclidean manifolds (Cartesian, Cylindrical, Polar, etc.). The Euclidean requirement is about geometric structure (flat space), not coordinate representation.

**Future Extensions:** The type system is designed to support non-Euclidean geometries through multiple dispatch. To add spherical/hyperbolic manifolds, implement manifold-specific methods with appropriate geodesic operations.

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

### Topology (Point Connectivity)

PointCloud, PointSurface, and PointVolume all support optional topology storing point neighborhoods for meshless stencils. All types use **immutable structs** with functional API (operations return new objects).

```julia
# Add k-nearest neighbor topology (returns new cloud)
cloud = set_topology(cloud, KNNTopology, 21)

# Or radius-based topology
cloud = set_topology(cloud, RadiusTopology, 2mm)

# Surface-level topology (local indices)
surf = set_topology(surf, KNNTopology, 10)
neighbors(surf, i)  # Local indices: 1..length(surf)

# Volume-level topology (local indices)
vol = set_topology(vol, KNNTopology, 15)
neighbors(vol, i)   # Local indices: 1..length(vol)

# Cloud-level topology (global indices)
neighbors(cloud, i) # Global indices: 1..length(cloud)

# Check state
hastopology(cloud)  # true if topology exists
```

**Key behaviors:**
- `NoTopology` is the default (backwards compatible)
- Topology is built eagerly when `set_topology` is called
- All operations return new objects (immutable design for AD compatibility)
- `repel` returns new cloud with `NoTopology` (points moved)
- No invalidation needed - immutable objects can't become stale

**Type hierarchy:**
- `AbstractTopology{S}` - abstract base with storage type parameter
- `NoTopology` - singleton, no connectivity
- `KNNTopology{S}` - k-nearest neighbors
- `RadiusTopology{S,R}` - radius-based neighbors

**Multi-level topology:**
- PointSurface has `topology::T` field for surface-local connectivity
- PointVolume has `topology::T` field for volume-local connectivity
- PointCloud has `topology::T` field for global connectivity
- Component topologies use local indices; cloud topology uses global indices

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
# Optimize point distribution (returns tuple)
cloud, convergence = repel(cloud, spacing; β=0.2, max_iters=1000)

# β controls repulsion strength
# Returns (new_cloud, convergence_vector) tuple
# New cloud has NoTopology since points moved
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

- `discretize` - Generate volume points from boundary (returns new cloud)
- `split_surface!` - Split boundary surfaces by normal angle threshold
- `combine_surfaces!` - Merge multiple surfaces into one
- `compute_normals` / `orient_normals!` - Normal vector handling
- `repel` - Optimize point distribution via node repulsion (returns new cloud, convergence)
- `isinside` - Test if point is inside domain
- `import_surface` - Load from STL/mesh files (via GeoIO.jl)
- `export_cloud` - Save to VTK format
- `visualize` - Makie-based visualization
- `set_topology` - Build point connectivity and return new object
- `rebuild_topology!` - Rebuild topology in place with same parameters
- `neighbors` - Access point neighborhoods from topology

## Testing Structure

Tests use `TestItemRunner.jl` with `@testitem` macros:

```
test/
├── runtests.jl          # Main orchestrator (@run_package_tests)
├── testsetup.jl         # Common imports and test data
├── points.jl            # Point utilities tests
├── normals.jl           # Normal computation tests
├── surface.jl           # PointSurface tests
├── boundary.jl          # PointBoundary tests
├── cloud.jl             # PointCloud tests
├── topology.jl          # Topology tests (KNNTopology, RadiusTopology)
├── isinside.jl          # Point-in-volume tests
└── data/
    ├── bifurcation.stl  # Test data (24,780 points)
    └── box.stl          # Test data
```

## CI/CD

`.github/workflows/CI.yml` runs on Ubuntu, macOS, Windows with Julia 1.10 and 1.11. Tests trigger on pushes to main, tags, and PRs (excluding docs/license changes).

Documentation builds automatically via `documenter.yml` and deploys to GitHub Pages.
