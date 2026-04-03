```@meta
CurrentModule = WhatsThePoint
```

# Concepts

This page explains the core design decisions and type hierarchy behind WhatsThePoint.jl.

## Type Hierarchy

WhatsThePoint builds point clouds through composition:

```
PointSurface     — Points with normals and areas (one contiguous surface)
    ↓
PointBoundary    — Named collection of surfaces forming a closed boundary
    ↓
PointCloud       — Boundary + volume points + optional topology
```

All geometric types inherit from `Domain{M,C}` where `M<:Manifold` describes the geometric space (currently Euclidean: `𝔼{2}` or `𝔼{3}`) and `C<:CRS` is the coordinate reference system (Cartesian, Cylindrical, Polar, etc.). This parameterization comes from [Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl) and enables type stability and proper dispatch.

```julia
# A 3D point cloud in Cartesian coordinates on a Euclidean manifold:
PointCloud{𝔼{3}, Cartesian3D}
```

## Surface Elements

Each boundary point is stored as a [`SurfaceElement`](@ref) containing:
- **point** — The geometric position (face center of the imported mesh, not a vertex)
- **normal** — Outward-pointing unit normal vector
- **area** — Associated surface area

A [`PointSurface`](@ref) stores its elements as a `StructArray{SurfaceElement}`, giving column-oriented memory layout. This means iterating over all normals reads contiguous memory rather than striding through interleaved fields — a significant cache performance advantage when processing thousands of surface elements.

## Immutability and Functional API

Most types in WhatsThePoint are **immutable**. Operations return new objects rather than mutating in place:

```julia
# set_topology returns a new cloud — the original is unchanged
cloud2 = set_topology(cloud, KNNTopology, 21)

# repel returns a new cloud (with NoTopology, since points moved)
cloud3 = repel(cloud2, spacing)
```

This design ensures compatibility with automatic differentiation frameworks and prevents stale state — if points move, the old topology object simply isn't used.

**Exceptions:** `split_surface!` and `combine_surfaces!` mutate a `PointBoundary`'s internal surface dictionary. These are in-place operations by convention (indicated by the `!` suffix) because they only reorganize existing surfaces without changing any point data.

## Topology

Topology represents point connectivity — the neighbor stencils used by meshless solvers.

**Type hierarchy:**
- [`AbstractTopology`](@ref) — Abstract base with storage type parameter
- [`NoTopology`](@ref) — Singleton default (no connectivity computed)
- [`KNNTopology`](@ref) — k-nearest neighbor connectivity
- [`RadiusTopology`](@ref) — All neighbors within a fixed radius

**Local vs global indices:** Topology on a `PointSurface` or `PointVolume` uses local indices (1 through `length(surf)`). Topology on a `PointCloud` uses global indices (1 through `length(cloud)`), where boundary points come first, followed by volume points.

```julia
# Surface topology — local indices
surf = set_topology(surf, KNNTopology, 10)
neighbors(surf, 3)  # indices into surf, e.g. [1, 4, 7, ...]

# Cloud topology — global indices
cloud = set_topology(cloud, KNNTopology, 21)
neighbors(cloud, 3) # indices into cloud, e.g. [1, 2, 5, 102, ...]
```

Topology is computed **eagerly** when `set_topology` is called and is never automatically invalidated. Operations that move points (like [`repel`](@ref)) return objects with [`NoTopology`](@ref) — call `set_topology` again after repulsion.

## Units

WhatsThePoint integrates with [Unitful.jl](https://github.com/PainterQubits/Unitful.jl). Physical units work directly throughout the API:

```julia
spacing = ConstantSpacing(1mm)
cloud = set_topology(cloud, RadiusTopology, 2mm)
split_surface!(boundary, 75°)
```

Units propagate through all operations — point coordinates, normals, areas, and spacing values all carry their units consistently.

## Parallelism

WhatsThePoint uses [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl) for threaded execution. The following operations are parallelized:

- Normal computation and orientation (`compute_normals`, `orient_normals!`)
- Point-in-volume testing (`isinside`)
- Node repulsion (`repel`)
- Octree queries and construction

To use multiple threads, start Julia with:

```bash
julia -t auto    # use all available cores
julia -t 8       # use 8 threads
```

No code changes are needed — parallel execution is automatic when threads are available.

## Euclidean Assumption

WhatsThePoint currently supports **Euclidean manifolds only** (`𝔼{2}` and `𝔼{3}`). Functions like `compute_normals`, `orient_normals!`, `discretize`, `repel`, and `isinside` have explicit Euclidean type constraints.

Any coordinate system (Cartesian, Cylindrical, Polar, etc.) is supported *on* a Euclidean manifold — the constraint is about geometric structure (flat space), not coordinate representation.

The type system is designed so that non-Euclidean geometries (spherical, hyperbolic) can be added in the future through new method dispatches with appropriate geodesic operations.
