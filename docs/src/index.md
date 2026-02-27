```@meta
CurrentModule = WhatsThePoint
```

# WhatsThePoint.jl

*Generate, optimize, and connect point clouds for meshless PDE methods.*

Meshless methods need well-distributed point clouds with neighbor connectivity, but getting from a CAD surface to solver-ready points is tedious. WhatsThePoint.jl handles the complete pipeline — from surface mesh import through volume discretization, point distribution optimization, and connectivity computation — in a few lines of Julia.

## Pipeline at a Glance

1. **Import Surface** — Load STL/OBJ via GeoIO.jl into a `PointBoundary`
2. **Split & Label** — Identify walls, inlets, outlets by normal angle
3. **Generate Volume** — Fill the interior with well-spaced points
4. **Optimize Distribution** — Node repulsion for solver accuracy
5. **Build Connectivity** — k-nearest neighbor or radius-based stencils
6. **Ready for Solver** — Export to VTK or pass directly to your meshless code

## Quick Example

Load a surface mesh and inspect the boundary structure:

```@example quickstart
using WhatsThePoint
boundary = PointBoundary(joinpath(@__DIR__, "assets/bunny.stl"))
```

```julia
using GLMakie
visualize(boundary; markersize=0.15)
```

![bunny boundary](assets/bunny-boundary.png)

Generate volume points with `discretize`:

```julia
spacing = ConstantSpacing(1m)
cloud = discretize(boundary, spacing; alg=VanDerSandeFornberg(), max_points=100_000)
```

```julia
visualize(cloud; markersize=0.15)
```

![bunny discretized](assets/bunny-discretized.png)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaMeshless/WhatsThePoint.jl")
```

## Key Features

**Pipeline**
- Import surface meshes (STL, OBJ, any GeoIO.jl format)
- Multiple discretization algorithms: `SlakKosec`, `VanDerSandeFornberg` (3D), `FornbergFlyer` (2D), `OctreeRandom`
- Node repulsion for distribution optimization (Miotti 2023)
- k-nearest neighbor and radius-based topology for meshless stencils
- Export to VTK

**Performance**
- `TriangleOctree` for O(1) point-in-volume queries on large meshes
- Threaded operations throughout via [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)
- `StructArray` layout for cache-friendly surface element access

**Correctness**
- Full [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) integration — `mm`, `m`, `°` work directly
- Type-safe geometry built on [Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl) with coordinate system support
- Immutable, AD-compatible design — operations return new objects

---

[![Build Status](https://github.com/JuliaMeshless/WhatsThePoint.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMeshless/WhatsThePoint.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMeshless.github.io/WhatsThePoint.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaMeshless.github.io/WhatsThePoint.jl/dev)
[![License File](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JuliaMeshless/WhatsThePoint.jl/blob/master/LICENSE)
