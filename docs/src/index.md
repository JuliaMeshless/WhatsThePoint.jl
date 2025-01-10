```@meta
CurrentModule = WhatsThePoint
```

# WhatsThePoint.jl

[![Build Status](https://github.com/JuliaMeshless/WhatsThePoint.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMeshless/WhatsThePoint.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMeshless.github.io/WhatsThePoint.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaMeshless.github.io/WhatsThePoint.jl/dev)
[![License File](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JuliaMeshless/WhatsThePoint.jl/blob/master/LICENSE)

Documentation for [WhatsThePoint](https://github.com/JuliaMeshless/WhatsThePoint.jl).

This package intends to provide functions for all things regarding point clouds.

## Installation

Simply install the latest stable release using Julia's package manager:

```julia
] add https://github.com/JuliaMeshless/WhatsThePoint.jl
```

## Quick Start

Although there are a number of features in this package, the initial use case is for
generating point clouds for use in numerical solution of PDEs via a meshless method.

You can load a surface mesh and extract the boundary points using the `PointBoundary`
constructor.

```@example quickstart
using WhatsThePoint
boundary = PointBoundary(joinpath(@__DIR__, "assets/bunny.stl"))
```

and we can visualize it with

```julia
using GLMakie
visualize(boundary; markersize=0.15)
```

![bunny boundary](assets/bunny-boundary.png)

Then we can generate a point cloud using the `PointCloud` constructor.

```julia
spacing = ConstantSpacing(1m)
cloud = discretize(boundary, spacing; alg=VanDerSandeFornberg(), max_points=100_000)
```

and we can visualize again with `visualize(cloud; markersize=0.15)`

![bunny discretized](assets/bunny-discretized.png)
