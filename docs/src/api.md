```@meta
CurrentModule = WhatsThePoint
```

# API Reference

## Core Types

The fundamental data structures for representing point clouds.

```@docs
AbstractSurface
PointSurface
SurfaceElement
PointBoundary
PointVolume
PointCloud
```

## Accessors

Common accessor functions for point cloud types.

```@docs
points
point
normal
area
topology
```

## Topology

Point connectivity for meshless stencils.

```@docs
AbstractTopology
NoTopology
KNNTopology
RadiusTopology
set_topology
rebuild_topology!
neighbors
hastopology
```

## Discretization

Volume point generation algorithms and spacing types.

```@docs
discretize
SlakKosec
VanDerSandeFornberg
FornbergFlyer
OctreeRandom
ConstantSpacing
LogLike
Power
```

## Boundary Operations

Normal computation and surface manipulation.

```@docs
compute_normals
orient_normals!
update_normals!
split_surface!
combine_surfaces!
```

## Shadow Points

Virtual points offset inward from the boundary for Hermite-type boundary condition enforcement.

```@docs
ShadowPoints
generate_shadows
```

## Geometry and Queries

Point-in-volume testing, octree acceleration, and spatial utilities.

```@docs
isinside
TriangleOctree
has_consistent_normals
emptyspace
Meshes.boundingbox
Meshes.centroid
```

## Node Repulsion

Point distribution optimization.

```@docs
repel
```

## Diagnostics

```@docs
metrics
```

## I/O

```@docs
import_surface
export_cloud
save
```

## Unexported API

```@autodocs
Modules = [WhatsThePoint]
Public = false
Order = [:type, :function]
```
