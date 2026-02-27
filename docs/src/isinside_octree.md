```@meta
CurrentModule = WhatsThePoint
```

# Point-in-Volume & Octree

WhatsThePoint provides three approaches for testing whether points lie inside a closed boundary, suited to different problem sizes and dimensions.

## Overview

| Method | Dimensions | Complexity | When to use |
|--------|-----------|------------|-------------|
| Winding number | 2D | O(M) per query | All 2D problems |
| Green's function | 3D | O(M) per query | Small/medium meshes |
| Octree-accelerated | 3D | O(1) most queries | Large meshes (>10k triangles) |

*M = number of boundary segments (2D) or triangles (3D).*

## 2D: Winding Number

For 2D domains, [`isinside`](@ref) uses the **winding number algorithm**. It counts the number of times the boundary winds around a query point — nonzero winding number means the point is inside.

## 3D: Green's Function

For 3D domains without an octree, `isinside` uses a **Green's function approach**. The solid angle subtended by each boundary triangle at the query point is summed — a total of 4π means the point is inside. This is O(M) per query where M is the number of boundary triangles.

## 3D: Octree-Accelerated

For large 3D meshes, the Green's function approach becomes expensive. The [`TriangleOctree`](@ref) accelerates queries to O(1) for most points by spatially decomposing the mesh and pre-classifying regions as interior or exterior.

### Building a TriangleOctree

```julia
using WhatsThePoint

# From a file path
octree = TriangleOctree("model.stl"; h_min=0.5)

# From a SimpleMesh
using GeoIO
mesh = GeoIO.load("model.stl") |> boundary
octree = TriangleOctree(mesh; h_min=0.5)
```

Parameters:
- `h_min` (required) — Minimum box size. Controls the finest resolution of the octree.
- `max_triangles_per_box` — Maximum triangles per leaf before subdivision (default: 50).
- `classify_leaves` — Whether to classify empty leaves as interior/exterior (default: `true`).
- `verify_orientation` — Check mesh normal consistency before building (default: `true`). Uses `has_consistent_normals` internally — if normals are inconsistent, the octree classification will be unreliable.

### Construction Process

1. Create a root box enclosing the mesh bounding box (with a small buffer)
2. Distribute triangles into leaves, subdividing when a leaf has too many triangles or is larger than `h_min`
3. Balance the tree to enforce a 2:1 refinement constraint (no adjacent leaves differ by more than one level)
4. Classify empty leaves as interior or exterior using signed distance queries

### Leaf Classification

Each leaf in the octree is classified as one of:
- **Interior** — Entirely inside the mesh surface
- **Exterior** — Entirely outside the mesh surface
- **Boundary** — Contains or is near the mesh surface

This classification enables O(1) point-in-volume queries for interior and exterior leaves.

### Octree-Accelerated isinside

```julia
octree = TriangleOctree("model.stl"; h_min=0.5)

# Single point query
result = isinside(point, octree)

# Batch query
results = isinside(points, octree)
```

**Performance:** For interior and exterior leaves, the result is a direct array lookup — O(1). For boundary leaves, a local signed distance is computed using only the triangles in that leaf — O(k) where k is typically 10–50 triangles. This is dramatically faster than the default O(M) Green's function approach for large meshes.

Both `SVector{3}` and Meshes.jl `Point` types are accepted.

## Integration with Discretization

### SlakKosec with Octree

Pass a `TriangleOctree` to `SlakKosec` to accelerate the `isinside` checks during volume point generation:

```julia
octree = TriangleOctree("model.stl"; h_min=0.5)
spacing = ConstantSpacing(1mm)
cloud = discretize(boundary, spacing; alg=SlakKosec(octree))
```

### Using OctreeRandom

`OctreeRandom` uses the octree directly to generate volume points. See the [Discretization](discretization.md) page for details.

```julia
cloud = discretize(boundary, OctreeRandom("model.stl"; h_min=0.5))
```

## Choosing an Approach

- **2D problems:** The winding number is used automatically — no configuration needed.
- **Small 3D meshes (<10k triangles):** The default Green's function works fine.
- **Large 3D meshes (>10k triangles):** Build a `TriangleOctree` and pass it to `isinside` or `SlakKosec` for orders-of-magnitude speedup.
- **When you need volume points directly:** Use `OctreeRandom` to skip the separate isinside step entirely.

## Query Functions

```julia
num_leaves(octree)                 # Number of leaf nodes
num_triangles(octree)              # Number of triangles in the mesh
has_consistent_normals(mesh)       # Check if mesh normals are consistently oriented
```

!!! note "`has_consistent_normals`"
    This function checks whether a mesh has consistently oriented normals — a prerequisite for reliable interior/exterior classification. It is called automatically during `TriangleOctree` construction when `verify_orientation=true` (the default).
