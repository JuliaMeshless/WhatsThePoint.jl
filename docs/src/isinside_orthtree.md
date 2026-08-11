```@meta
CurrentModule = WhatsThePoint
```

# Point-in-Volume & Orthtree

WhatsThePoint provides four approaches for testing whether points lie inside a closed boundary, suited to different problem sizes and dimensions. The two accelerated approaches use orthtree spatial indexes — a quadtree in 2D ([`SegmentQuadtree`](@ref)), an octree in 3D ([`TriangleOctree`](@ref)).

## Overview

| Method | Dimensions | Complexity | When to use |
|--------|-----------|------------|-------------|
| Winding number | 2D | O(M) per query | Small boundaries, few queries |
| Quadtree-accelerated | 2D | O(1) most queries | Many queries or large boundaries |
| Green's function | 3D | O(M) per query | Small/medium meshes |
| Octree-accelerated | 3D | O(1) most queries | Large meshes (>10k triangles) |

*M = number of boundary segments (2D) or triangles (3D).*

## 2D: Winding Number

For 2D domains, [`isinside`](@ref) uses the **winding number algorithm**. It counts the number of times the boundary winds around a query point — nonzero winding number means the point is inside.

## 2D: Quadtree-Accelerated

The [`SegmentQuadtree`](@ref) is the 2D counterpart of the `TriangleOctree`: it indexes the boundary loops of a `PointBoundary` and pre-classifies regions as interior or exterior, so most queries are O(1) lookups. Multiple loops describe multiply-connected domains — outer boundary plus holes, resolved automatically by nesting parity.

```julia
quadtree = SegmentQuadtree(bnd)   # each surface of the PointBoundary is one closed loop
isinside(point, quadtree)
```

This is the same geometry index the 2D `Orthtree` discretization algorithm builds internally (`Orthtree(bnd; spacing)`).

## 3D: Green's Function

For 3D domains without an octree, `isinside` uses a **Green's function approach**. The solid angle subtended by each boundary triangle at the query point is summed — a total of 4π means the point is inside. This is O(M) per query where M is the number of boundary triangles.

## 3D: Octree-Accelerated

For large 3D meshes, the Green's function approach becomes expensive. The [`TriangleOctree`](@ref) accelerates queries to O(1) for most points by spatially decomposing the mesh and pre-classifying regions as interior or exterior.

### Building a TriangleOctree

```julia
using WhatsThePoint
using Unitful: mm

# Load the mesh once (unit required — files carry no unit metadata)
mesh = import_mesh("model.stl", mm)
octree = TriangleOctree(mesh; min_ratio=1e-6)
```

Parameters:
- `tolerance_relative` — Vertex coincidence tolerance as fraction of domain diagonal.
- `min_ratio` — Minimum box size as fraction of domain diagonal.
- `classify_leaves` — Whether to classify empty leaves as interior/exterior (default: `true`).
- `verify_orientation` — Check mesh orientation before building (default: `true`). Uses `has_consistent_normals` internally — if normals are inconsistent, the octree classification will be unreliable. When `classify_leaves=true`, a globally inside-out mesh (consistent winding, but normals pointing into the solid) is also rejected with an `ArgumentError` via a signed-volume check, since it would classify the complement of the domain as interior.

### Construction Process

1. Create a root box enclosing the mesh bounding box (with a small buffer)
2. Subdivide adaptively when a box contains more than one unique in-box vertex
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
octree = TriangleOctree(import_mesh("model.stl", mm); min_ratio=1e-6)

# Single point query
result = isinside(point, octree)

# Batch query
results = isinside(points, octree)
```

**Performance:** For interior and exterior leaves, the result is a direct array lookup — O(1). For boundary leaves, the exact signed distance is computed: an octree-pruned nearest-triangle search, with the sign taken from the angle-weighted pseudonormal of the closest feature (face, edge, or vertex — Bærentzen & Aanæs 2005). The sign is provably correct for watertight, consistently outward-oriented meshes. This is dramatically faster than the default O(M) Green's function approach for large meshes.

Both `SVector{3}` and Meshes.jl `Point` types are accepted.

## Integration with Discretization

### SlakKosec with Octree

Pass a `TriangleOctree` to `SlakKosec` to accelerate the `isinside` checks during volume point generation:

```julia
octree = TriangleOctree(import_mesh("model.stl", mm); min_ratio=1e-6)
spacing = ConstantSpacing(1mm)
cloud = discretize(boundary, spacing; alg=SlakKosec(octree))
```

### Using Orthtree

`Orthtree` uses the geometry index directly to generate volume points (a `TriangleOctree` in 3D, a `SegmentQuadtree` in 2D). See the [Discretization](discretization.md) page for details.

```julia
mesh = import_mesh("model.stl", mm)
cloud = discretize(boundary, spacing; alg=Orthtree(mesh), max_points=200_000)
```

## Choosing an Approach

- **2D problems:** The winding number is used automatically — no configuration needed. For many queries or large boundaries, build a `SegmentQuadtree`.
- **Small 3D meshes (<10k triangles):** The default Green's function works fine.
- **Large 3D meshes (>10k triangles):** Build a `TriangleOctree` and pass it to `isinside` or `SlakKosec` for orders-of-magnitude speedup.
- **When you need volume points directly:** Use `Orthtree` to skip the separate isinside step entirely.

## Query Functions

```julia
num_leaves(octree)                 # Number of leaf nodes
num_triangles(octree)              # Number of triangles in the mesh
num_segments(quadtree)             # Number of boundary segments in a SegmentQuadtree
has_consistent_normals(mesh)       # Check if mesh normals are consistently oriented
```

!!! note "`has_consistent_normals`"
    This function checks whether a mesh has consistently oriented normals — a prerequisite for reliable interior/exterior classification. It is called automatically during `TriangleOctree` construction when `verify_orientation=true` (the default).
