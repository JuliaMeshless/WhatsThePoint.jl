# Octree

WhatsThePoint provides an octree data structure for accelerating spatial queries on triangle meshes. The primary type is `TriangleOctree`, which subdivides 3D space around a surface mesh and classifies leaf nodes as interior, boundary, or exterior.

## TriangleOctree

### Construction

Build a `TriangleOctree` from a mesh file or a `SimpleMesh`:

```julia
using WhatsThePoint

# From a file path
octree = TriangleOctree("model.stl"; min_ratio=1e-6)

# From a SimpleMesh
using GeoIO
mesh = GeoIO.load("model.stl") |> boundary
octree = TriangleOctree(mesh; min_ratio=1e-6)
```

Parameters:
- `tolerance_relative` — Vertex coincidence tolerance as fraction of domain diagonal.
- `min_ratio` — Minimum box size as fraction of domain diagonal.
- `classify_leaves` — Whether to classify empty leaves as interior/exterior (default: `true`).
- `verify_orientation` — Check mesh normal consistency before building (default: `true`).

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

## Octree-Accelerated isinside

The primary use of `TriangleOctree` is fast point-in-volume testing:

```julia
octree = TriangleOctree("model.stl"; min_ratio=1e-6)

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
octree = TriangleOctree("model.stl"; min_ratio=1e-6)
spacing = ConstantSpacing(1mm)
cloud = discretize(boundary, spacing; alg=SlakKosec(octree))
```

### Using OctreeRandom

`OctreeRandom` uses the octree directly to generate volume points. See the [Discretization](@ref) page for details.

```julia
cloud = discretize(boundary, OctreeRandom("model.stl"; min_ratio=1e-6))
```

## Query Functions

```julia
num_leaves(octree)                 # Number of leaf nodes
num_triangles(octree)              # Number of triangles in the mesh
has_consistent_normals(mesh)       # Check if mesh normals are consistently oriented
```
