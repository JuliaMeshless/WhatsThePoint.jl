# Octree

WhatsThePoint provides an octree data structure for accelerating spatial queries on triangle meshes. The primary type is `TriangleOctree`, which subdivides 3D space around a surface mesh and classifies leaf nodes as interior, boundary, or exterior.

## TriangleOctree

### Construction

Build a `TriangleOctree` from a mesh file or a `SimpleMesh`:

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
- `verify_orientation` — Check mesh normal consistency before building (default: `true`).

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

## Octree-Accelerated isinside

The primary use of `TriangleOctree` is fast point-in-volume testing:

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

`OctreeRandom` uses the octree directly to generate volume points. See the [Discretization](@ref) page for details.

```julia
cloud = discretize(boundary, OctreeRandom("model.stl"; h_min=0.5))
```

## Query Functions

```julia
num_leaves(octree)                 # Number of leaf nodes
num_triangles(octree)              # Number of triangles in the mesh
has_consistent_normals(mesh)       # Check if mesh normals are consistently oriented
```
