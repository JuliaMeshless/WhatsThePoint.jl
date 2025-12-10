# Design Document: Topology Support for PointCloud

## Motivation

WhatsThePoint.jl serves as a geometry preprocessor for meshless PDE simulations. In meshless methods, differential operators (stencils) are built from local point neighborhoods. Currently, PointCloud stores only geometry (points, normals, areas) without connectivity information.

In Meshes.jl parlance:
- **Domain**: Collection of geometries (what PointCloud is today)
- **Mesh**: Domain + Topology (connectivity/adjacency)

To be the meshless equivalent of a "mesh", PointCloud should include topology - the connectivity structure defining which points form each stencil's neighborhood.

### Current Pain Points

1. **Redundant computation**: Algorithms like `repel!`, `compute_normals`, and `split_surface!` each rebuild neighborhood structures from scratch
2. **No persistence**: Stencil neighborhoods must be recomputed every time the cloud is used
3. **Downstream friction**: PDE solvers using WhatsThePoint must independently compute and manage connectivity

### Benefits of Integrated Topology

1. **Single source of truth**: Connectivity computed once, stored with geometry
2. **Meshless-ready output**: Downstream solvers receive complete stencil information
3. **Performance**: Avoid redundant neighbor searches across operations
4. **Consistency**: Ensure all operations use the same neighborhood definition

---

## Design

### Type Hierarchy

```julia
# Abstract base - extensible for future connectivity types
abstract type AbstractTopology{S} end

# Distance-based: all points within radius r
struct RadiusTopology{S,R} <: AbstractTopology{S}
    neighbors::S                     # storage (adjl, sparse matrix, etc.)
    radius::R                        # search radius (can be function of position)
end

# k-nearest neighbors
struct KNNTopology{S} <: AbstractTopology{S}
    neighbors::S                     # storage (adjl, sparse matrix, etc.)
    k::Int                          # number of neighbors
end

# No topology (current behavior)
struct NoTopology <: AbstractTopology{Nothing} end
```

The storage type parameter `S` allows flexibility in how neighbors are stored:
- `Vector{Vector{Int}}` - adjacency list (default, variable neighborhood sizes)
- `SparseMatrixCSC{Bool,Int}` - sparse adjacency matrix (efficient for graph operations)
- `Matrix{Int}` - dense matrix (fixed k neighbors, cache-friendly iteration)
- Custom types as needed

### Updated PointCloud

```julia
mutable struct PointCloud{M<:Manifold,C<:CRS,T<:AbstractTopology} <: Domain{M,C}
    boundary::PointBoundary{M,C}
    volume::PointVolume{M,C}
    topology::T
end
```

### Storage Format Options

The type parameter `S` enables multiple storage formats. Default is adjacency list:

| Format | Type | Best For |
|--------|------|----------|
| Adjacency list | `Vector{Vector{Int}}` | Variable neighborhood sizes, radius-based |
| Sparse matrix | `SparseMatrixCSC{Bool,Int}` | Graph algorithms, symmetric queries |
| Dense matrix | `Matrix{Int}` | Fixed k neighbors, SIMD-friendly iteration |

Default to adjacency list (`Vector{Vector{Int}}`) for initial implementation - most flexible for meshless stencils with variable neighborhood sizes.

### Constructor Patterns

```julia
# Default: no topology (backwards compatible)
PointCloud(boundary) = PointCloud(boundary, PointVolume{M,C}(), NoTopology())

# Build with topology
PointCloud(boundary, volume, KNNTopology(k))
PointCloud(boundary, volume, RadiusTopology(r))

# Add topology to existing cloud
set_topology!(cloud, KNNTopology(k))
set_topology!(cloud, RadiusTopology(r))

# Access
topology(cloud::PointCloud) = cloud.topology
neighbors(cloud::PointCloud) = neighbors(topology(cloud))
neighbors(cloud::PointCloud, i::Int) = neighbors(topology(cloud))[i]
```

---

## Pros and Cons

### Pros

1. **Conceptual alignment**: PointCloud becomes the meshless equivalent of Mesh
2. **API clarity**: `topology(cloud)` mirrors Meshes.jl's `topology(mesh)`
3. **Extensibility**: Abstract type allows future topology variants (adaptive, hierarchical)
4. **Backwards compatible**: `NoTopology` default preserves current behavior
5. **Performance**: Pre-computed neighbors avoid repeated spatial queries
6. **Downstream value**: PDE solvers receive ready-to-use stencil structure

### Cons

1. **Memory overhead**: Storing adjacency lists (~8 bytes/neighbor per point)
2. **Staleness risk**: Topology can become invalid if points move (e.g., after `repel!`)
3. **Complexity**: More state to manage and synchronize
4. **Scope creep**: May encourage feature bloat (weighted edges, multi-level, etc.)

### Mitigation Strategies

- **Staleness**: Invalidate topology on mutation (`repel!`, point removal); provide `rebuild_topology!`
- **Memory**: Only PointCloud gets topology (not PointBoundary/PointSurface)
- **Complexity**: Keep topology types minimal; resist adding weights/attributes initially

---

## Design Decisions

1. **Eager computation**: Topology built immediately when `set_topology!` called
2. **Invalidate only on mutation**: `repel!` marks topology stale; user calls `rebuild_topology!` manually
3. **PointCloud only**: No topology support for PointBoundary or PointSurface

---

## Implementation Plan

### Phase 1: Core Types (src/topology.jl)

1. Define `AbstractTopology{S}` abstract type (S = storage type parameter)
2. Define `NoTopology <: AbstractTopology{Nothing}` singleton (default, no connectivity)
3. Define `KNNTopology{S}` struct with:
   - `neighbors::S` - storage (parameterized, default `Vector{Vector{Int}}`)
   - `k::Int` - number of neighbors
   - `valid::Bool` - staleness flag
4. Define `RadiusTopology{S,R}` struct with:
   - `neighbors::S` - storage (parameterized, default `Vector{Vector{Int}}`)
   - `radius::R` - search radius (can be scalar or function)
   - `valid::Bool` - staleness flag
5. Implement `neighbors(t::AbstractTopology)` and `neighbors(t, i::Int)`
6. Implement `isvalid(t::AbstractTopology)` and `invalidate!(t::AbstractTopology)`
7. Implement internal `_build_neighbors()` using `KNearestSearch`/`BallSearch`
8. Type alias for convenience: `const AdjListTopology{T} = T{Vector{Vector{Int}}} where T<:AbstractTopology`

### Phase 2: PointCloud Integration (src/cloud.jl)

1. Add third type parameter `T<:AbstractTopology` to `PointCloud{M,C,T}`
2. Add `topology::T` field
3. Update inner constructor: `PointCloud(boundary, volume, topology=NoTopology())`
4. Backwards-compatible outer constructors (default to `NoTopology()`)
5. Add accessors:
   - `topology(cloud)` - returns topology object
   - `neighbors(cloud)` - returns all neighbor lists (errors if invalid)
   - `neighbors(cloud, i)` - returns neighbors of point i
   - `hastopology(cloud)` - true if not NoTopology
6. Add mutators:
   - `set_topology!(cloud, KNNTopology(k))` - builds eagerly
   - `set_topology!(cloud, RadiusTopology(r))` - builds eagerly
   - `rebuild_topology!(cloud)` - rebuilds using same parameters
   - `invalidate_topology!(cloud)` - marks as stale

### Phase 3: Invalidation in repel! (src/repel.jl)

1. At end of `repel!`, call `invalidate_topology!(cloud)` if `hastopology(cloud)`
2. No auto-rebuild (user decision)

### Phase 4: Tests (test/topology.jl)

1. `KNNTopology` construction with known point set, verify neighbor counts
2. `RadiusTopology` construction, verify neighbors within radius
3. `neighbors()` accessor returns correct indices
4. `invalidate!()` marks topology invalid
5. Accessing invalid topology throws error
6. `rebuild_topology!()` restores validity
7. `NoTopology` default - `hastopology()` returns false
8. Backwards compatibility: old PointCloud code still works

### Phase 5: Documentation

1. Update `CLAUDE.md` topology section
2. Docstrings for all new types and functions
3. Example usage in docstrings

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/topology.jl` | **NEW** - topology types and builders |
| `src/cloud.jl` | Add type parameter, field, accessors |
| `src/WhatsThePoint.jl` | Include topology.jl, exports |
| `src/repel.jl` | Invalidate topology after mutation |
| `test/topology.jl` | **NEW** - topology tests |
| `test/runtests.jl` | Include topology tests |
| `CLAUDE.md` | Document topology feature |

---

## Example Usage

```julia
using WhatsThePoint
using Unitful: m, mm

# Create point cloud as usual
boundary = PointBoundary("model.stl")
cloud = discretize(boundary, ConstantSpacing(1mm); alg=VanDerSandeFornberg())

# Add k-nearest neighbor topology (eager build)
set_topology!(cloud, KNNTopology(21))

# Access neighbors
all_neighbors = neighbors(cloud)        # Vector{Vector{Int}}
point_5_neighbors = neighbors(cloud, 5) # Vector{Int}

# Check topology state
hastopology(cloud)  # true
isvalid(topology(cloud))  # true

# After repel!, topology becomes invalid
repel!(cloud, ConstantSpacing(1mm))
isvalid(topology(cloud))  # false

# Rebuild when ready
rebuild_topology!(cloud)
isvalid(topology(cloud))  # true

# Alternative: radius-based topology
set_topology!(cloud, RadiusTopology(2mm))
```
