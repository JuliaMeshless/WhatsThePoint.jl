# Design: Multi-Level Topology in WhatsThePoint.jl (Immutable Style)

## Requirements

1. **PointSurface** and **PointVolume** should each have their own topology (like SimpleMesh)
2. **PointCloud** should have its own unified topology (for cross-component neighbors)
3. Component topologies are **preserved** when combining (not discarded)
4. **All types immutable** (Meshes.jl style) for AD compatibility

**Use cases:**
- Surface-local stencils (e.g., surface Laplacian, surface PDE)
- Portable surfaces (pass PointSurface to another function with connectivity intact)
- Consistency with Meshes.jl (PointSurface ≈ SimpleMesh pattern)
- Future AutoDiff support (Zygote.jl, Enzyme.jl)

---

## Meshes.jl Pattern

Meshes.jl uses **immutable structs** with operations that **return new objects**:

```julia
# Meshes.jl pattern
struct SimpleMesh{...} <: Mesh{...}  # immutable
  vertices
  topology
end

# Operations return new meshes
newmesh = topoconvert(HalfEdgeTopology, mesh)
smoothed = smooth(mesh)
```

---

## Proposed Architecture

```
PointSurface (immutable) ───┬─→ PointBoundary (immutable) ───┬─→ PointCloud (immutable)
└─ topology                 │   (aggregates)                 │   └─ topology (cloud-level)
                            │                                │
PointVolume (immutable) ────┴────────────────────────────────┘
└─ topology
```

### Type Changes

```julia
# PointSurface: Keep immutable, add topology field
struct PointSurface{M,C,S,T<:AbstractTopology} <: AbstractSurface{M,C}
    geoms::StructVector{SurfaceElement}
    shadow::S
    topology::T
end

# PointVolume: Make immutable, add topology field
struct PointVolume{M,C,T<:AbstractTopology} <: Domain{M,C}
    points::Domain{M,C}
    topology::T
end

# PointCloud: Make immutable
struct PointCloud{M,C,T<:AbstractTopology} <: Domain{M,C}
    boundary::PointBoundary{M,C}
    volume::PointVolume{M,C}
    topology::T
end
```

---

## API Design (Functional Style)

**Key change:** Operations return new objects instead of mutating.

```julia
# Surface-level topology (returns new surface)
surface_with_topo = set_topology(surface, KNNTopology, k)  # No !
neighbors(surface, i)  # Local indices: 1..length(surface)

# Volume-level topology (returns new volume)
volume_with_topo = set_topology(volume, KNNTopology, k)
neighbors(volume, i)  # Local indices: 1..length(volume)

# Cloud-level topology (returns new cloud)
cloud_with_topo = set_topology(cloud, KNNTopology, k)
neighbors(cloud, i)  # Global indices: 1..length(cloud)

# Iterative operations (returns new objects)
repelled_cloud = repel(cloud, spacing)  # No !
```

### Breaking Changes

| Old API | New API |
|---------|---------|
| `set_topology!(cloud, KNNTopology, k)` | `cloud = set_topology(cloud, KNNTopology, k)` |
| `repel!(cloud, spacing)` | `cloud = repel(cloud, spacing)` |
| `invalidate_topology!(cloud)` | Not needed - create new object |
| `rebuild_topology!(cloud)` | `cloud = rebuild_topology(cloud)` |

---

## No More Invalidation

**With immutable design, invalidation is unnecessary:**

- Objects are values, not mutable state
- When points change, you get a new object (with `NoTopology` or fresh topology)
- No "stale" state can exist

```julia
# Old (mutable)
repel!(cloud, spacing)
# Now cloud has stale topology - must invalidate!
invalidate_topology!(cloud)
rebuild_topology!(cloud)

# New (immutable)
cloud = repel(cloud, spacing)
# repel() returns new cloud with NoTopology or rebuilt topology
# No invalidation needed - old cloud is unchanged
```

---

## Implementation Considerations

### 1. PointVolume Changes

Current PointVolume is mutable for iterative point updates. Two options:

**Option A: Pure immutable (recommended for AD)**
```julia
# Loop reassignment pattern
vol = initial_volume
for i in 1:max_iters
    vol = repel_step(vol, ...)
end
```

**Option B: Wrapper with mutable interior (if performance critical)**
```julia
struct PointVolume{M,C,T}
    points::Ref{Domain{M,C}}  # Mutable ref, but PointVolume itself immutable
    topology::T
end
```

### 2. Mutable Contents vs Mutable Struct

**Key insight:** An immutable struct can have mutable contents:

```julia
struct PointSurface{...}  # immutable struct
    geoms::StructVector{...}  # mutable container!
end

# Legal - mutating container contents:
surface.geoms.normal[i] = new_normal

# Illegal - reassigning field:
surface.geoms = new_structvector
```

**Operations that mutate container CONTENTS can keep `!` suffix:**
- `orient_normals!` - mutates `surface.geoms.normal` vector elements ✓
- `split_surface!` - rearranges elements within existing containers ✓

**Operations that REPLACE fields need functional style:**
- `set_topology` - replaces `topology` field → needs new object
- `discretize` - creates new volume → needs new cloud
- `repel!` → `repel` - line 55 does `cloud.volume = ...` (replaces field!)
- `filter!(volume)` → `filter` - replaces `points` field

### 3. Type Parameter for Topology

Use type parameter `T<:AbstractTopology` to allow specialization:

```julia
struct PointSurface{M,C,S,T<:AbstractTopology}
    # ...
    topology::T
end

# NoTopology version - lightweight
PointSurface{..., NoTopology}

# With KNN topology
PointSurface{..., KNNTopology{Vector{Vector{Int}}}}
```

---

## Final Design Decisions

### 1. All Immutable (Meshes.jl Style)

**Decision:** Make all types immutable (`struct`, not `mutable struct`).

**Rationale:**
- Best for AD frameworks (Zygote.jl, ChainRules.jl)
- Matches Meshes.jl philosophy
- Thread-safe by default
- Easier to reason about
- No invalidation complexity

### 2. Default Topology

All components default to `NoTopology()` - zero cost when unused.

### 3. PointBoundary Topology

**Decision:** No boundary-wide topology in first iteration.

---

## Implementation Plan

### Phase 1: Add Topology to PointSurface (Already Immutable)
1. Add `topology::T` type parameter and field (default `NoTopology`)
2. Add `set_topology`, `neighbors`, `hastopology`, `topology` accessor methods
3. Update all constructors to accept/default topology
4. Update tests

**Files:** `src/surface.jl`, `test/surface.jl`

### Phase 2: Add Topology to PointVolume + Make Immutable
1. Change `mutable struct PointVolume` to `struct PointVolume`
2. Add `topology::T` type parameter and field (default `NoTopology`)
3. Add `set_topology`, `neighbors`, `hastopology`, `topology` methods
4. `filter!` → `filter` (returns new PointVolume) - this was the only field-replacing operation
5. Update constructors
6. Update tests

**Files:** `src/volume.jl`, `test/volume.jl`

### Phase 3: Make PointCloud Immutable + Update Topology API
1. Change `mutable struct PointCloud` to `struct PointCloud`
2. Add topology type parameter `T<:AbstractTopology`
3. Refactor `set_topology!` → `set_topology` (returns new cloud)
4. Remove invalidation functions (not needed with immutable design)
5. Update `discretize` to return new cloud (was `discretize!` but creates new volume)
6. Update tests

**Files:** `src/cloud.jl`, `src/topology.jl`, `src/discretization/*.jl`, `test/cloud.jl`, `test/topology.jl`

### Phase 4: Refactor Field-Replacing Operations
After code review, these functions **replace fields** and need functional versions:
- `repel!` → `repel` - does `cloud.volume = ...` on line 55
- Operations that still work (mutate contents only):
  - `orient_normals!` ✓
  - `split_surface!` ✓

**Files:** `src/repel.jl`

### Phase 5: Documentation
1. Update CLAUDE.md with new patterns
2. Add examples showing topology at each level

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/surface.jl` | Add `topology::T` type param and field |
| `src/volume.jl` | Change to immutable, add `topology::T`, `filter!` → `filter` |
| `src/cloud.jl` | Change to immutable, add topology type param |
| `src/topology.jl` | `set_topology` (no !), remove invalidation functions |
| `src/repel.jl` | `repel!` → `repel` (returns new cloud) |
| `src/discretization/*.jl` | Verify `discretize` returns new cloud properly |
| `test/surface.jl` | Add surface topology tests |
| `test/volume.jl` | Add volume topology tests, update for immutable |
| `test/cloud.jl` | Update for immutable cloud |
| `test/topology.jl` | Update for functional API |

---

## Migration Notes

**Breaking changes:**
- `set_topology!` → `set_topology` (returns new object)
- `PointVolume` now immutable
- `PointCloud` now immutable
- `filter!(volume)` → `filter(volume)` (returns new volume)

```julia
# Before
set_topology!(cloud, KNNTopology, 21)

# After
cloud = set_topology(cloud, KNNTopology, 21)
```

**Still works (mutable containers):**
- `orient_normals!` - mutates normals in-place ✓
- `split_surface!` - rearranges surface elements ✓

**Also needs refactoring (replaces field):**
- `repel!` → `repel` - returns new cloud (does `cloud.volume = ...`)
