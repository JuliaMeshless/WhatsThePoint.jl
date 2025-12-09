# Euclidean Manifold Constraints: Design Document

**Status**: Planned (Not Yet Implemented)
**Created**: 2025-01-18
**Purpose**: Document the rationale and implementation plan for adding explicit Euclidean manifold constraints to geometry functions

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Background: Manifold vs CRS](#background-manifold-vs-crs)
3. [Current State Analysis](#current-state-analysis)
4. [Design Decision: Manifold-Only Constraints](#design-decision-manifold-only-constraints)
5. [Implementation Plan](#implementation-plan)
6. [Examples](#examples)
7. [Testing Strategy](#testing-strategy)
8. [Future Extensions](#future-extensions)

---

## Problem Statement

WhatsThePoint.jl currently operates exclusively in Euclidean space (flat geometry), but the type system uses generic `Manifold` type parameters that theoretically allow non-Euclidean geometries. All core algorithms (normal computation, discretization, repulsion, etc.) make implicit Euclidean assumptions but lack explicit type constraints.

**Goals:**
1. Make Euclidean assumptions explicit via type constraints
2. Provide informative errors when non-Euclidean manifolds are used
3. Enable future support for non-Euclidean geometries through dispatch
4. Maintain flexibility for different coordinate systems on Euclidean space

---

## Background: Manifold vs CRS

### Type Parameter Architecture

All core types inherit from `Domain{M,C}`:
```julia
Domain{M<:Manifold, C<:CRS}
├── PointSurface{M,C,S}
├── PointBoundary{M,C}
├── PointVolume{M,C}
└── PointCloud{M,C}
```

### Manifold (M): Geometric Structure

**Definition**: Represents the geometric/topological space structure

**Purpose:**
- Defines the **metric** (how distances are measured)
- Defines **curvature** (flat vs curved space)
- Determines **valid geometric operations** (angles, cross products, etc.)

**Examples:**
- `𝔼{2}` - 2D Euclidean (flat plane)
- `𝔼{3}` - 3D Euclidean (flat 3D space)
- `🌐{R}` - Spherical manifold with radius R
- Hyperbolic, cylindrical, etc.

**Current Usage:**
- Primary dispatch mechanism for 2D vs 3D algorithms
- Extracted via `manifold(point)` or `manifold(domain)`

### CRS (C): Coordinate Reference System

**Definition**: Represents how coordinates are represented and stored

**Purpose:**
- Defines **coordinate representation** (Cartesian, polar, cylindrical, etc.)
- Controls **numeric type** via `CoordRefSystems.mactype(C)`
- Manages **unit systems** (via Unitful.jl)
- Determines **storage precision** (Float64, Float32, etc.)

**Examples:**
- `Cartesian2D{Float64}` - 2D Cartesian coordinates
- `Polar{Float64}` - Polar coordinates (r, θ)
- `Cartesian3D{Float64}` - 3D Cartesian coordinates
- `Cylindrical{Float64}` - Cylindrical coordinates (ρ, φ, z)

**Current Usage:**
- Type extraction: `T = CoordRefSystems.mactype(C)`
- Unit handling in constructors
- Rarely used for dispatch

### Key Distinction

**Analogy**:
- **Manifold** = "What kind of space?" (flat vs curved)
- **CRS** = "How do we measure positions in that space?" (coordinate system)

**Critical Insight:**
Once you're on a Euclidean manifold, the CRS is just a representation choice. The underlying geometric operations (distances, norms, dot products) are defined by the **manifold's metric**, not the CRS.

---

## Current State Analysis

### Functions With Implicit Euclidean Assumptions

#### 1. Normal Computation (`src/normals.jl`)

**Functions:**
- `compute_normals` - Uses PCA on raw coordinates
- `orient_normals!` - Uses dot products for orientation
- `_angle` functions - Use cross products (3D Euclidean only)

**Euclidean Assumptions:**
```julia
# PCA assumes flat Euclidean covariance
cov_matrix = (coords' * coords) / n

# Dot products assume Euclidean inner product
consistency = dot(normal1, normal2)

# Cross products assume 3D Euclidean space
n = (v1 × v2) / norm(v1 × v2)
```

**Current Constraints:** None (generic `Manifold`)

#### 2. Distance Calculations (`src/discretization/spacings.jl`)

**Function:**
```julia
distance(p1, p2) = evaluate(Euclidean(), p1, p2)
```

**Euclidean Assumption:** Explicitly uses `Euclidean()` metric

**Current Constraints:** None

#### 3. Neighbor Search (`src/neighbors.jl`)

**Function:**
```julia
KNearestSearch(cloud, k; metric=Euclidean())
```

**Euclidean Assumption:** Default metric is Euclidean

**Current Constraints:** None

#### 4. Discretization Algorithms (`src/discretization/algorithms/`)

**Files:**
- `slak_kosec.jl` - 3D Euclidean point generation
- `vandersande_fornberg.jl` - 3D Euclidean discretization
- `fornberg_flyer.jl` - 2D Euclidean discretization

**Euclidean Assumptions:**
```julia
# Spherical coordinate generation (3D Euclidean)
unit_points = @. Point(r * cos(λ) * cos(ϕ),
                       r * sin(λ) * cos(ϕ),
                       r * sin(ϕ))

# Assumes flat space geometry
```

**Current Constraints:** ✅ Already properly constrained
```julia
discretize!(cloud::PointCloud{𝔼{3},C}, ...) where {C<:CRS}
discretize!(cloud::PointCloud{𝔼{2},C}, ...) where {C<:CRS}
```

#### 5. Repulsion Algorithm (`src/repel.jl`)

**Function:** `repel!` - Node repulsion for optimization

**Euclidean Assumptions:**
```julia
# Euclidean distance
rij = norm(xi - xj)

# Force based on Euclidean distance
F(r / s) * (xi - xj) / r
```

**Current Constraints:** None

#### 6. Point-in-Volume Testing (`src/isinside.jl`)

**Green's Function Method:**
```julia
dist = testpoint - point
result = area * dist ⋅ normal / norm(dist)^3
```

**Euclidean Assumptions:** Euclidean norm and dot product

**Current Constraints:** None for generic method; `𝔼{2}` for winding number (correct)

### Summary of Current State

**Already Constrained (✅):**
- Discretization algorithms (`𝔼{2}` or `𝔼{3}`)
- 2D winding number (`𝔼{2}`)
- Visualization functions (`𝔼{2}` or `𝔼{3}`)

**Missing Constraints (❌):**
- Normal computation and orientation
- Distance functions
- Neighbor search
- Repulsion algorithm
- Green's function point-in-volume test

---

## Design Decision: Manifold-Only Constraints

### The Question

Should we constrain:
1. **Manifold only**: `{𝔼{N},C} where {N,C<:CRS}`
2. **Both M and C**: `{𝔼{N},Cartesian{N}D} where {N}`

### Analysis

#### What Is The Euclidean Requirement About?

The requirement is fundamentally about **geometric structure** (flat vs curved space), not **coordinate representation**.

**Examples of valid use cases on Euclidean space:**
- Cylindrical coordinates for axisymmetric problems
- Polar coordinates for radial discretization patterns
- Different numeric precision (Float32 vs Float64)
- Different unit systems

All of these work because:
- The manifold is Euclidean (flat space with Euclidean metric)
- Meshes.jl handles coordinate conversions automatically
- `norm()`, `distance()`, dot products use the manifold's metric, not CRS

#### Code Evidence

```julia
# This works regardless of CRS
rij = norm(xi - xj)  # Uses Euclidean metric from manifold

# Point construction accepts any valid CRS on 𝔼{3}
Point(Cylindrical(ρ, φ, z))  # Valid on 𝔼{3}
Point(Cartesian(x, y, z))    # Also valid on 𝔼{3}

# Meshes.jl abstracts away CRS differences
distance = evaluate(Euclidean(), p1, p2)  # Works with any CRS
```

#### Decision: **Manifold-Only Constraints**

**Rationale:**

1. **Semantically Correct**
   - Euclidean requirement is about manifold geometry, not coordinate choice
   - CRS is an implementation detail for a given manifold

2. **Maximum Flexibility**
   - Supports polar, cylindrical, or any CRS on Euclidean space
   - Doesn't block legitimate use cases

3. **Consistent Pattern**
   - Discretization algorithms already use `{𝔼{N},C} where {C<:CRS}`
   - Maintains established convention

4. **Simpler Maintenance**
   - No need to enumerate valid CRS types
   - CRS validity is Meshes.jl's responsibility

5. **Trust the Abstraction**
   - Meshes.jl guarantees CRS validity on given manifolds
   - Don't duplicate their type constraints

6. **Future-Proof**
   - Easy to add non-Euclidean manifolds without CRS complications
   - Each manifold can support its natural coordinate systems

### Constraint Pattern

**For dimension-agnostic functions:**
```julia
function compute_normals(surf::PointSurface{𝔼{N},C}) where {N,C<:CRS}
    # Works on 𝔼{2} or 𝔼{3}, any CRS
end
```

**For dimension-specific functions:**
```julia
function discretize!(cloud::PointCloud{𝔼{3},C}, ...) where {C<:CRS}
    # Only 3D Euclidean, any CRS
end
```

**Error fallback:**
```julia
function compute_normals(surf::PointSurface{M,C}) where {M<:Manifold,C<:CRS}
    error("""
    compute_normals requires Euclidean manifold.
    Got: $M
    Supported: 𝔼{2}, 𝔼{3}

    This function uses PCA which assumes flat Euclidean geometry.
    """)
end
```

---

## Implementation Plan

### Phase 1: Add Euclidean Constraints

#### File: `src/normals.jl`

**Functions to constrain:**

1. **`compute_normals`**
   ```julia
   # Current (no constraint)
   function compute_normals(surf::PointSurface{M,C}, k::Int) where {M<:Manifold,C<:CRS}

   # New (Euclidean only)
   function compute_normals(surf::PointSurface{𝔼{N},C}, k::Int) where {N,C<:CRS}

   # Add error fallback
   function compute_normals(surf::PointSurface{M,C}, k::Int) where {M<:Manifold,C<:CRS}
       error("""
       compute_normals requires Euclidean manifold (𝔼{2} or 𝔼{3}).
       Got: $M

       This function uses Principal Component Analysis which assumes
       flat Euclidean geometry. Non-Euclidean manifolds require different
       normal computation methods (e.g., geodesic-based for spherical).
       """)
   end
   ```

2. **`orient_normals!`**
   ```julia
   # Current (no constraint)
   function orient_normals!(surf::PointSurface{M,C}, k::Int) where {M<:Manifold,C<:CRS}

   # New (Euclidean only)
   function orient_normals!(surf::PointSurface{𝔼{N},C}, k::Int) where {N,C<:CRS}

   # Add error fallback
   function orient_normals!(surf::PointSurface{M,C}, k::Int) where {M<:Manifold,C<:CRS}
       error("""
       orient_normals! requires Euclidean manifold (𝔼{2} or 𝔼{3}).
       Got: $M

       This function uses Euclidean dot products for orientation consistency.
       Non-Euclidean manifolds require geodesic-based orientation methods.
       """)
   end
   ```

3. **`_angle` functions**
   ```julia
   # Already dimension-specific, add error fallback
   function _angle(n1::Vec3, n2::Vec3)  # Keep as is

   # Add fallback for wrong input type
   function _angle(n1, n2)
       error("_angle requires 3D Euclidean vectors (Vec3)")
   end
   ```

**Documentation updates:**
- Add `# Note: Requires Euclidean manifold` to docstrings
- Explain why (PCA, dot products, cross products)

#### File: `src/discretization/spacings.jl`

**Functions to constrain:**

1. **`distance`**
   ```julia
   # Current (no constraint)
   distance(p1, p2) = evaluate(Euclidean(), p1, p2)

   # New (Euclidean only)
   distance(p1::Point{𝔼{N}}, p2::Point{𝔼{N}}) where {N} =
       evaluate(Euclidean(), p1, p2)

   # Add error fallback
   distance(p1::Point{M}, p2::Point{M}) where {M<:Manifold} =
       error("""
       distance function requires Euclidean manifold.
       Got: $M

       For non-Euclidean manifolds, use appropriate geodesic distance
       functions (e.g., Haversine for spherical, geodesic for hyperbolic).
       """)
   ```

2. **Spacing evaluation** (if they compute distances directly)
   - Review `(spacing::ConstantSpacing)(...)` and similar
   - Add constraints if needed

#### File: `src/repel.jl`

**Function to constrain:**

1. **`repel!`**
   ```julia
   # Current (no constraint)
   function repel!(cloud::PointCloud{M,C}, ...) where {M<:Manifold,C<:CRS}

   # New (Euclidean only)
   function repel!(cloud::PointCloud{𝔼{N},C}, ...) where {N,C<:CRS}

   # Add error fallback
   function repel!(cloud::PointCloud{M,C}, ...) where {M<:Manifold,C<:CRS}
       error("""
       repel! requires Euclidean manifold (𝔼{2} or 𝔼{3}).
       Got: $M

       This node repulsion algorithm uses Euclidean distances and forces.
       Non-Euclidean manifolds require geodesic-based repulsion methods.
       """)
   end
   ```

#### File: `src/isinside.jl`

**Functions to constrain:**

1. **Green's function methods**
   ```julia
   # Current (generic)
   function isinside(testpoint::Point{M}, cloud::PointCloud{M}) where {M<:Manifold}

   # New (Euclidean only for Green's function)
   function isinside(testpoint::Point{𝔼{N}}, cloud::PointCloud{𝔼{N}}) where {N}

   # Add error fallback
   function isinside(testpoint::Point{M}, cloud::PointCloud{M}) where {M<:Manifold}
       error("""
       isinside (Green's function) requires Euclidean manifold.
       Got: $M

       This point-in-volume test uses Green's function with Euclidean norms.
       For non-Euclidean manifolds, implement appropriate geodesic methods.
       """)
   end
   ```

**Note:** 2D winding number already has `𝔼{2}` constraint ✅

#### File: `src/neighbors.jl`

**Review needed:**
- Check if functions need explicit constraints
- Most likely OK if they accept `metric` parameter
- Add error methods if hard-coded Euclidean assumptions exist

### Phase 2: Documentation

#### Update docstrings

For each constrained function, add:
```julia
"""
    function_name(args...)

Description of function.

# Geometric Requirements
Requires Euclidean manifold (`𝔼{2}` or `𝔼{3}`). This function assumes:
- Flat space geometry
- Euclidean distance metric
- Standard dot/cross products

For non-Euclidean geometries, see [future extensions or alternatives].

# Examples
```julia
# 2D Euclidean
surf = PointSurface{𝔼{2}, Cartesian2D}(...)
normals = compute_normals(surf, 20)

# 3D Euclidean
surf = PointSurface{𝔼{3}, Cartesian3D}(...)
normals = compute_normals(surf, 20)
```
"""
```

#### Update CLAUDE.md

Add section:
```markdown
## Geometric Assumptions

WhatsThePoint.jl currently supports **Euclidean manifolds only** (`𝔼{2}` and `𝔼{3}`).

### Functions Requiring Euclidean Space

The following functions have explicit Euclidean constraints:
- `compute_normals` - Uses PCA (flat geometry assumption)
- `orient_normals!` - Uses Euclidean dot products
- `discretize` and algorithms - Euclidean point generation
- `repel!` - Euclidean distance-based repulsion
- `isinside` (Green's function) - Euclidean norms
- `distance` - Explicitly uses Euclidean metric

### Coordinate Systems

Any coordinate system (CRS) is supported **on Euclidean manifolds**:
- Cartesian (x, y, z)
- Cylindrical (ρ, φ, z)
- Polar (r, θ)
- Any CRS supported by Meshes.jl/CoordRefSystems.jl

### Future Support

The type system is designed to support non-Euclidean geometries through
multiple dispatch. To add spherical/hyperbolic manifolds, implement
manifold-specific methods with appropriate geodesic operations.
```

#### Update README.md

Add note in appropriate section about geometric assumptions.

### Phase 3: Testing

#### Test 1: Verify Euclidean Dispatch Works

```julia
@testset "Euclidean constraints" begin
    # 2D Euclidean
    points2d = [Point(rand(2)...) for _ in 1:100]
    surf2d = PointSurface(points2d)
    @test surf2d isa PointSurface{𝔼{2}}
    normals2d = compute_normals(surf2d, 10)  # Should work
    @test length(normals2d) == 100

    # 3D Euclidean
    points3d = [Point(rand(3)...) for _ in 1:100]
    surf3d = PointSurface(points3d)
    @test surf3d isa PointSurface{𝔼{3}}
    normals3d = compute_normals(surf3d, 10)  # Should work
    @test length(normals3d) == 100
end
```

#### Test 2: Verify Error Messages

```julia
@testset "Non-Euclidean error messages" begin
    # This test requires a non-Euclidean manifold type from Meshes.jl
    # If not available, skip or mock

    # Example (if spherical manifold available):
    # points_sphere = [Point(Spherical(...)) for _ in 1:100]
    # surf_sphere = PointSurface(points_sphere)
    # @test_throws ErrorException("requires Euclidean manifold") compute_normals(surf_sphere, 10)

    # For now, document that errors will trigger on non-Euclidean input
end
```

#### Test 3: Different CRS on Euclidean Space

```julia
@testset "Multiple CRS on Euclidean manifold" begin
    # Test that different CRS work on Euclidean space
    # (if CoordRefSystems.jl supports these)

    # Cartesian (default)
    cartesian_points = [Point(rand(3)...) for _ in 1:100]
    surf_cart = PointSurface(cartesian_points)
    normals_cart = compute_normals(surf_cart, 10)
    @test length(normals_cart) == 100

    # TODO: Test cylindrical, polar if feasible
end
```

#### Test 4: Existing Tests Still Pass

```julia
# Run full test suite
julia --project -e 'using Pkg; Pkg.test()'

# All existing tests should pass with new constraints
```

---

## Examples

### Current Behavior After Implementation

```julia
using WhatsThePoint

# ✅ Works: 2D Euclidean with Cartesian coordinates
points_2d = [Point(rand(2)...) for _ in 1:100]
surf_2d = PointSurface(points_2d)
typeof(surf_2d)  # PointSurface{𝔼{2}, Cartesian2D{...}}
normals = compute_normals(surf_2d, 20)  # ✅ Works

# ✅ Works: 3D Euclidean with Cartesian coordinates
points_3d = [Point(rand(3)...) for _ in 1:100]
surf_3d = PointSurface(points_3d)
typeof(surf_3d)  # PointSurface{𝔼{3}, Cartesian3D{...}}
normals = compute_normals(surf_3d, 20)  # ✅ Works

# ✅ Works: Different CRS on Euclidean space (theoretical)
# If using cylindrical coordinates on 𝔼{3}
points_cyl = [Point(Cylindrical(...)) for _ in 1:100]
surf_cyl = PointSurface(points_cyl)
typeof(surf_cyl)  # PointSurface{𝔼{3}, Cylindrical{...}}
normals = compute_normals(surf_cyl, 20)  # ✅ Works - still Euclidean geometry

# ❌ Error: Non-Euclidean manifold
# (requires Meshes.jl to support this manifold type)
points_sphere = [Point(Spherical(6371km, θ, φ)) for ...]  # Hypothetical
surf_sphere = PointSurface(points_sphere)
typeof(surf_sphere)  # PointSurface{🌐{6371000.0}, ...}
normals = compute_normals(surf_sphere, 20)
# ERROR: compute_normals requires Euclidean manifold (𝔼{2} or 𝔼{3}).
#        Got: 🌐{6.371e6}
#        This function uses PCA which assumes flat Euclidean geometry.
```

---

## Future Extensions

### Adding Spherical Manifold Support

When non-Euclidean support is needed, add new methods:

#### Example: Spherical Normal Computation

```julia
"""
Compute normals for points on a sphere.
For spherical geometry, normals point radially from sphere center.
"""
function compute_normals(surf::PointSurface{🌐{R},C}, k::Int) where {R,C<:CRS}
    # Extract coordinates
    coords = coordinates.(surf.elements)

    # For sphere, normal at point P is simply P/|P| (radial direction)
    normals = map(coords) do p
        # Convert to Cartesian if needed
        cart = convert(Cartesian3D, p)
        center = Point(0.0, 0.0, 0.0)  # Sphere center

        # Normal points radially outward
        r = cart - center
        return r / norm(r)
    end

    return normals
end

# Euclidean version remains unchanged
function compute_normals(surf::PointSurface{𝔼{N},C}, k::Int) where {N,C<:CRS}
    # PCA-based computation for flat space
    # ... existing code ...
end
```

#### Example: Geodesic Distance on Sphere

```julia
"""
Compute geodesic distance between two points on a sphere.
Uses great circle distance (Haversine formula).
"""
function distance(p1::Point{🌐{R}}, p2::Point{🌐{R}}) where {R}
    return evaluate(Haversine(R), p1, p2)
end

# Euclidean version remains unchanged
function distance(p1::Point{𝔼{N}}, p2::Point{𝔼{N}}) where {N}
    return evaluate(Euclidean(), p1, p2)
end
```

### Dispatch Mechanism

Julia automatically selects the correct method:

```julia
# Input determines which method is called
euclidean_surf = PointSurface{𝔼{3}, Cartesian3D}(...)
compute_normals(euclidean_surf, 20)  # → Calls Euclidean version (PCA)

spherical_surf = PointSurface{🌐{6371e3}, Spherical}(...)
compute_normals(spherical_surf, 20)  # → Calls spherical version (radial)

# No changes to existing code required!
```

### Benefits of This Design

1. **Incremental Support**: Add manifolds one at a time
2. **No Breaking Changes**: Existing Euclidean code untouched
3. **Type Safety**: Compile-time dispatch guarantees correct algorithm
4. **Clear Separation**: Each manifold has its own implementations
5. **Extensible**: Third-party packages can add new manifold methods

---

## Implementation Checklist

### Code Changes

- [ ] `src/normals.jl`
  - [ ] Add `{𝔼{N},C}` constraint to `compute_normals`
  - [ ] Add `{𝔼{N},C}` constraint to `orient_normals!`
  - [ ] Add `𝔼{3}` constraint to `_angle` functions (already 3D-specific)
  - [ ] Add error fallback methods with informative messages
  - [ ] Update docstrings

- [ ] `src/discretization/spacings.jl`
  - [ ] Add `{𝔼{N}}` constraint to `distance` function
  - [ ] Add error fallback method
  - [ ] Review spacing evaluation functions
  - [ ] Update docstrings

- [ ] `src/repel.jl`
  - [ ] Add `{𝔼{N},C}` constraint to `repel!`
  - [ ] Add error fallback method
  - [ ] Update docstrings

- [ ] `src/isinside.jl`
  - [ ] Add `{𝔼{N}}` constraint to Green's function methods
  - [ ] Add error fallback method
  - [ ] Verify 2D winding number already has `𝔼{2}` constraint
  - [ ] Update docstrings

- [ ] `src/neighbors.jl`
  - [ ] Review for hard-coded Euclidean assumptions
  - [ ] Add constraints if needed
  - [ ] Add error methods if needed

### Documentation Changes

- [ ] Update function docstrings with geometric requirements
- [ ] Add "Geometric Requirements" section to each constrained function
- [ ] Update `CLAUDE.md` with geometric assumptions section
- [ ] Update `README.md` with note about Euclidean-only support
- [ ] Add examples showing different CRS on Euclidean space

### Testing

- [ ] Add test for 2D Euclidean dispatch
- [ ] Add test for 3D Euclidean dispatch
- [ ] Add test for different CRS on Euclidean space (if feasible)
- [ ] Add test for error messages (if non-Euclidean types available)
- [ ] Verify all existing tests still pass
- [ ] Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`

### Validation

- [ ] Verify no breaking changes to existing code
- [ ] Confirm error messages are clear and helpful
- [ ] Check that different CRS work on Euclidean space
- [ ] Ensure type stability (use `@code_warntype` if needed)

---

## References

### Relevant Issues/PRs

- (Add links to related GitHub issues or PRs)

### Related Documentation

- Meshes.jl Manifolds: https://github.com/JuliaGeometry/Meshes.jl
- CoordRefSystems.jl: https://github.com/JuliaEarth/CoordRefSystems.jl
- Multiple Dispatch in Julia: https://docs.julialang.org/en/v1/manual/methods/

### Academic References

- Hoppe et al. (1992) - Surface Reconstruction from Unorganized Points (PCA normals)
- Miotti (2023) - Node repulsion algorithm

---

## Notes

### Why Not Trait-Based Dispatch?

We decided against using traits (e.g., `IsEuclidean`) because:
1. Julia's type system already provides dispatch via `M` parameter
2. Adding traits adds complexity without clear benefit
3. Simple type constraints are more idiomatic Julia
4. Future maintainers can add traits later if needed

### Why Manifold-Only (Not M+C)?

Constraining both M and C would block legitimate use cases:
- Cylindrical coordinates on Euclidean space
- Polar coordinates for radial problems
- Different numeric precisions
- Different unit systems

The Euclidean requirement is about geometric structure (manifold), not coordinate representation (CRS).

### Migration Path

This change should be **non-breaking** for existing code because:
1. All current usage is already on Euclidean manifolds
2. We're making implicit constraints explicit
3. Error messages guide users if they try non-Euclidean inputs

---

**Document Status**: Ready for implementation
**Next Step**: Begin Phase 1 implementation starting with `src/normals.jl`
