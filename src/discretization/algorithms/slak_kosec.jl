"""
    SlakKosec <: AbstractNodeGenerationAlgorithm

Slak-Kosec algorithm for volume point generation with optional octree acceleration.

The algorithm generates candidate points on spheres around existing points and accepts
them if they are inside the domain and sufficiently far from existing points.

# Fields
- `n::Int` - Number of candidate points per sphere (default: 10)
- `octree::Union{Nothing,TriangleOctree}` - Optional octree for fast isinside queries

# Constructors
```julia
SlakKosec()                          # Default: n=10, no octree
SlakKosec(20)                        # Custom n, no octree
SlakKosec(octree::TriangleOctree)    # Use octree acceleration with n=10
SlakKosec(20, octree)                # Custom n with octree acceleration
```

# Performance
- **Without octree**: Uses Green's function for isinside (~50ms per query)
- **With octree**: Uses spatial indexing (~0.05ms per query, 1000× faster!)

# Usage Examples

## Standard Usage (Green's function)
```julia
using WhatsThePoint

# Load boundary
boundary = PointBoundary("model.stl")
cloud = PointCloud(boundary)

# Discretize without octree (slow for large domains)
spacing = ConstantSpacing(1.0u"m")
result = discretize(cloud, spacing; alg=SlakKosec(), max_points=10_000)
```

## Octree-Accelerated Usage (Recommended for large domains)
```julia
using WhatsThePoint

# Load boundary points
boundary = PointBoundary("model.stl")
cloud = PointCloud(boundary)

# Build octree from STL file (Option 1: simplest)
octree = TriangleOctree("model.stl"; h_min=0.01, classify_leaves=true)

# Or from SimpleMesh (Option 2)
# mesh = GeoIO.load("model.stl").geometry
# octree = TriangleOctree(mesh; h_min=0.01, classify_leaves=true)

# Use octree-accelerated discretization (100-1000× faster!)
spacing = ConstantSpacing(1.0u"m")
alg = SlakKosec(octree)  # Pass octree to algorithm
result = discretize(cloud, spacing; alg=alg, max_points=100_000)
```

# References
Šlak J, Kosec G. "On generation of node distributions for meshless PDE discretizations" (2019)
"""
struct SlakKosec{O <: Union{Nothing, TriangleOctree}} <: AbstractNodeGenerationAlgorithm
    n::Int
    octree::O
end
SlakKosec(n::Int = 10) = SlakKosec{Nothing}(n, nothing)
SlakKosec(octree::TriangleOctree{M, C, T}) where {M, C, T} =
    SlakKosec{TriangleOctree{M, C, T}}(10, octree)
SlakKosec(n::Int, octree::TriangleOctree{M, C, T}) where {M, C, T} =
    SlakKosec{TriangleOctree{M, C, T}}(n, octree)

function _discretize_volume(
        cloud::PointCloud{𝔼{3}, C},
        spacing::AbstractSpacing,
        alg::SlakKosec;
        max_points = 1_000,
    ) where {C}
    seeds = copy(points(boundary(cloud)))
    search_method = KNearestSearch(seeds, 1)
    new_points = Point{𝔼{3}, C}[]

    i = 0
    while !isempty(seeds) && i < max_points
        p = popfirst!(seeds)
        r = spacing(p)
        candidates = _get_candidates(p, r; n = alg.n)
        for c in candidates
            # Use octree-based isinside if available, otherwise use standard isinside
            inside = if !isnothing(alg.octree)
                _isinside_octree(c, alg.octree)
            else
                isinside(c, cloud)
            end

            if inside
                _, dist = searchdists(c, search_method)
                if first(dist) > r
                    if i >= max_points
                        @warn "discretization stopping early, reached max points ($max_points)"
                        return PointVolume(new_points)
                    end
                    push!(seeds, c)
                    push!(new_points, c)
                    search_method = KNearestSearch(seeds, 1)
                    i += 1
                end
            end
        end
    end

    return PointVolume(new_points)
end

"""
    _isinside_octree(point::Point{𝔼{3},C}, octree::TriangleOctree) -> Bool

Convert Point to SVector and use octree-based isinside query.

This provides significant speedup (100-1000×) over standard Green's function approach
by leveraging spatial indexing.
"""
function _isinside_octree(
        point::Point{𝔼{3}, C},
        octree::TriangleOctree{M, CRS, T},
    ) where {C, M, CRS, T}
    # Convert Point to SVector (stripping units if present)
    coords = to(point)
    svec = SVector{3, T}(ustrip.(coords)...)

    # Use octree-based isinside query
    return isinside(svec, octree)
end

function _get_candidates(p::Point{𝔼{3}, C}, r; n = 10) where {C}
    T = CoordRefSystems.mactype(C)

    u = rand(T, n)
    v = rand(T, n)

    one_T = one(T)
    ϕ = @. acos(2u - one_T) - oftype(one_T, π / 2)
    λ = 2π * v
    coords = to(p)
    unit_points = @. Point(r * cos(λ) * cos(ϕ), r * sin(λ) * cos(ϕ), r * sin(ϕ))
    return Ref(coords) .+ unit_points
end
