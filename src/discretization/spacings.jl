"""
    AbstractSpacing

Interface for spacing functions that control node density during discretization.

Subtypes must be callable with a single `Point` or `Vec` argument and return a `Unitful.Length`
representing the desired node spacing at that location.

    (s::MySpacing)(p::Union{Point, Vec}) -> Unitful.Length

See [`ConstantSpacing`](@ref), [`LogLike`](@ref), and [`BoundaryLayerSpacing`](@ref) for
concrete implementations.
"""
abstract type AbstractSpacing end
abstract type VariableSpacing <: AbstractSpacing end

# O(log n) nearest-neighbor query via KDTree. Distances come back in the
# tree's (= boundary data's) machine type regardless of the query point's.
function _min_distance(p, boundary, tree::KDTree)
    q = ustrip.(to(p))
    idxs, dists = knn(tree, q, 1)
    return dists[1] * unit(eltype(to(first(boundary))))
end

function _build_boundary_tree(boundary_points)
    coords = [ustrip.(to(p)) for p in boundary_points]
    return KDTree(coords)
end

"""
    ConstantSpacing{L<:Unitful.Length} <: AbstractSpacing

Constant node spacing.
"""
struct ConstantSpacing{L <: Unitful.Length} <: AbstractSpacing
    Δx::L
end
(s::ConstantSpacing)() = s.Δx
(s::ConstantSpacing)(_) = s.Δx

"""
    LogLike <: VariableSpacing

Node spacing based on a log-like function of the distance to nearest boundary ``x/(x+a)``
    where ``x`` is the distance to the nearest boundary and ``a`` is a parameter to control
    the growth rate as ``a = 1 - (g - 1)`` where ``g`` is the conventional growth rate
    parameter.
"""
struct LogLike{B, G, P, K <: KDTree} <: VariableSpacing
    boundary::P
    base_size::B
    growth_rate::G
    tree::K
end

function LogLike(cloud::PointCloud, base_size, growth_rate)
    # TODO extract only points/surfaces used for growth rate
    return LogLike(points(cloud), base_size, growth_rate)
end

function LogLike(boundary_points, base_size, growth_rate)
    isempty(boundary_points) &&
        throw(ArgumentError("boundary_points must be non-empty"))
    return LogLike(boundary_points, base_size, growth_rate, _build_boundary_tree(boundary_points))
end

function (s::LogLike)(p::Union{Point, Vec})
    x = _min_distance(p, s.boundary, s.tree)
    inv_growth = 1 - (s.growth_rate - 1)
    a = s.base_size * inv_growth  # characteristic length scale with proper units
    return s.base_size * x / (a + x)
end

"""
    BoundaryLayerSpacing <: VariableSpacing

Smooth spacing transition from fine spacing at the boundary to coarse spacing in the bulk.

Uses physical boundary layer intuition with clear parameters:
- `at_wall`: Spacing at the boundary surface (fine)
- `bulk`: Spacing far from boundaries (coarse)
- `layer_thickness`: Distance over which transition occurs

# Example
```julia
# Fine 0.5m spacing at walls, coarse 10m in bulk, 8m boundary layer
spacing = BoundaryLayerSpacing(boundary, at_wall=0.5m, bulk=10m, layer_thickness=8m)
```

Internally uses sigmoid: `h(d) = at_wall + (bulk - at_wall) * σ(d)`
where `σ(d) = 1 / (1 + exp(-(d - δ/2) / (δ/6)))` and δ = layer_thickness.
"""
struct BoundaryLayerSpacing{B, L, P, K <: KDTree} <: VariableSpacing
    boundary::P
    at_wall::B
    bulk::B
    layer_thickness::L
    tree::K
end

function BoundaryLayerSpacing(boundary_points; at_wall, bulk, layer_thickness)
    isempty(boundary_points) &&
        throw(ArgumentError("boundary_points must be non-empty"))
    ustrip(layer_thickness) > 0 ||
        throw(ArgumentError("layer_thickness must be positive, got $layer_thickness"))

    # Ensure at_wall and bulk have compatible types
    B = promote_type(typeof(at_wall), typeof(bulk))
    h_wall = convert(B, at_wall)
    h_bulk = convert(B, bulk)

    return BoundaryLayerSpacing(
        boundary_points,
        h_wall,
        h_bulk,
        layer_thickness,
        _build_boundary_tree(boundary_points),
    )
end

function (s::BoundaryLayerSpacing)(p::Union{Point, Vec})
    # Distance to nearest boundary point
    x = _min_distance(p, s.boundary, s.tree)
    d = ustrip(x)

    # Sigmoid transition: center at δ/2, width ≈ δ/6 (smooth S-curve over boundary layer)
    δ = ustrip(s.layer_thickness)
    center = δ / 2
    width = δ / 6

    σ = inv(1 + exp(-(d - center) / width))
    return s.at_wall + (s.bulk - s.at_wall) * σ
end
