abstract type AbstractSpacing end
abstract type VariableSpacing <: AbstractSpacing end

# Helper function for computing Euclidean distance between points
# Used by variable spacing implementations
# Requires Euclidean manifold - uses flat space distance metric
distance(p1::Point{𝔼{N}}, p2::Point{𝔼{N}}) where {N} = evaluate(Euclidean(), p1, p2)
distance(p1::Vec{N}, p2::Vec{N}) where {N} = evaluate(Euclidean(), p1, p2)

# Find minimum distance from point p to any point in boundary without allocating
function _min_distance(p, boundary)
    dmin = distance(p, first(boundary))
    @inbounds for i in 2:length(boundary)
        d = distance(p, boundary[i])
        dmin = ifelse(d < dmin, d, dmin)
    end
    return dmin
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
struct LogLike{B, G, P} <: VariableSpacing
    boundary::P
    base_size::B
    growth_rate::G
end

function LogLike(cloud::PointCloud, base_size, growth_rate)
    # TODO extract only points/surfaces used for growth rate
    return LogLike(collect(points(cloud)), base_size, growth_rate)
end

function (s::LogLike)(p::Union{Point, Vec})
    x = _min_distance(p, s.boundary)
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
struct BoundaryLayerSpacing{B, L, P} <: VariableSpacing
    boundary::P
    at_wall::B
    bulk::B
    layer_thickness::L
end

function BoundaryLayerSpacing(boundary_points; at_wall, bulk, layer_thickness)
    # Validate inputs
    δ = Float64(ustrip(layer_thickness))
    δ > 0 || throw(ArgumentError("layer_thickness must be positive, got $layer_thickness"))

    # Ensure at_wall and bulk have compatible types
    B = promote_type(typeof(at_wall), typeof(bulk))
    h_wall = convert(B, at_wall)
    h_bulk = convert(B, bulk)

    return BoundaryLayerSpacing{B, typeof(layer_thickness), typeof(boundary_points)}(
        boundary_points,
        h_wall,
        h_bulk,
        layer_thickness,
    )
end

function (s::BoundaryLayerSpacing)(p::Union{Point, Vec})
    # Distance to nearest boundary point
    x = _min_distance(p, s.boundary)
    d = Float64(ustrip(x))

    # Sigmoid transition: center at δ/2, width ≈ δ/6 (smooth S-curve over boundary layer)
    δ = Float64(ustrip(s.layer_thickness))
    center = δ / 2
    width = δ / 6

    σ = inv(1 + exp(-(d - center) / width))
    return s.at_wall + (s.bulk - s.at_wall) * σ
end
