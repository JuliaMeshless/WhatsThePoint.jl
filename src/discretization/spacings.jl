abstract type AbstractSpacing end
abstract type VariableSpacing <: AbstractSpacing end

# Helper function for computing Euclidean distance between points
# Used by variable spacing implementations
# Requires Euclidean manifold - uses flat space distance metric
distance(p1::Point{𝔼{N}}, p2::Point{𝔼{N}}) where {N} = evaluate(Euclidean(), p1, p2)
distance(p1::Vec{N}, p2::Vec{N}) where {N} = evaluate(Euclidean(), p1, p2)

"""
    ConstantSpacing{L<:Unitful.Length} <: AbstractSpacing

Constant node spacing.
"""
struct ConstantSpacing{L<:Unitful.Length} <: AbstractSpacing
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
struct LogLike{B,G} <: VariableSpacing
    boundary::Any
    base_size::B
    growth_rate::G
end

function LogLike(cloud::PointCloud, base_size, growth_rate)
    # TODO extract only points/surfaces used for growth rate
    return LogLike(points(cloud), base_size, growth_rate)
end

function (s::LogLike)(p::Union{Point,Vec})
    x, _ = findmin_turbo(distance.(p, s.boundary))
    inv_growth = 1 - (s.growth_rate - 1)
    a = s.base_size * inv_growth  # characteristic length scale with proper units
    return s.base_size * x / (a + x)
end

"""
    Power <: VariableSpacing

Node spacing based on a power of the distance to nearest boundary ``x^{g}`` where ``x`` is
    the distance to the nearest boundary and ``g`` is the growth_rate.
"""
struct Power{B,G} <: VariableSpacing
    boundary::Any
    base_size::B
    growth_rate::G
    function Power(cloud::PointCloud, surfaces, base_size::Real, growth_rate::Real)
        # TODO extract only points/surfaces used for growth rate
        error("TODO extract only points/surfaces used for growth rate")
        return new{B,G}(points, base_size, growth_rate)
    end
end

function (s::Power)(p::Union{Point,Vec})
    x, _ = findmin_turbo(distance.(p, s.boundary))
    return s.base_size * x^s.growth_rate
end

"""
    PiecewiseSpacing <: VariableSpacing

Piecewise-constant spacing as function of distance to nearest boundary point.

The mapping is defined by distance bins `(d_max, h)`:
- If `distance <= d_max`, spacing is `h`
- Bins are evaluated in ascending `d_max` order
- Last bin should typically use `Inf` as catch-all

# Example
```julia
spacing = PiecewiseSpacing(points(boundary), [(10.0, 1.0), (Inf, 5.0)])
```
"""
struct PiecewiseSpacing{B} <: VariableSpacing
    boundary::Any
    bins::Vector{Tuple{Float64,B}}
end

function PiecewiseSpacing(boundary_points, bins)
    isempty(bins) && throw(ArgumentError("bins must contain at least one (d_max, h) entry"))

    parsed = Tuple{Float64,Any}[]
    for (dmax, h) in bins
        dmax_val = dmax == Inf ? Inf : Float64(ustrip(dmax))
        push!(parsed, (dmax_val, h))
    end
    sort!(parsed, by=first)

    h_type = typeof(parsed[1][2])
    typed_bins = Vector{Tuple{Float64,h_type}}(undef, length(parsed))
    for i in eachindex(parsed)
        typed_bins[i] = (parsed[i][1], convert(h_type, parsed[i][2]))
    end

    return PiecewiseSpacing{h_type}(boundary_points, typed_bins)
end

function (s::PiecewiseSpacing)(p::Union{Point,Vec})
    x, _ = findmin_turbo(distance.(p, s.boundary))
    dx = Float64(ustrip(x))

    @inbounds for (dmax, h) in s.bins
        dx <= dmax && return h
    end

    # Fallback to last bin (should be unreachable if last dmax is Inf).
    return s.bins[end][2]
end

"""
    SigmoidSpacing <: VariableSpacing

Smooth spacing transition from `h_boundary` near the surface to `h_interior`
away from the surface using a logistic curve.

`distance` is computed to the nearest boundary point and the blend is:
`σ = 1 / (1 + exp(-(distance - center) / width))`

spacing = `h_boundary + (h_interior - h_boundary) * σ`
"""
struct SigmoidSpacing{B,C,W} <: VariableSpacing
    boundary::Any
    h_boundary::B
    h_interior::B
    transition_center::C
    transition_width::W
end

function SigmoidSpacing(
    boundary_points,
    h_boundary,
    h_interior,
    transition_center,
    transition_width,
)
    width = Float64(ustrip(transition_width))
    width > 0 || throw(ArgumentError("transition_width must be positive, got $transition_width"))

    B = promote_type(typeof(h_boundary), typeof(h_interior))
    hb = convert(B, h_boundary)
    hi = convert(B, h_interior)
    return SigmoidSpacing{B,typeof(transition_center),typeof(transition_width)}(
        boundary_points,
        hb,
        hi,
        transition_center,
        transition_width,
    )
end

function (s::SigmoidSpacing)(p::Union{Point,Vec})
    x, _ = findmin_turbo(distance.(p, s.boundary))
    dx = Float64(ustrip(x))
    c = Float64(ustrip(s.transition_center))
    w = Float64(ustrip(s.transition_width))

    σ = inv(1 + exp(-(dx - c) / w))
    return s.h_boundary + (s.h_interior - s.h_boundary) * σ
end
