abstract type AbstractSpacing end
abstract type VariableSpacing <: AbstractSpacing end

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
    surfaces
    base_size::B
    growth_rate::G
end

function LogLike(cloud::PointCloud, base_size::Real, growth_rate::Real)
    # TODO extract only points/surfaces used for growth rate
    return LogLike(cloud.points, base_size, growth_rate)
end

function (s::LogLike)(p::Union{Point,Vec})
    x, _ = findmin_turbo(distance.(p, s.boundary))
    inv_growth = 1 - (s.growth_rate - 1)
    return s.base_size * x / (inv_growth + x)
end

"""
    Power <: VariableSpacing

Node spacing based on a power of the distance to nearest boundary ``x^{g}`` where ``x`` is
    the distance to the nearest boundary and ``g`` is the growth_rate.
"""
struct Power{B,G} <: VariableSpacing
    surfaces
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
