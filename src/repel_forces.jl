"""
    RepelForceModel

Abstract type for node-repulsion force laws used by [`repel`](@ref).

A concrete subtype `M <: RepelForceModel` must implement

    compute_force(m::M, u::Real) -> Real

where `u = r / s` is the ratio of inter-point distance to local target spacing.
The returned scalar multiplies the unit vector `(xᵢ − xⱼ) / r` in the repulsion
step, so positive values push `xᵢ` away from `xⱼ` and negative values pull it
toward `xⱼ`.
"""
abstract type RepelForceModel end

"""
    InverseDistanceForce(β=0.2)

Original Miotti (2023) force law `F(u) = 1 / (u² + β)²`. Purely repulsive and
monotonically decreasing. `β > 0` softens the force near `u = 0` and keeps it
finite. Has no root, so equilibrium is reached only through damping (`α`).
"""
struct InverseDistanceForce{T <: Real} <: RepelForceModel
    β::T
end

InverseDistanceForce() = InverseDistanceForce(0.2)

@inline compute_force(m::InverseDistanceForce, u::Real) = inv((u * u + m.β)^2)

"""
    SpacingEquilibriumForce(β=0.2)

Force law `F(u) = (1 − u²) / (u² + β)²` with a zero at `u = 1`. Repulsive for
`u < 1` (points closer than the target spacing push apart), attractive for
`u > 1` (points farther than the target pull together), and zero at the target
spacing itself. `β > 0` softens the amplitude near `u = 0`.

Shares the `β`-softened denominator with [`InverseDistanceForce`](@ref); the
numerator `(1 − u²)` introduces the equilibrium at `u = 1` without changing
the small-`u` behavior.
"""
struct SpacingEquilibriumForce{T <: Real} <: RepelForceModel
    β::T
end

SpacingEquilibriumForce() = SpacingEquilibriumForce(0.2)

@inline function compute_force(m::SpacingEquilibriumForce, u::Real)
    u² = u * u
    return (1 - u²) / (u² + m.β)^2
end
