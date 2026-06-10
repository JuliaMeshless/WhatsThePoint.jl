"""
    RepelForceModel

Abstract type for node-repulsion force laws used by [`repel`](@ref).

A concrete subtype `M <: RepelForceModel` must implement
[`compute_force(m::M, u::Real)`](@ref compute_force).
"""
abstract type RepelForceModel end

"""
    compute_force(model::RepelForceModel, u::Real) -> Real

Evaluate the force magnitude at normalized separation `u = r / s`, where `r` is
the distance between two points and `s` is the local target spacing.

The returned scalar multiplies the unit vector `(xᵢ − xⱼ) / r` in the repulsion
step, so positive values push `xᵢ` away from `xⱼ` and negative values pull it
toward `xⱼ`. Concrete subtypes of [`RepelForceModel`](@ref) must implement a
method for this function.
"""
function compute_force end

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

"""
    StrongSpacingForce(β=0.2, γ=3)

Force law `F(u) = (1 − u²) / (u² + β)^γ` with a zero at `u = 1` and a
configurable singularity strength `γ`. Like [`SpacingEquilibriumForce`](@ref)
but with a stronger repulsive core: at small `u` the force scales as
`u^(-2γ)` instead of `u^(-4)`. This breaks balanced standoffs where
neighbor forces cancel the weaker default core.

`γ = 2` recovers [`SpacingEquilibriumForce`](@ref). `γ = 3` (default) is
strong enough to break typical standoffs in a few iterations. Higher values
increase the repulsive kick at close range but may require more iterations
to settle; the displacement cap in [`repel`](@ref) prevents runaway.
"""
struct StrongSpacingForce{T <: Real} <: RepelForceModel
    β::T
    γ::T
end

StrongSpacingForce() = StrongSpacingForce(0.2, 3.0)
StrongSpacingForce(β::Real) = StrongSpacingForce(β, 3.0)

@inline function compute_force(m::StrongSpacingForce, u::Real)
    u² = u * u
    return (1 - u²) / (u² + m.β)^m.γ
end
