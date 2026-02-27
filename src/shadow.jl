"""
    ShadowPoints(Δ, order=1)
    ShadowPoints(Δ::Number, order)

Shadow point configuration for generating virtual points offset inward from the boundary.
`Δ` is the offset distance (constant or a function of position). `order` is the derivative
order for Hermite-type boundary condition enforcement.
"""
struct ShadowPoints{O, T}
    Δ::T
    function ShadowPoints(Δ::T, order = 1) where {T}
        return new{order, T}(Δ)
    end
end

ShadowPoints(Δ::T, order) where {T <: Number} = ShadowPoints(_ -> Δ, order)
(s::ShadowPoints)(p) = s.Δ(p)

"""
    generate_shadows(points, normals, shadow::ShadowPoints)
    generate_shadows(surf::PointSurface, shadow::ShadowPoints)
    generate_shadows(cloud::PointCloud, shadow::ShadowPoints)

Generate shadow points offset inward from the boundary along the normal direction by the
distance specified in `shadow`. Returns a vector of `Point` objects.
"""
function generate_shadows(points, normals, shadow::ShadowPoints)
    return map(points, normals) do p, n
        # Handle both Point objects and raw coordinate vectors
        coords = p isa Point ? to(p) : p
        # For Δ function evaluation, convert coords back to Point if needed
        p_for_func = p isa Point ? p : Point(p...)
        Δ_val = shadow(p_for_func)
        shadow_coords = coords - Δ_val * n
        return Point(shadow_coords...)
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::ShadowPoints{O, T}) where {O, T}
    println(io, "ShadowPoints{$O}: $(s.Δ)")
    return nothing
end

Base.show(io::IO, ::ShadowPoints) = println(io, "ShadowPoints")
