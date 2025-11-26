struct ShadowPoints{O,T}
    Δ::T
    function ShadowPoints(Δ::T, order=1) where {T}
        return new{order,T}(Δ)
    end
end

ShadowPoints(Δ::T) where {T<:Number} = ShadowPoints(_ -> Δ)

function generate_shadows(points, normals, shadow::ShadowPoints)
    Δ_func = shadow.Δ isa Function ? shadow.Δ : (_ -> shadow.Δ)
    return map(points, normals) do p, n
        # Handle both Point objects and raw coordinate vectors
        coords = p isa Point ? to(p) : p
        # For Δ function evaluation, convert coords back to Point if needed
        p_for_func = p isa Point ? p : Point(p...)
        Δ_val = Δ_func(p_for_func)
        shadow_coords = coords - Δ_val * n
        return Point(shadow_coords...)
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::ShadowPoints{O,T}) where {O,T}
    println(io, "ShadowPoints{$O}: $(s.Δ)")
    return nothing
end

Base.show(io::IO, ::ShadowPoints) = println(io, "ShadowPoints")
