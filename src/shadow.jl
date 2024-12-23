struct ShadowPoints{O,T}
    Δ::T
    function ShadowPoints(Δ::T, order=1) where {T}
        return new{order,T}(Δ)
    end
end

ShadowPoints(Δ::T) where {T<:Number} = ShadowPoints(_ -> Δ)

function generate_shadows(points, normals, shadow::ShadowPoints)
    return map((p, n) -> Point((p - shadow.Δ(p) * n)...), points, normals)
end

function Base.show(io::IO, ::MIME"text/plain", s::ShadowPoints{O,T}) where {O,T}
    println(io, "ShadowPoints{$O}: $(s.Δ)")
    return nothing
end

Base.show(io::IO, ::ShadowPoints) = println(io, "ShadowPoints")
