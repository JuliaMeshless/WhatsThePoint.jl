"""
    emptyspace(testpoint, points)

Check if a point occupies empty space within a certain tolerance.

# Examples
```jldoctest
julia> emptyspace(Point(0.5, 0.5), Point.([(1, 0), (0, 0), (1, 1), (0, 1)]), 0.5m)
true

julia> emptyspace(Point(0.5, 0.5), Point.([(1, 0), (0, 0), (1, 1), (0, 1)]), 1m)
false
```

"""
function emptyspace(testpoint::P, points::Vector{P}, tol::Unitful.Length) where {P<:Point}
    return all(p -> norm(testpoint - p) > tol, points)
end

function emptyspace(points::Vector{P}, testpoint::P, tol::Unitful.Length) where {P<:Point}
    return emptyspace(testpoint, points, tol)
end
