"""
    emptyspace(testpoint, points)

Check if a point occupies empty space within a certain tolerance.
"""
function emptyspace(testpoint::P, points::Vector{P}, tol::Unitful.Length) where {P<:Point}
    return all(p -> norm(testpoint - p) > tol, points)
end

function emptyspace(points::Vector{P}, testpoint::P, tol::Unitful.Length) where {P<:Point}
    return emptyspace(testpoint, points, tol)
end
