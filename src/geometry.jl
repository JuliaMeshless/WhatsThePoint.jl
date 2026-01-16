"""
    centroid(pts::AbstractVector{<:Point})

Compute the centroid (geometric center) of a collection of points.
"""
function centroid(pts::AbstractVector{<:Point{M, C}}) where {M, C}
    isempty(pts) && throw(ArgumentError("Cannot compute centroid of empty collection"))
    coords_all = to.(pts)
    coords_sum = reduce(+, coords_all)
    return Point(Tuple(coords_sum ./ length(pts)))
end

"""
    boundingbox(pts::AbstractVector{<:Point})

Compute the axis-aligned bounding box of a collection of points.
"""
function boundingbox(pts::AbstractVector{<:Point{M, C}}) where {M, C}
    isempty(pts) && throw(ArgumentError("Cannot compute bounding box of empty collection"))
    coords_all = to.(pts)
    mins = reduce((a, b) -> min.(a, b), coords_all)
    maxs = reduce((a, b) -> max.(a, b), coords_all)
    return Box(Point(Tuple(mins)), Point(Tuple(maxs)))
end
