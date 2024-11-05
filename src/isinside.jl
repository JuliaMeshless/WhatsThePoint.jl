function isinside(testpoint::Point{𝔼{2}}, points::PointSet{𝔼{2},C}) where {C}
    # WARNING: this only works if the points are ordered in a loop...

    # first check if point is coincident with any surf surface point and return true if so
    r = map(p -> norm(p - testpoint), points)
    T = CoordRefSystems.mactype(C)
    unitful_eps = eps(T) * unit(lentype(C))
    any(r .< 1e2 * unitful_eps) && return true

    # compute sum of angles from first to last points
    @views sumangles = sum(∠.(points[1:(end - 1)], testpoint, points[2:end]))

    # compute last segment from last point to first to complete the loop
    sumangles += ∠(points[end], testpoint, points[1])

    # TODO need to add a check if a multiple of 2*pi or 0 ??? does this make sense

    # if point is inside, sum of angles is 2pi, if outside, sum of angles is 0.
    return abs(sumangles) < (1e3 * eps(T) * Unitful.rad) ? false : true
end

function isinside(testpoint::Point{𝔼{2}}, surf::Union{PointCloud{𝔼{2}},PointPart{𝔼{2}}})
    return isinside(testpoint, point(surf))
end
function isinside(testpoint::Point{𝔼{2}}, surf::PointSurface{𝔼{2}})
    return isinside(testpoint, point(surf))
end

function isinside(testpoint::AbstractVector, surf::PointSurface{𝔼{Dim}}) where {Dim}
    return isinside(Point{𝔼{Dim}}(testpoint), surf)
end

function isinside(
    testpoint::Point{M}, cloud::Union{PointCloud{M},PointPart{M}}
) where {M<:Manifold}
    g = mapreduce(s -> _greens(testpoint, s), +, surfaces(cloud))
    # include the -4π missing from _greens in the inequality here
    unitful_2π = 2π * unit(lentype(crs(testpoint)))
    return g < -unitful_2π ? true : false
end

function _greens(testpoint::Point, surf::PointSurface)
    _greens_kernel = let testpoint = testpoint
        geom -> begin
            (; point, normal, area) = geom
            dist = testpoint - point
            return area * dist ⋅ normal / norm(dist)^3
        end
    end
    g = tmapreduce(_greens_kernel, +, surf)
    return g # true ∇G⋅n eval should be divided by -4π here
end
