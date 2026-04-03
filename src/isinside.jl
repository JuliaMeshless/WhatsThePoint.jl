function isinside(testpoint::Point{𝔼{2}}, pts::AbstractVector{<:Point{𝔼{2}, C}}) where {C}
    # WARNING: this only works if the points are ordered in a loop...

    # first check if point is coincident with any surf surface point and return true if so
    r = map(p -> norm(p - testpoint), pts)
    T = CoordRefSystems.mactype(C)
    unitful_eps = eps(T) * unit(lentype(C))
    any(r .< 1.0e2 * unitful_eps) && return true

    # compute sum of angles from first to last points
    @views sumangles = sum(∠.(pts[1:(end - 1)], testpoint, pts[2:end]))

    # compute last segment from last point to first to complete the loop
    sumangles += ∠(pts[end], testpoint, pts[1])

    # TODO need to add a check if a multiple of 2*pi or 0 ??? does this make sense

    # if point is inside, sum of angles is 2pi, if outside, sum of angles is 0.
    return abs(sumangles) < (1.0e3 * eps(T) * Unitful.rad) ? false : true
end

function isinside(testpoint::Point{𝔼{2}}, cloud::PointCloud{𝔼{2}})
    return isinside(testpoint, points(boundary(cloud)))
end

function isinside(testpoint::Point{𝔼{2}}, surf::PointSurface{𝔼{2}})
    return isinside(testpoint, points(surf))
end

function isinside(testpoint::AbstractVector, surf::PointSurface{𝔼{Dim}}) where {Dim}
    return isinside(Point{𝔼{Dim}}(testpoint), surf)
end

function isinside(
        testpoint::Point{𝔼{N}},
        cloud::Union{PointCloud{𝔼{N}}, PointBoundary{𝔼{N}}},
    ) where {N}
    g = mapreduce(s -> _greens(testpoint, s), +, surfaces(cloud))
    # include the -4π missing from _greens in the inequality here
    return g < -2π ? true : false
end

function _greens(testpoint::Point{𝔼{N}}, surf::PointSurface{𝔼{N}, C}) where {N, C <: CRS}
    T = CoordRefSystems.mactype(C)
    _greens_kernel = let testpoint = testpoint
        geom -> begin
            (; point, normal, area) = geom
            dist = testpoint - point
            return area * dist ⋅ normal / norm(dist)^3
        end
    end
    g = tmapreduce(_greens_kernel, +, surf; init = zero(T))
    return g # true ∇G⋅n eval should be divided by -4π here
end
