"""
    isinside(testpoint::Point{𝔼{2}}, pts::AbstractVector{<:Point{𝔼{2}}}) -> Bool
    isinside(testpoint::Point{𝔼{N}}, cloud::Union{PointCloud, PointBoundary}) -> Bool

Test whether `testpoint` lies inside the closed domain defined by the boundary points.

For 2D, uses the winding number algorithm — `pts` must be ordered sequentially around
the polygon boundary (clockwise or counter-clockwise). An `ArgumentError` is thrown
if the points do not form a valid ordered polygon.

For 3D, uses a Green's function approach over the boundary surfaces.

!!! note
    WhatsThePoint's `isinside` tests point-in-polygon/volume membership for meshless
    point clouds. This is distinct from Meshes.jl's `isinside` which operates on
    geometric domain objects. No dispatch collision exists — argument types differ.
"""
function isinside(testpoint::Point{𝔼{2}}, pts::AbstractVector{<:Point{𝔼{2}, C}}) where {C}
    _validate_polygon_ordering(pts)

    # first check if point is coincident with any surf surface point and return true if so
    r = map(p -> norm(p - testpoint), pts)
    T = CoordRefSystems.mactype(C)
    unitful_eps = eps(T) * unit(lentype(C))
    any(r .< 1.0e2 * unitful_eps) && return true

    # compute sum of angles from first to last points
    @views sumangles = sum(∠.(pts[1:(end - 1)], testpoint, pts[2:end]))

    # compute last segment from last point to first to complete the loop
    sumangles += ∠(pts[end], testpoint, pts[1])

    # if point is inside, sum of angles is 2pi, if outside, sum of angles is 0.
    return abs(sumangles) < (1.0e3 * eps(T) * Unitful.rad) ? false : true
end

function _validate_polygon_ordering(pts::AbstractVector{<:Point{𝔼{2}, C}}) where {C}
    n = length(pts)
    n < 3 && throw(
        ArgumentError(
            "need at least 3 points to define a polygon, got $n"
        )
    )
    # Signed area via shoelace formula — zero area indicates self-intersecting
    # (unordered) vertices or collinear points
    T = CoordRefSystems.mactype(C)
    sa = zero(T)
    xmin = xmax = ustrip(to(pts[1])[1])
    ymin = ymax = ustrip(to(pts[1])[2])
    for i in 1:n
        j = mod1(i + 1, n)
        xi, yi = ustrip(to(pts[i])[1]), ustrip(to(pts[i])[2])
        xj, yj = ustrip(to(pts[j])[1]), ustrip(to(pts[j])[2])
        sa += xi * yj - xj * yi
        xmin = min(xmin, xi)
        xmax = max(xmax, xi)
        ymin = min(ymin, yi)
        ymax = max(ymax, yi)
    end
    sa /= 2
    bbox_area = (xmax - xmin) * (ymax - ymin)
    if bbox_area > zero(T) && abs(sa) < T(1.0e-10) * bbox_area
        throw(
            ArgumentError(
                "polygon points do not appear to be ordered sequentially around the boundary; " *
                    "the 2D isinside winding number algorithm requires points ordered in a loop " *
                    "(clockwise or counter-clockwise)"
            )
        )
    end
    return nothing
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
