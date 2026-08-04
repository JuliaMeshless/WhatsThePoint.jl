"""
    AbstractSpacing

Interface for spacing functions that control node density during discretization.

Subtypes must be callable with a single `Point` or `Vec` argument and return a `Unitful.Length`
representing the desired node spacing at that location.

    (s::MySpacing)(p::Union{Point, Vec}) -> Unitful.Length

See [`ConstantSpacing`](@ref), [`LogLike`](@ref), and [`BoundaryLayerSpacing`](@ref) for
concrete implementations.
"""
abstract type AbstractSpacing end
abstract type VariableSpacing <: AbstractSpacing end

# O(log n) nearest-neighbor query via KDTree. Distances come back in the
# tree's (= boundary data's) machine type regardless of the query point's.
function _min_distance(p, boundary, tree::KDTree)
    q = ustrip.(to(p))
    idxs, dists = knn(tree, q, 1)
    return dists[1] * unit(eltype(to(first(boundary))))
end

function _build_boundary_tree(boundary_points)
    coords = [ustrip.(to(p)) for p in boundary_points]
    return KDTree(coords)
end

"""
    ConstantSpacing{L<:Unitful.Length} <: AbstractSpacing

Constant node spacing.
"""
struct ConstantSpacing{L <: Unitful.Length} <: AbstractSpacing
    Δx::L
end
(s::ConstantSpacing)() = s.Δx
(s::ConstantSpacing)(_) = s.Δx

"""
    LogLike <: VariableSpacing

Node spacing based on a log-like function of the distance to nearest boundary
    ``h(x) = h_0 \\, x/(x+a)`` where ``x`` is the distance to the nearest boundary,
    ``h_0`` is `base_size`, and ``a = h_0 (2 - g)`` is the characteristic length
    controlling the growth rate, with ``g`` the conventional growth rate parameter.
"""
struct LogLike{B, G, P, K <: KDTree} <: VariableSpacing
    boundary::P
    base_size::B
    growth_rate::G
    tree::K
end

function LogLike(cloud::PointCloud, base_size, growth_rate)
    # TODO extract only points/surfaces used for growth rate
    return LogLike(points(cloud), base_size, growth_rate)
end

function LogLike(boundary_points, base_size, growth_rate)
    isempty(boundary_points) &&
        throw(ArgumentError("boundary_points must be non-empty"))
    return LogLike(boundary_points, base_size, growth_rate, _build_boundary_tree(boundary_points))
end

function (s::LogLike)(p::Union{Point, Vec})
    x = _min_distance(p, s.boundary, s.tree)
    inv_growth = 1 - (s.growth_rate - 1)
    a = s.base_size * inv_growth  # characteristic length scale with proper units
    return s.base_size * x / (a + x)
end

"""
    BoundaryLayerSpacing <: VariableSpacing

Smooth spacing transition from fine spacing at the boundary to coarse spacing in the bulk.

Uses physical boundary layer intuition with clear parameters:
- `at_wall`: Spacing at the boundary surface (fine)
- `bulk`: Spacing far from boundaries (coarse)
- `layer_thickness`: Distance over which transition occurs

# Example
```julia
# Fine 0.5m spacing at walls, coarse 10m in bulk, 8m boundary layer
spacing = BoundaryLayerSpacing(boundary, at_wall=0.5m, bulk=10m, layer_thickness=8m)
```

Internally uses sigmoid: `h(d) = at_wall + (bulk - at_wall) * σ(d)`
where `σ(d) = 1 / (1 + exp(-(d - δ/2) / (δ/6)))` and δ = layer_thickness.
"""
struct BoundaryLayerSpacing{B, L, P, K <: KDTree} <: VariableSpacing
    boundary::P
    at_wall::B
    bulk::B
    layer_thickness::L
    tree::K
end

function BoundaryLayerSpacing(boundary_points; at_wall, bulk, layer_thickness)
    isempty(boundary_points) &&
        throw(ArgumentError("boundary_points must be non-empty"))
    ustrip(layer_thickness) > 0 ||
        throw(ArgumentError("layer_thickness must be positive, got $layer_thickness"))

    # Ensure at_wall and bulk have compatible types
    B = promote_type(typeof(at_wall), typeof(bulk))
    h_wall = convert(B, at_wall)
    h_bulk = convert(B, bulk)

    return BoundaryLayerSpacing(
        boundary_points,
        h_wall,
        h_bulk,
        layer_thickness,
        _build_boundary_tree(boundary_points),
    )
end

function (s::BoundaryLayerSpacing)(p::Union{Point, Vec})
    # Distance to nearest boundary point
    x = _min_distance(p, s.boundary, s.tree)
    d = ustrip(x)

    # Sigmoid transition: center at δ/2, width ≈ δ/6 (smooth S-curve over boundary layer)
    δ = ustrip(s.layer_thickness)
    center = δ / 2
    width = δ / 6

    σ = inv(1 + exp(-(d - center) / width))
    return s.at_wall + (s.bulk - s.at_wall) * σ
end

"""
    PatchLayerSpacing <: VariableSpacing

Spacing that tightens toward one patch and grows without bound away from it:

``h(d) = at\\_wall \\, (1 + d/\\delta)``

where ``d`` is the distance to the nearest seed point and ``\\delta`` is
`layer_thickness`. Unlike [`BoundaryLayerSpacing`](@ref) there is **no
far-field**: the bulk ceiling belongs in a [`MinSpacing`](@ref) alongside it, so
that bulk is set in exactly one place no matter how many patches are refined.

Two properties this form has and the sigmoid does not, both of which matter when
`at_wall` is the abscissa of a refinement sequence:

  - `h(0) == at_wall` exactly. `BoundaryLayerSpacing` returns
    `at_wall + (bulk - at_wall)/(1 + exp(3))`, i.e. ~4.7% of the way to `bulk`
    already at the wall — a bias that varies with `at_wall`.
  - The field scales in pure amplitude with `at_wall`, so halving `at_wall`
    halves `h` everywhere the patch governs. The shape of the graded region is
    invariant; only its level moves.

The growth rate is `at_wall/δ`, and since `h = at_wall + (at_wall/δ)·d` with `d`
1-Lipschitz, the field is exactly `(at_wall/δ)`-Lipschitz. Setting
`δ = at_wall / g` therefore pins `|∇h| = g`.

That matters most for **surface** sampling, which has no gradient limiter of its
own: `Octree` smooths the volume field to `max_growth`, but `PointBoundary` uses
whatever it is handed verbatim. A boundary field built as a [`MinSpacing`](@ref)
of `PatchLayerSpacing`s with `δᵢ = at_wallᵢ / g`, plus a `ConstantSpacing`
ceiling (0-Lipschitz), is `g`-Lipschitz everywhere — a min of `g`-Lipschitz
functions is `g`-Lipschitz — so refinement around one patch grades smoothly onto
its neighbours instead of stopping at the patch boundary. `BoundaryLayerSpacing`
carries no such bound (its sigmoid slope peaks near `1.5·(bulk − at_wall)/δ`)
and should not be used where surface smoothness is required.
"""
struct PatchLayerSpacing{B, L, P, K <: KDTree} <: VariableSpacing
    seeds::P
    at_wall::B
    layer_thickness::L
    tree::K
end

function PatchLayerSpacing(seed_points; at_wall, layer_thickness)
    isempty(seed_points) && throw(ArgumentError("seed_points must be non-empty"))
    ustrip(layer_thickness) > 0 ||
        throw(ArgumentError("layer_thickness must be positive, got $layer_thickness"))
    return PatchLayerSpacing(
        seed_points, at_wall, layer_thickness, _build_boundary_tree(seed_points)
    )
end

function (s::PatchLayerSpacing)(p::Union{Point, Vec})
    d = _min_distance(p, s.seeds, s.tree)
    return s.at_wall * (1 + d / s.layer_thickness)
end

"""
    MinSpacing(parts...) <: VariableSpacing

Pointwise minimum of several spacing fields: `h(p) = min(partᵢ(p))`.

Each part is keyed on its own seeds, so one surface can be refined without that
refinement having to be expressed as a single distance-to-nearest-wall function.
The idiom is one [`PatchLayerSpacing`](@ref) per patch plus a single
`ConstantSpacing` carrying the bulk ceiling:

```julia
h = MinSpacing(
    ConstantSpacing(2u"mm"),                                                  # bulk, set once
    PatchLayerSpacing(fin_pts;  at_wall = 0.35u"mm", layer_thickness = 2u"mm"),
    PatchLayerSpacing(bore_pts; at_wall = 0.45u"mm", layer_thickness = 2u"mm"),
)
```

`min` is the correct combinator because each `PatchLayerSpacing` is monotone
non-decreasing away from its own patch: the minimum therefore picks the locally
governing constraint, and the result is continuous with kinks and no jumps.

A part only *binds* where it is finer than every other part. An overlay asking
for a spacing coarser than what the existing fields already deliver is a no-op —
check the achieved spacing rather than assuming the request took effect.
"""
struct MinSpacing{T <: Tuple} <: VariableSpacing
    parts::T

    # Inner constructor: a single-field struct whose field is a `Tuple` already
    # gets an outer `MinSpacing(::Tuple)` for free, so validating in an outer
    # method of the same signature would overwrite it.
    function MinSpacing(parts::T) where {T <: Tuple}
        isempty(parts) && throw(ArgumentError("MinSpacing needs at least one part"))
        return new{T}(parts)
    end
end

MinSpacing(parts::AbstractSpacing...) = MinSpacing(parts)

"""
    ScaledSpacing(inner, factor) <: VariableSpacing

`h(p) = factor * inner(p)`. Ties one spacing field to another by a constant
ratio instead of restating it.

The motivating use is sampling a boundary at a fixed multiple of the *interior*
field, so that surface and interior resolution stay in a known proportion
everywhere rather than drifting apart wherever the interior is refined:

```julia
h_bnd = ScaledSpacing(h_vol, 1.15)   # walls 15% coarser than the interior meets them
```

`factor > 1` keeps the surface coarser than the interior, which matters when
stencil selection expects a boundary point's neighbours to be predominantly
interior points.
"""
struct ScaledSpacing{S, T <: Real} <: VariableSpacing
    inner::S
    factor::T

    function ScaledSpacing(inner::S, factor::T) where {S, T <: Real}
        factor > 0 || throw(ArgumentError("factor must be positive, got $factor"))
        return new{S, T}(inner, factor)
    end
end

(s::ScaledSpacing)(p::Union{Point, Vec}) = s.inner(p) * s.factor

# `map` over a Tuple stays inferable when the parts have different types.
(s::MinSpacing)(p::Union{Point, Vec}) = reduce(min, map(f -> f(p), s.parts))
