# ============================================================================
# Spacing guidance — the "step 0" before discretize
# ============================================================================
#
# Picking a spacing `h` blind is the most common way to get an unusable cloud:
# `h` coarser than the domain can host yields a near-empty Poisson-disk
# interior (a 1×1×1 cube with h = 1 m fits *zero* interior points — every
# candidate lands within one disk radius of a wall). `suggest_spacing` is a
# quick, Linux-`du -sh`-style probe of a geometry that reports its extent and a
# sensible baseline spacing, so the user starts from an "ok-ish" cloud and then
# refines. `_guard_coarse_spacing` is the companion safety net inside the
# bridson sampler: it clamps (loudly) a spacing that is too coarse everywhere
# rather than returning an empty cloud silently.

"""
    suggest_spacing(mesh; n_points=nothing, bridson_factor=0.75, verbose=true)
    suggest_spacing(boundary; ...)
    suggest_spacing("model.stl", u"mm"; ...)

Quick geometry probe that recommends a baseline node spacing — the "step 0"
before [`discretize`](@ref). Reports the domain extent, enclosed volume, and
three spacing landmarks, and (with `verbose=true`) prints a short summary.

The recommendation is driven by the shortest bounding-box axis `L_min`, because
that axis sets how coarse a Poisson-disk fill can be before its interior
collapses. With the bridson disk radius `r = bridson_factor·h`, the interior
along `L_min` has width `L_min − 2r`, so it is empty once
`h ≥ L_min/(2·bridson_factor)` — the reported `h_ceiling`. Returned landmarks:

- `h_ceiling` — coarsest spacing that still hosts *any* interior; spacings at or
  above it yield an empty bridson cloud. **Stay well below this.**
- `h_baseline` — recommended starting point: ≈10 points across the shortest
  axis (or, when `n_points` is given, `cbrt(volume / n_points)` capped to stay
  fillable). Good enough to run a first simulation, then refine where needed.
- `h_fine` — `h_baseline/2`, a second rung for resolving features.

`n_baseline`/`n_fine` are rough volume-point counts (`volume / h³`).

Returns a `NamedTuple` with `extent`, `min_extent`, `max_extent`, `diagonal`,
`volume`, `n_triangles`, `bridson_factor`, `h_ceiling`, `h_baseline`, `h_fine`,
`n_baseline`, and `n_fine` (all spacings/lengths carry units).

# Example
```julia
mesh = import_mesh("bunny.stl", u"m")
g = suggest_spacing(mesh)
cloud = discretize(PointBoundary(mesh), g.h_baseline; alg=Octree(mesh))
```
"""
function suggest_spacing(
        mesh::SimpleMesh{M, C};
        n_points::Union{Nothing, Integer} = nothing,
        bridson_factor::Real = 0.75,
        verbose::Bool = true,
        name::AbstractString = "mesh",
    ) where {M, C}
    T = CoordRefSystems.mactype(C)
    bmin, bmax = _compute_bbox(T, mesh)
    ext = bmax - bmin
    V = abs(_signed_volume(T, mesh))
    lu = unit(Meshes.to(first(Meshes.vertices(mesh)))[1])
    return _spacing_guidance(
        ext, V, Meshes.nelements(mesh), lu;
        n_points, bridson_factor = T(bridson_factor), verbose, name,
        exact_volume = true, count_label = "facets",
    )
end

function suggest_spacing(
        bnd::PointBoundary{M, C};
        n_points::Union{Nothing, Integer} = nothing,
        bridson_factor::Real = 0.75,
        verbose::Bool = true,
        name::AbstractString = "boundary",
    ) where {M, C}
    T = CoordRefSystems.mactype(C)
    pts = points(bnd)
    isempty(pts) && throw(ArgumentError("boundary has no points"))
    lu = unit(Meshes.to(first(pts))[1])
    coords = [SVector{3, T}(ustrip.(Meshes.to(p))...) for p in pts]
    bmin = reduce((a, b) -> min.(a, b), coords)
    bmax = reduce((a, b) -> max.(a, b), coords)
    ext = bmax - bmin
    # No triangles ⇒ no enclosed volume; the bounding-box volume is an
    # over-estimate that keeps the point-count ballparks in the right order of
    # magnitude (flagged as approximate when printed).
    V = prod(ext)
    return _spacing_guidance(
        ext, V, length(pts), lu;
        n_points, bridson_factor = T(bridson_factor), verbose, name,
        exact_volume = false, count_label = "bnd pts",
    )
end

function suggest_spacing(
        path::AbstractString, unit::Unitful.Units;
        name::AbstractString = basename(path), kwargs...,
    )
    return suggest_spacing(import_mesh(path, unit); name, kwargs...)
end

# Core recommendation math, shared by all entry points. `ext`/`V` are stripped
# (in coordinate units), `lu` is the length unit used to re-attach units.
function _spacing_guidance(
        ext::SVector{3, T}, V::T, n_facets::Integer, lu;
        n_points, bridson_factor, verbose, name, exact_volume, count_label,
    ) where {T}
    Lmin = minimum(ext)
    Lmax = maximum(ext)
    Lmin > 0 || throw(ArgumentError("degenerate geometry: zero-width bounding box ($(ext .* lu))"))
    diag = sqrt(sum(abs2, ext))
    f = T(bridson_factor)

    h_ceiling = Lmin / (2f)                 # interior empty at/above this
    raw_baseline = isnothing(n_points) ? Lmin / 10 : cbrt(V / T(n_points))
    # Keep the baseline comfortably under the ceiling (≤ L_min/3 for f=0.75),
    # so even a budget-driven request stays fillable.
    h_baseline = min(raw_baseline, h_ceiling / 2)
    h_fine = h_baseline / 2
    nfun(h) = round(Int, V / h^3)

    res = (
        extent = ext .* lu,
        min_extent = Lmin * lu,
        max_extent = Lmax * lu,
        diagonal = diag * lu,
        volume = V * lu^3,
        n_triangles = n_facets,
        bridson_factor = f,
        h_ceiling = h_ceiling * lu,
        h_baseline = h_baseline * lu,
        h_fine = h_fine * lu,
        n_baseline = nfun(h_baseline),
        n_fine = nfun(h_fine),
    )
    verbose && _print_guidance(res, name, exact_volume, count_label)
    return res
end

function _print_guidance(g, name, exact_volume, count_label)
    rnd(x) = round(ustrip(x); sigdigits = 3) * unit(x)
    vlabel = exact_volume ? "volume       " : "volume (bbox)"
    e = ustrip.(g.extent)
    u = unit(g.min_extent)
    println("geometry: $name")
    println("  extent        $(round(e[1]; sigdigits = 3)) × $(round(e[2]; sigdigits = 3)) × $(round(e[3]; sigdigits = 3)) $u   (min axis $(rnd(g.min_extent)))")
    println("  $vlabel ≈ $(rnd(g.volume))")
    println("  $(rpad(count_label, 13)) $(g.n_triangles)")
    println("  ── spacing (bridson_factor $(g.bridson_factor)) ─────────────")
    println("  h_ceiling     $(rnd(g.h_ceiling))   (coarser ⇒ EMPTY interior — stay below)")
    println("  h_baseline    $(rnd(g.h_baseline))   (≈ $(g.n_baseline) vol pts) ← start here")
    println("  h_fine        $(rnd(g.h_fine))   (≈ $(g.n_fine) vol pts)")
    return nothing
end

# ============================================================================
# Coarse-spacing guard for the bridson sampler
# ============================================================================

"""
    _ClampedSpacing(inner, hmax) <: AbstractSpacing

Spacing that caps `inner` at `hmax`: `min(inner(p), hmax)`. Used by
[`_guard_coarse_spacing`](@ref) to make an everywhere-too-coarse spacing
fillable without discarding the user's variation where it is already fine.
"""
struct _ClampedSpacing{S, Q} <: AbstractSpacing
    inner::S
    hmax::Q
end

function (s::_ClampedSpacing)(p::Union{Point, Vec})
    h = s.inner(p)
    return h < s.hmax ? h : s.hmax
end

function _extract_min_spacing(s::_ClampedSpacing)
    im = _extract_min_spacing(s.inner)
    hm = float(ustrip(s.hmax))
    return isnothing(im) ? hm : min(im, hm)
end

# Finest spacing the *interior* prescribes — the quantity that decides whether
# bridson can fill anything. Sampled on an interior grid rather than via the
# field's global minimum on purpose: a distance-based field bottoms out only in
# an infinitesimal wall shell (`at_wall`), which cannot host a packing on its
# own, so the boundary minimum is misleadingly optimistic. The interior grid
# reflects what the bulk of the volume actually demands; if even its finest
# sample is too coarse, the interior is unfillable and we clamp.
function _probe_min_spacing(spacing, bmin::SVector{3, T}, bmax::SVector{3, T}; n::Int = 5) where {T}
    ext = bmax - bmin
    hmin = typemax(T)
    for i in 0:(n - 1), j in 0:(n - 1), k in 0:(n - 1)
        t = SVector{3, T}(2i + 1, 2j + 1, 2k + 1) / (2n)
        h = _spacing_value(T, spacing, bmin + t .* ext)
        h < hmin && (hmin = h)
    end
    return hmin
end

"""
    _guard_coarse_spacing(spacing, tri_octree, bridson_factor) -> spacing

Bridson safety net. When the finest prescribed spacing over the domain is at or
above the Poisson-disk ceiling `L_min/(2·bridson_factor)` — i.e. a saturated
front would leave the interior empty — emit a loud `@warn` and return a
[`_ClampedSpacing`](@ref) capped at a usable baseline (`L_min/10`) so generation
still yields a cloud. Otherwise returns `spacing` unchanged (the request is
viable and is respected).
"""
function _guard_coarse_spacing(spacing, tri_octree, bridson_factor)
    bmin = tri_octree.mesh_bbox_min
    bmax = tri_octree.mesh_bbox_max
    Lmin = minimum(bmax - bmin)
    h_ceiling = Lmin / (2 * bridson_factor)
    hmin_domain = _probe_min_spacing(spacing, bmin, bmax)
    hmin_domain < h_ceiling && return spacing

    h_target = Lmin / 10
    lu = unit(spacing(Point((0.5 .* (bmin + bmax))...)))
    @warn(
        "Spacing too coarse for this domain — clamping so generation is not empty. " *
            "The finest prescribed spacing exceeds the Poisson-disk ceiling " *
            "(L_min/$(2 * bridson_factor)), so a saturated bridson front would leave the " *
            "interior empty. Capping spacing to ≈10 points across the shortest axis. " *
            "Run `suggest_spacing(mesh)` to choose a spacing deliberately.",
        domain_min_extent = Lmin * lu,
        prescribed_min_spacing = hmin_domain * lu,
        poisson_disk_ceiling = h_ceiling * lu,
        clamped_to = h_target * lu,
    )
    return _ClampedSpacing(spacing, h_target * lu)
end
