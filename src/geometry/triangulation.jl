# Triangulation — the multi-STL triangle geometry.
#
# A `Triangulation` is one or more named STL surfaces ("patches"), each with a
# role (`:container` or `:obstacle`), merged into a single consistently-oriented
# `TriangleIndex` whose normals all point *out of the fluid domain*. That single
# index is what `TriangleOctree`, `isinside`, and `discretize` consume — one
# `find_leaf` + signed-distance query classifies the fluid domain, no CSG.
#
# Surface = points (`PointSurface`); a Triangulation is triangles. Nothing here
# retains a `SimpleMesh` — `import_mesh` is the import-only door, immediately
# reduced to `TriangleIndex`.

const _TRIANGULATION_ROLES = (:container, :obstacle)

"""
    Triangulation{T}

A merged, consistently-oriented collection of triangulated STL surfaces. Built
by [`load_triangulation`](@ref). Holds one `TriangleIndex{T}` (all patches
concatenated, obstacle windings reversed so every normal points out of the
fluid) plus the per-patch identity needed for spacing and boundary-condition
tagging.

Fields:
- `index`   — the merged `TriangleIndex{T}`.
- `names`   — patch names, in load order.
- `ranges`  — triangle-index range of each patch within `index`.
- `roles`   — `:container` or `:obstacle` per patch.

Scope — closed watertight shells only: each patch must be a closed,
outward-oriented solid. Assembling one domain from *open* patch STLs (separate
inlet/outlet/wall files) is unsupported by design — concatenating open patches
leaves unwelded seam vertices, which corrupts the angle-weighted pseudonormals
right where sign queries are least robust (`split_surface!` already covers BC
tagging on a single closed shell). Obstacles must not *cross* the container
wall, but an obstacle resting **flush** on the container (e.g. an impeller
sitting on the tank floor — coincident surfaces) is supported:
[`TriangleOctree`](@ref) detects such seams at build and vetoes the
container's ambiguous fluid vote there in favor of the obstacle.
"""
struct Triangulation{T <: Real}
    index::TriangleIndex{T}
    names::Vector{Symbol}
    ranges::Vector{UnitRange{Int}}
    roles::Vector{Symbol}
end

"""
    patches(tri::Triangulation) -> Vector{Symbol}

Patch names, in load order.
"""
patches(tri::Triangulation) = tri.names

"""
    npatches(tri::Triangulation) -> Int

Number of patches (STL surfaces) in the triangulation.
"""
npatches(tri::Triangulation) = length(tri.names)

num_triangles(tri::Triangulation) = num_triangles(tri.index)

Base.length(tri::Triangulation) = num_triangles(tri.index)

function _patch_position(tri::Triangulation, name::Symbol)
    pos = findfirst(==(name), tri.names)
    isnothing(pos) && throw(
        ArgumentError(
            "no patch named :$name — patches are $(tri.names)",
        ),
    )
    return pos
end

"""
    patch_range(tri::Triangulation, name::Symbol) -> UnitRange{Int}

Triangle-index range of patch `name` within the merged index.
"""
patch_range(tri::Triangulation, name::Symbol) = tri.ranges[_patch_position(tri, name)]

"""
    role(tri::Triangulation, name::Symbol) -> Symbol

Role (`:container` / `:obstacle`) of patch `name`.
"""
role(tri::Triangulation, name::Symbol) = tri.roles[_patch_position(tri, name)]

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    load_triangulation(pairs...; units)
    load_triangulation(filepath; units, name, role)

Load a set of STL (or any GeoIO-supported) surfaces into a single, consistently
oriented [`Triangulation`](@ref).

Multi-part form — each pair is `name => (filepath, role)`:

```julia
tri = load_triangulation(
    :tank     => ("tank.stl",     :container),   # normals as-is
    :impeller => ("impeller.stl", :obstacle),    # winding reversed
    :baffles  => ("baffles.stl",  :obstacle);
    units = u"mm",
)
```

Single-part convenience — one `:container` patch named after the file stem:

```julia
tri = load_triangulation("part.stl"; units = u"mm")
```

Roles orient the merged normals so they all point **out of the fluid domain**:
a `:container` (the tank/vessel) keeps its outward-facing normals; an
`:obstacle` (impeller, baffles) has its triangle winding reversed. Every patch
must be a closed solid with outward-facing normals as stored — the per-patch
signed-volume guard rejects an open or inside-out surface, naming the offending
patch (obstacles are checked *before* their flip). Machine types are promoted to
a common `T` across patches (binary STL loads `Float32`, ASCII `Float64`).

Only closed watertight shells are supported: open-patch assembly (separate
inlet/outlet/wall files) is out of scope by design, and obstacles must not
cross the container wall (flush mounting *is* supported — see the
[`Triangulation`](@ref) docstring for why).
"""
function load_triangulation(
        pairs::Pair{Symbol}...;
        units::Unitful.Units,
    )
    isempty(pairs) && throw(ArgumentError("load_triangulation needs at least one patch"))

    names = Symbol[]
    filepaths = String[]
    roles = Symbol[]
    for (name, spec) in pairs
        (spec isa Tuple && length(spec) == 2) || throw(
            ArgumentError(
                "patch :$name must map to (filepath, role), got $(spec)",
            ),
        )
        filepath, r = spec
        filepath isa AbstractString || throw(
            ArgumentError("patch :$name filepath must be a string, got $(filepath)"),
        )
        r in _TRIANGULATION_ROLES || throw(
            ArgumentError(
                "patch :$name has role :$r — must be one of $(_TRIANGULATION_ROLES)",
            ),
        )
        name in names && throw(ArgumentError("duplicate patch name :$name"))
        push!(names, name)
        push!(filepaths, String(filepath))
        push!(roles, r)
    end

    return _build_triangulation(names, filepaths, roles, units)
end

function load_triangulation(
        filepath::AbstractString;
        units::Unitful.Units,
        name::Symbol = Symbol(first(splitext(basename(filepath)))),
        role::Symbol = :container,
    )
    role in _TRIANGULATION_ROLES || throw(
        ArgumentError("role :$role must be one of $(_TRIANGULATION_ROLES)"),
    )
    return _build_triangulation([name], [String(filepath)], [role], units)
end

function _build_triangulation(
        names::Vector{Symbol},
        filepaths::Vector{String},
        roles::Vector{Symbol},
        units::Unitful.Units,
    )
    # Import each STL once (SimpleMesh lives only here), then promote to a
    # common machine type before building any index.
    meshes = [import_mesh(fp, units) for fp in filepaths]
    T = reduce(promote_type, (CoordRefSystems.mactype(crs(m)) for m in meshes))

    indices = TriangleIndex[]
    for (name, mesh, r) in zip(names, meshes, roles)
        idx = TriangleIndex(T, mesh)
        _verify_patch_orientation(idx, name, r)
        push!(indices, idx)
    end

    merged, ranges = _merge_indices(T, indices, roles)

    _signed_volume(merged) > 0 || @warn(
        "merged triangulation has non-positive signed volume — the obstacles " *
            "may enclose or exceed the container, leaving no valid fluid domain. " *
            "Check patch roles and geometry.",
    )

    return Triangulation{T}(merged, names, ranges, roles)
end

# A patch must be a closed solid with outward-facing normals as stored. The
# scale-relative floor lets a genuinely closed solid through while rejecting
# open surfaces (signed volume ≈ 0) and inside-out ones (negative) — the same
# witness `TriangleOctree` uses, applied per patch before any flip.
function _verify_patch_orientation(index::TriangleIndex{T}, name::Symbol, r::Symbol) where {T}
    bbox_min, bbox_max = _compute_bbox(index)
    floor = T(_DEGENERATE_EPS) * norm(bbox_max - bbox_min)^3
    if _signed_volume(index) <= floor
        throw(
            ArgumentError(
                "patch :$name (role :$r) has non-positive signed volume: it is " *
                    "either an open surface or inside-out. Every patch must be a " *
                    "closed solid with outward-facing normals as stored (obstacles " *
                    "are flipped automatically during merge). Inspect the file, or " *
                    "check the role assignment.",
            ),
        )
    end
    return nothing
end

# Concatenate patch indices into one (vertices, triangles) set, offsetting
# triangle vertex indices per patch and reversing obstacle winding (swap the
# 2nd/3rd vertex) so the rebuilt face normals point out of the fluid.
function _merge_indices(
        ::Type{T},
        indices::Vector{<:TriangleIndex},
        roles::Vector{Symbol},
    ) where {T}
    all_vertices = SVector{3, T}[]
    all_triangles = NTuple{3, Int32}[]
    ranges = UnitRange{Int}[]

    for (idx, r) in zip(indices, roles)
        voff = Int32(length(all_vertices))
        append!(all_vertices, idx.vertices)
        tstart = length(all_triangles) + 1
        flip = r === :obstacle
        for t in idx.triangles
            a = t[1] + voff
            b = t[2] + voff
            c = t[3] + voff
            push!(all_triangles, flip ? (a, c, b) : (a, b, c))
        end
        push!(ranges, tstart:length(all_triangles))
    end

    len_unit = first(indices).len_unit
    merged = TriangleIndex(T, all_vertices, all_triangles, len_unit)
    return merged, ranges
end

# ---------------------------------------------------------------------------
# Octree entry point
# ---------------------------------------------------------------------------

"""
    TriangleOctree(tri::Triangulation; kwargs...)

Build the geometry-adaptive octree over a triangulation's merged index.

Multi-patch addition: coincident container/obstacle **seams** — an obstacle
resting flush on the container (e.g. an impeller sitting on the tank floor) —
are detected once at build and stored as a [`SeamVeto`](@ref): container
triangles within `seam_tolerance_relative` (of the bbox diagonal) of any
obstacle surface are flagged, and signed-distance queries cross-check them
against the obstacle so the obstacle wins the otherwise-arbitrary tie
(without it, the obstacle interior above a flush seam classifies as fluid).
"""
function TriangleOctree(
        tri::Triangulation{T};
        classify_leaves::Bool = true,
        seam_tolerance_relative = 1.0e-3,
        kwargs...,
    ) where {T}
    # Build the (unclassified) tree first — seam detection queries it.
    oct = TriangleOctree(tri.index; classify_leaves = false, kwargs...)
    veto = _build_seam_veto(tri, oct.tree, T(seam_tolerance_relative))

    classification = if classify_leaves
        _classify_leaves(oct.tree, tri.index; seam_veto = veto)
    else
        nothing
    end
    return TriangleOctree{T}(oct.tree, tri.index, classification, veto)
end

# Coincident-seam detection: for every obstacle triangle, collect the
# CONTAINER triangles in its `tol`-dilated bbox via the tree (a complete
# range query — no sampling gaps) and flag those within `tol`
# (triangle-triangle distance, vertex-based both ways). At a flush contact
# the two surfaces coincide, so the container surface alone classifies the
# obstacle interior above the seam as fluid — the flag tells the query to
# cross-check the obstacle (see `_compute_signed_distance_octree`'s veto).
function _build_seam_veto(tri::Triangulation{T}, tree, tol_rel::T) where {T}
    obstacle_ranges = [r for (r, ro) in zip(tri.ranges, tri.roles) if ro === :obstacle]
    container_ranges = [r for (r, ro) in zip(tri.ranges, tri.roles) if ro === :container]
    (isempty(obstacle_ranges) || isempty(container_ranges)) && return nothing

    index = tri.index
    tol = tol_rel * norm(index.bbox_max - index.bbox_min)

    flags = falses(num_triangles(index))
    candidates = Int[]
    for obs_range in obstacle_ranges, ti in obs_range
        ov = _get_triangle_vertices(index, ti)
        lo = min.(min.(ov[1], ov[2]), ov[3]) .- tol
        hi = max.(max.(ov[1], ov[2]), ov[3]) .+ tol
        empty!(candidates)
        _collect_in_box!(candidates, tree, 1, lo, hi, container_ranges)
        for ct in candidates
            flags[ct] && continue
            cv = _get_triangle_vertices(index, ct)
            if _tri_tri_distance(ov, cv) <= tol
                flags[ct] = true
            end
        end
    end
    any(flags) || return nothing
    return SeamVeto(flags, obstacle_ranges, tol)
end

@inline function _in_any_range(ti, ranges)
    for r in ranges
        ti in r && return true
    end
    return false
end

# Collect leaf elements accepted by `accept` from boxes intersecting [lo, hi].
function _collect_in_box!(out, tree, box_idx, lo, hi, ranges)
    bmin, bmax = box_bounds(tree, box_idx)
    (any(bmax .< lo) || any(bmin .> hi)) && return out
    if is_leaf(tree, box_idx)
        for ti in tree.element_lists[box_idx]
            _in_any_range(ti, ranges) && push!(out, ti)
        end
        return out
    end
    for c in children(tree, box_idx)
        _collect_in_box!(out, tree, c, lo, hi, ranges)
    end
    return out
end

# Vertex-based triangle-triangle distance (both directions): the minimum over
# each triangle's vertices of the point-to-other-triangle distance. Exact when
# the surfaces touch at a vertex or overlap coplanarly — the flush-contact
# cases the seam veto is built for.
function _tri_tri_distance(tv1, tv2)
    d = typemax(eltype(tv1[1]))
    for q in tv1
        cp, _ = closest_point_on_triangle_feature(q, tv2[1], tv2[2], tv2[3])
        d = min(d, norm(q - cp))
    end
    for q in tv2
        cp, _ = closest_point_on_triangle_feature(q, tv1[1], tv1[2], tv1[3])
        d = min(d, norm(q - cp))
    end
    return d
end

function Base.show(io::IO, ::MIME"text/plain", tri::Triangulation{T}) where {T}
    println(io, "Triangulation{$T}")
    println(io, "├─Triangles: $(num_triangles(tri))")
    println(io, "└─Patches ($(npatches(tri))):")
    for (name, rng, r) in zip(tri.names, tri.ranges, tri.roles)
        println(io, "  ├─:$name ($r) — $(length(rng)) triangles")
    end
    return nothing
end
