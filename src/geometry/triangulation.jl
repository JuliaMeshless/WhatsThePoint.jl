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
"""
function TriangleOctree(tri::Triangulation; kwargs...)
    return TriangleOctree(tri.index; kwargs...)
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
