"""
    abstract type AbstractSurface{M<:Manifold,C<:CRS} end

A surface of a [PointSurface](@ref).
"""
abstract type AbstractSurface{M <: Manifold, C <: CRS} <: Domain{M, C} end

"""
    struct SurfaceElement{M,C,N,A}

Representation of a point on a `<:PointSurface`.
"""
struct SurfaceElement{M, C, N, A} <: Geometry{M, C}
    point::Point{M, C}
    normal::N
    area::A
end

Meshes.crs(::Type{<:SurfaceElement{M, C}}) where {M, C} = C
Meshes.crs(se::SurfaceElement) = crs(se.point)

"""
    struct PointSurface{M,C,S,T} <: AbstractSurface{M,C}

This is a typical representation of a surface via points.

# Type Parameters
- `M<:Manifold` - manifold type
- `C<:CRS` - coordinate reference system
- `S` - shadow type
- `T<:AbstractTopology` - topology type for surface-local connectivity
"""
struct PointSurface{M <: Manifold, C <: CRS, S, T <: AbstractTopology} <: AbstractSurface{M, C}
    geoms::StructVector{SurfaceElement}
    shadow::S
    topology::T
    function PointSurface(
            geoms::StructVector{SurfaceElement},
            shadow::S = nothing,
            topology::T = NoTopology(),
        ) where {S, T <: AbstractTopology}
        p = first(geoms.point)
        M = manifold(p)
        C = crs(p)
        return new{M, C, S, T}(geoms, shadow, topology)
    end
end

function PointSurface(
        points::AbstractVector{Point{M, C}},
        normals::N,
        areas::A;
        shadow::S = nothing,
        topology::T = NoTopology(),
    ) where {M <: Manifold, C <: CRS, N, A, S, T <: AbstractTopology}
    @assert length(points) == length(normals) == length(areas) "All inputs must be same length. Got $(length(points)), $(length(normals)), $(length(areas))."
    geoms = StructArray{SurfaceElement}((points, normals, areas))
    return PointSurface(geoms, shadow, topology)
end

function PointSurface(
        pts::Domain,
        normals,
        areas;
        shadow = nothing,
        topology = NoTopology(),
    )
    p = _get_underlying_vector(pts)
    return PointSurface(p, normals, areas; shadow = shadow, topology = topology)
end
_get_underlying_vector(pts::SubDomain) = pts.domain[pts.inds]
_get_underlying_vector(pts::AbstractVector) = pts

function PointSurface(points::AbstractVector, normals; topology = NoTopology())
    T = CoordRefSystems.mactype(crs(first(points)))
    # TODO estimate areas
    areas = zeros(T, length(points))
    return PointSurface(points, normals, areas; topology = topology)
end

function PointSurface(points::AbstractVector; k::Int = 5, topology = NoTopology())
    normals = compute_normals(points; k = k)
    T = CoordRefSystems.mactype(crs(first(points)))
    # TODO estimate areas
    areas = zeros(T, length(normals))
    surf = PointSurface(points, normals, areas; topology = topology)
    orient_normals!(surf; k = k)
    return surf
end

function PointSurface(filepath::String; topology = NoTopology())
    points, normals, areas, _ = import_surface(filepath)
    return PointSurface(points, normals, areas; topology = topology)
end

function (s::PointSurface)(shadow::ShadowPoints)
    # Shadow creation strips topology (shadows have no inherited connectivity)
    return PointSurface(point(s), normal(s), area(s); shadow = shadow)
end

Base.parent(surf::PointSurface) = surf.geoms
Base.firstindex(::PointSurface) = 1
Base.lastindex(surf::PointSurface) = length(surf)
Base.getindex(surf::PointSurface, index::Int) = getindex(parent(surf), index)
function Base.iterate(surf::PointSurface, state = 1)
    state > length(surf) && return nothing
    return (surf[state], state + 1)
end

"""
    points(surf::PointSurface)

Return vector of points from surface. Alias for `point(surf)`.
"""
points(surf::PointSurface) = point(surf)
Meshes.elements(surf::PointSurface) = (elem for elem in parent(surf))
Meshes.nelements(surf::PointSurface) = length(parent(surf))
Meshes.centroid(surf::PointSurface) = centroid(point(surf))
Meshes.boundingbox(surf::PointSurface) = boundingbox(point(surf))

ChunkSplitters.is_chunkable(::PointSurface) = true

Base.view(surf::PointSurface, range::UnitRange) = view(parent(surf), range)
Base.view(surf::PointSurface, range::StepRange) = view(parent(surf), range)

to(surf::PointSurface) = to.(parent(surf).point)
point(surf::PointSurface) = parent(surf).point
normal(surf::PointSurface) = parent(surf).normal
area(surf::PointSurface) = parent(surf).area

# Topology accessors
"""
    topology(surf::PointSurface)

Return the topology of the surface.
"""
topology(surf::PointSurface) = surf.topology

"""
    hastopology(surf::PointSurface)

Check if surface has a topology (not NoTopology).
"""
hastopology(surf::PointSurface) = !isa(topology(surf), NoTopology)

"""
    neighbors(surf::PointSurface)

Return all neighbor lists from the surface topology. Throws error if no topology.
"""
neighbors(surf::PointSurface) = neighbors(topology(surf))

"""
    neighbors(surf::PointSurface, i::Int)

Return neighbors of point `i` in surface-local indices. Throws error if no topology.
"""
neighbors(surf::PointSurface, i::Int) = neighbors(topology(surf), i)

"""
    set_topology(surf::PointSurface, ::Type{KNNTopology}, k::Int)

Build and return new surface with k-nearest neighbor topology.
"""
function set_topology(surf::PointSurface, ::Type{KNNTopology}, k::Int)
    pts = points(surf)
    adj = _build_knn_neighbors(pts, k)
    topo = KNNTopology(adj, k)
    return PointSurface(
        point(surf),
        normal(surf),
        area(surf);
        shadow = surf.shadow,
        topology = topo,
    )
end

"""
    set_topology(surf::PointSurface, ::Type{RadiusTopology}, radius)

Build and return new surface with radius-based topology.
"""
function set_topology(surf::PointSurface, ::Type{RadiusTopology}, radius)
    pts = points(surf)
    adj = _build_radius_neighbors(pts, radius)
    topo = RadiusTopology(adj, radius)
    return PointSurface(
        point(surf),
        normal(surf),
        area(surf);
        shadow = surf.shadow,
        topology = topo,
    )
end

"""
    rebuild_topology!(surf::PointSurface)

Rebuild topology in place using same parameters. No-op if NoTopology.
"""
function rebuild_topology!(surf::PointSurface)
    pts = points(surf)
    rebuild_topology!(topology(surf), pts)
    return nothing
end

function generate_shadows(surf::PointSurface, shadow::ShadowPoints)
    return generate_shadows(to(surf), normal(surf), shadow)
end

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", surf::PointSurface{M, C}) where {M, C}
    println(io, "PointSurface{$M,$C}")
    println(io, "├─Number of points: $(length(surf))")

    a = area(surf)
    if isnothing(a)
        println(io, "├─Area: not defined")
    else
        println(io, "├─Area: $(sum(a))")
    end

    s = surf.shadow
    _shadow_order(::ShadowPoints{O}) where {O} = O
    if isnothing(s)
        println(io, "├─Shadow: none")
    else
        println(io, "├─Shadow: $(_shadow_order(s))")
    end

    topo = topology(surf)
    topo_name = nameof(typeof(topo))
    println(io, "└─Topology: $topo_name")
    return nothing
end

Base.show(io::IO, ::PointSurface) = println(io, "PointSurface")
