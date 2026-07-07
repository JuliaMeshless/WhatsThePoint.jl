# Mesh files carry no unit metadata and GeoIO assigns meters by default, so the
# stored numbers are reinterpreted in the user-supplied unit (46 -> 46 mm, no
# conversion). Raw coordinates never change — downstream numerics are label-blind.
function _reinterpret_unit(mesh::SimpleMesh, unit::Unitful.Units)
    dimension(unit) == u"𝐋" || throw(ArgumentError("unit must be a length unit, got $unit"))
    C = crs(mesh)
    C <: Cartesian || throw(ArgumentError("unit reinterpretation requires Cartesian coordinates, got $C"))
    verts = map(Meshes.vertices(mesh)) do p
        Point((CoordRefSystems.raw(coords(p)) .* unit)...)
    end
    return SimpleMesh(verts, Meshes.topology(mesh))
end

"""
    import_mesh(filepath, unit::Unitful.Units) -> SimpleMesh

Load a surface mesh from a file (STL, OBJ, or any format supported by GeoIO.jl)
and *reinterpret* its raw coordinates in `unit`: a stored `46` becomes `46 mm`
with `unit = u"mm"` — no conversion happens, because mesh files carry no unit
metadata (GeoIO's default of meters is discarded). Topology and coordinate
machine type (`Float32` stays `Float32`) are preserved.

This is the single gateway for file geometry: the returned mesh feeds
[`PointBoundary`](@ref), [`TriangleOctree`](@ref), and [`Octree`](@ref).
Use [`geometry_info`](@ref) to probe the raw extents when unsure of the unit.
"""
function import_mesh(filepath::AbstractString, unit::Unitful.Units)
    geom = GeoIO.load(filepath).geometry
    geom isa SimpleMesh ||
        throw(ArgumentError("$filepath did not load as a SimpleMesh, got $(nameof(typeof(geom)))"))
    return _reinterpret_unit(geom, unit)
end

"""
    import_surface(filepath, unit::Unitful.Units)

Load a surface mesh from a file (STL, OBJ, or any format supported by GeoIO.jl). Returns a
tuple of `(points, normals, areas, mesh)` where points are face centers. Coordinates are
reinterpreted in `unit` (see [`import_mesh`](@ref)); areas therefore carry `unit^2`.
"""
function import_surface(filepath::AbstractString, unit::Unitful.Units)
    geo = GeoIO.load(filepath)
    geo.geometry isa SimpleMesh ||
        throw(ArgumentError("$filepath did not load as a SimpleMesh, got $(nameof(typeof(geo.geometry)))"))
    mesh = _reinterpret_unit(geo.geometry, unit)
    points = map(centroid, elements(mesh))
    n = if hasproperty(geo, :normal)
        geo.normal
    else
        @info "No normals found in file, computing via PCA"
        compute_normals(points)
    end
    normals = map(x -> ustrip.(x / norm(x)), n)
    area = map(Meshes.area, elements(mesh))
    return points, normals, area, mesh
end

"""
    save(filename::String, cloud::PointCloud; format=:jld2)

Save a point cloud to a file.

- `format=:jld2` (default): Serialize via FileIO.jl.
- `format=:vtk`: Export to VTK format with boundary and volume points, normals, and a
  point type indicator (`1` = boundary, `2` = volume).
"""
function FileIO.save(filename::String, cloud::PointCloud; format::Symbol = :jld2)
    if format === :jld2
        return FileIO.save(filename, LittleDict("cloud" => cloud))
    elseif format === :vtk
        export_vtk(filename, cloud)
    else
        throw(ArgumentError("unsupported format: $format. Use :jld2 or :vtk."))
    end
    return nothing
end

"""
    export_vtk(filename, cloud::PointCloud; fields=(), verbose=false)

Write `cloud` to a ParaView-ready `.vtu` (one `VTK_VERTEX` cell per point). Open
it in ParaView and set *Representation* to *Point Gaussian* (or *Points*).

Always-attached point data:
- `point_type` — `1` = boundary, `2` = volume (colour to separate wall from bulk).
- `surface_id` — `1..N` in `names(cloud)` order, `0` for volume (colour by named
  surface). Pass `verbose=true` to print the integer→name legend.
- `normals` — boundary normals (zero on volume points).

`fields` attaches solution data, so a `.vtu` can be re-exported after solving and
viewed like any CAE result. It is an iterable of `name => values` pairs (e.g. a
`Dict` or a tuple of pairs); `values` may be scalars or per-point vectors and
**must be ordered like [`points`](@ref)`(cloud)` — boundary points first, then
volume** (the natural global DOF order). Units are stripped automatically.

# Examples
```julia
export_vtk("cloud", cloud)                                 # geometry only
export_vtk("sol", cloud; fields = ("T" => temp, "U" => velocity))
```
"""
function export_vtk(filename::String, cloud::PointCloud; fields = (), verbose::Bool = false)
    bnd_pts = to(boundary(cloud))
    vol_pts = length(volume(cloud)) > 0 ? to(volume(cloud)) : eltype(bnd_pts)[]
    all_pts = vcat(bnd_pts, vol_pts)

    nbnd = length(bnd_pts)
    nvol = length(vol_pts)
    npts = nbnd + nvol

    # type indicator: 1 = boundary, 2 = volume
    point_type = vcat(ones(Int, nbnd), fill(2, nvol))

    # per-point surface id: 1..N in names(cloud) order (matches the boundary
    # point concatenation), 0 for volume points.
    surface_id = zeros(Int, npts)
    offset = 0
    for (i, (name, surf)) in enumerate(namedsurfaces(boundary(cloud)))
        ns = length(surf)
        surface_id[(offset + 1):(offset + ns)] .= i
        offset += ns
        verbose && println("surface_id $i -> $name")
    end

    # normals: boundary has normals, volume gets zeros
    bnd_normals = normal(boundary(cloud))
    zero_normal = zero(first(bnd_normals))
    all_normals = vcat(bnd_normals, fill(zero_normal, nvol))

    # Accept a single bare `name => values` pair as well as an iterable of pairs.
    field_pairs = fields isa Pair ? (fields,) : fields
    for (name, vals) in field_pairs
        length(vals) == npts || throw(
            DimensionMismatch(
                "solution field \"$name\" has $(length(vals)) values but the cloud has " *
                    "$npts points; fields must be ordered like points(cloud) (boundary then volume)",
            ),
        )
    end
    data = [all_normals, point_type, surface_id, (_strip_field(vals) for (_, vals) in field_pairs)...]
    names = ["normals", "point_type", "surface_id", (String(name) for (name, _) in field_pairs)...]

    exportvtk(filename, all_pts, data, names)
    return nothing
end

# Strip units for VTK: scalar fields element-wise, vector-valued fields per entry.
_strip_field(v::AbstractVector{<:Number}) = ustrip.(v)
_strip_field(v::AbstractVector) = [ustrip.(x) for x in v]

"""
    save(filename::String, boundary::PointBoundary; format=:jld2)

Save a boundary to a file.

- `format=:jld2` (default): Serialize via FileIO.jl.
- `format=:vtk`: Export to VTK format with boundary points and normals.
"""
function FileIO.save(filename::String, bnd::PointBoundary; format::Symbol = :jld2)
    if format === :jld2
        return FileIO.save(filename, LittleDict("boundary" => bnd))
    elseif format === :vtk
        exportvtk(filename, to(bnd), [normal(bnd)], ["normals"])
    else
        throw(ArgumentError("unsupported format: $format. Use :jld2 or :vtk."))
    end
    return nothing
end

"""
    save(filename::String, surf::PointSurface; format=:jld2)

Save a surface to a file.

- `format=:jld2` (default): Serialize via FileIO.jl.
- `format=:vtk`: Export to VTK format with surface points, normals, and areas.
"""
function FileIO.save(filename::String, surf::PointSurface; format::Symbol = :jld2)
    if format === :jld2
        return FileIO.save(filename, LittleDict("surface" => surf))
    elseif format === :vtk
        a = area(surf)
        data = isnothing(a) ? [normal(surf)] : [normal(surf), ustrip.(a)]
        names = isnothing(a) ? ["normals"] : ["normals", "areas"]
        exportvtk(filename, to(surf), data, names)
    else
        throw(ArgumentError("unsupported format: $format. Use :jld2 or :vtk."))
    end
    return nothing
end

function exportvtk(
        filename::String,
        points::AbstractVector{V},
        data::AbstractVector,
        names::Vector;
        triangulate = false,
    ) where {V <: AbstractVector}
    # Strip units from points for VTK compatibility
    p = reduce(hcat, map(pt -> ustrip.(pt), points))
    cells = createvtkcells(p, triangulate)
    vtkfile = createvtkfile(filename, p, cells)
    for (name, field) in zip(names, data)
        addfieldvtk!(vtkfile, name, field)
    end
    savevtk!(vtkfile)
    return nothing
end

function createvtkcells(coords, triangulate = true, nonconvex = false)
    # only save as points/vertexes
    return [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:size(coords, 2)]
end

function createvtkfile(filename::String, coords, cells)
    return vtk_grid(filename, coords, cells)
end

function addfieldvtk!(vtkfile, scalarname::String, data)
    # Convert vector of SVectors to matrix for WriteVTK
    data = _hcat_data(data)
    return vtkfile[scalarname, VTKPointData()] = data
end

_hcat_data(data) = data
_hcat_data(data::AbstractVector) = reduce(hcat, data)

function savevtk!(vtkfile)
    return vtk_save(vtkfile)
end

function _print_geometry_entry(name, bmin, bmax, ext)
    rnd(t) = round.(t; sigdigits = 4)
    println("─── $name ───")
    println("  min:    $(rnd(bmin))")
    println("  max:    $(rnd(bmax))")
    println("  extent: $(rnd(ext))")
    return nothing
end

"""
    geometry_info(filepath, filepaths...; verbose=true) -> Vector{NamedTuple}

Inspect the raw bounding box of one or more mesh files before constructing
anything — the "what do these numbers mean?" probe for files without unit
metadata (STL coordinates are just numbers; GeoIO.jl assigns meters by
default). Returns one `(file, min, max, extent)` named tuple per file, with
values as raw (unitless) coordinate tuples. With `verbose=true` prints each
bounding box and, for multiple files, their union — useful when parts must fit
together.

Once you know the unit, pass it to [`import_mesh`](@ref) or
[`PointBoundary`](@ref) — the raw numbers are reinterpreted in that unit.

# Example
```julia
geometry_info("intake.stl", "exhaust.stl")
# ─── intake.stl ───
#   min:    (0.0, 0.0, 0.0)
#   max:    (120.5, 87.3, 42.0)
#   extent: (120.5, 87.3, 42.0)
# ─── exhaust.stl ───
#   ...
# ─── Union ───
#   ...
```
"""
function geometry_info(filepath::AbstractString, filepaths::AbstractString...; verbose::Bool = true)
    entries = map((filepath, filepaths...)) do f
        box = boundingbox(Meshes.vertices(GeoIO.load(f).geometry))
        bmin = Tuple(ustrip.(to(minimum(box))))
        bmax = Tuple(ustrip.(to(maximum(box))))
        (file = f, min = bmin, max = bmax, extent = bmax .- bmin)
    end
    if verbose
        for e in entries
            _print_geometry_entry(basename(e.file), e.min, e.max, e.extent)
        end
        if length(entries) > 1
            umin = reduce((a, b) -> min.(a, b), (e.min for e in entries))
            umax = reduce((a, b) -> max.(a, b), (e.max for e in entries))
            _print_geometry_entry("Union", umin, umax, umax .- umin)
        end
    end
    return collect(entries)
end
