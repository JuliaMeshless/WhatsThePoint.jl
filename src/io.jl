"""
    import_surface(filepath::String)

Load a surface mesh from a file (STL, OBJ, or any format supported by GeoIO.jl). Returns a
tuple of `(points, normals, areas, mesh)` where points are face centers.
"""
function import_surface(filepath::String)
    geo = GeoIO.load(filepath)
    mesh = geo.geometry
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
        _save_vtk_cloud(filename, cloud)
    else
        throw(ArgumentError("unsupported format: $format. Use :jld2 or :vtk."))
    end
    return nothing
end

function _save_vtk_cloud(filename::String, cloud::PointCloud)
    bnd_pts = to(boundary(cloud))
    vol_pts = length(volume(cloud)) > 0 ? to(volume(cloud)) : eltype(bnd_pts)[]
    all_pts = vcat(bnd_pts, vol_pts)

    nbnd = length(bnd_pts)
    nvol = length(vol_pts)

    # type indicator: 1 = boundary, 2 = volume
    point_type = vcat(ones(Int, nbnd), fill(2, nvol))

    # normals: boundary has normals, volume gets zeros
    bnd_normals = normal(boundary(cloud))
    zero_normal = zero(first(bnd_normals))
    vol_normals = fill(zero_normal, nvol)
    all_normals = vcat(bnd_normals, vol_normals)

    exportvtk(
        filename, all_pts,
        [all_normals, point_type],
        ["normals", "point_type"],
    )
    return nothing
end

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
