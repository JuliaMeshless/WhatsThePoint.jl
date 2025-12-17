
"""
    import_surface(filepath::String)

Import a surface mesh. Re-uses code from MeshBridge.jl, did not use their package because I needed to calculate face centers which they do not do.
"""
function import_surface(filepath::String)
    geo = GeoIO.load(filepath)
    mesh = geo.geometry
    points = map(centroid, elements(mesh))
    n = try
        geo.normal
    catch
        compute_normals(points)
    end
    normals = map(x -> ustrip.(x / norm(x)), n)
    area = map(Meshes.area, elements(mesh))
    return points, normals, area, mesh
end

function export_cloud(filename::String, cloud::PointCloud)
    exportvtk(filename, to(boundary(cloud)), [normal(cloud)], ["normals"])
    return nothing
end

function exportvtk(
    filename::String,
    points::AbstractVector{V},
    data::AbstractVector,
    names::Vector;
    triangulate=false,
) where {V<:AbstractVector}
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

function createvtkcells(coords, triangulate=true, nonconvex=false)
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

function FileIO.save(filename::String, cloud::PointCloud)
    return save(filename, LittleDict("cloud" => cloud))
end
