
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
    normals = map(x -> ustrip(x / norm(x)), n)
    area = map(Meshes.area, elements(mesh))
    return points, normals, area, mesh
end

function export_cloud(filename::String, cloud::PointCloud)
    exportvtk(filename, to(boundary(cloud)), [normals(cloud)], ["normals"])
    return nothing
end

function exportvtk(
    filename::String,
    points::AbstractVector{V},
    data::AbstractVector,
    names::Vector;
    triangulate=false,
) where {V<:AbstractVector}
    p = reduce(hcat, points)
    cells = createvtkcells(p, triangulate)
    vtkfile = createvtkfile(filename, p, cells)
    for (name, field) in zip(names, data)
        addfieldvtk!(vtkfile, name, field)
    end
    savevtk!(vtkfile)
    return nothing
end

function createvtkcells(coords, triangulate=true, nonconvex=false)
    if triangulate
        # compute delaunay triangulation so you can view in paraview as a surface
        p = Matrix(transpose(coords))
        conn = convert(Matrix{Int32}, delaunay(p).simplices)

        if nonconvex
            keep = Int32[]
            for i in axes(conn, 1)
                center = [mean(coords[1, conn[i, :]]), mean(coords[2, conn[i, :]])]
                if isinside2d(center, pointsboundary)
                    push!(keep, i)
                end
            end
            conn2 = zeros(Int32, 3, size(keep, 1))
            for i in axes(keep, 1)
                conn2[:, i] = conn[keep[i], :]
            end
            cells = MeshCell{VTKCellType,Vector{Int32}}[]
            for i in axes(conn2, 2)
                push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, conn2[:, i]))
            end
            return cells
        end

        cells = MeshCell{VTKCellType,Vector{Int32}}[]
        for i in axes(conn, 1)
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, conn[i, :]))
        end
        return cells
    else
        # only save as points/vertexes
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:size(coords, 2)]
        return cells
    end
end

function createvtkfile(filename::String, coords, cells)
    return vtk_grid(filename, coords, cells)
end

function createmultiblockvtk(filename::String)
    return vtk_multiblock(filename)
end

function addgrid!(vtmfile, grid)
    return vtk_grid(vtmfile, grid)
end

function addfieldvtk!(vtkfile, scalarname::String, data)
    return vtkfile[scalarname, VTKPointData()] = data
end

function savevtk!(vtkfile)
    return vtk_save(vtkfile)
end

FileIO.save(filename::String, cloud::PointCloud) = save(filename, Dict("cloud" => cloud))
