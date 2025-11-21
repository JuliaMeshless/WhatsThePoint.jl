using SafeTestsets

@safetestset "Utils" begin
    include("utils.jl")
end

@safetestset "Points" begin
    include("points.jl")
end

@safetestset "Normals" begin
    include("normals.jl")
end

@safetestset "PointSurface" begin
    include("surface.jl")
end

@safetestset "PointVolume" begin
    include("volume.jl")
end

@safetestset "PointCloud" begin
    include("cloud.jl")
end

@safetestset "PointBoundary" begin
    include("boundary.jl")
end

@safetestset "isinside" begin
    include("isinside.jl")
end

@safetestset "ShadowPoints" begin
    include("shadow.jl")
end

@safetestset "Discretization" begin
    include("discretization.jl")
end

@safetestset "Repel" begin
    include("repel.jl")
end

@safetestset "Neighbors" begin
    include("neighbors.jl")
end

@safetestset "Metrics" begin
    include("metrics.jl")
end

@safetestset "SurfaceOperations" begin
    include("surface_operations.jl")
end

@safetestset "IO" begin
    include("io.jl")
end
