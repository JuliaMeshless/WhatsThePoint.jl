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

@safetestset "IO" begin
    include("io.jl")
end
