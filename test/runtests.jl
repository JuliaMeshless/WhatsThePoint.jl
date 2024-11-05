using SafeTestsets

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

@safetestset "PointPart" begin
    include("part.jl")
end

@safetestset "isinside" begin
    include("isinside.jl")
end
