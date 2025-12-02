using WhatsThePoint
using Meshes
using Unitful: m

# Helper function to capture stdout
function capture_stdout(f)
    old_stdout = stdout
    rd, wr = redirect_stdout()
    result = f()
    redirect_stdout(old_stdout)
    close(wr)
    output = read(rd, String)
    close(rd)
    return output, result
end

@testset "metrics function" begin
    # Create a simple point cloud from test data
    cloud = PointCloud(joinpath(@__DIR__, "data", "box.stl"))

    @testset "metrics with default k" begin
        # Test that metrics runs without error with default k=20
        output, result = capture_stdout(() -> metrics(cloud))

        @test output isa String
        @test occursin("Cloud Metrics", output)
        @test occursin("avg. distance to 20 nearest neighbors", output)
        @test occursin("std. distance to 20 nearest neighbors", output)
        @test occursin("max. distance to 20 nearest neighbors", output)
        @test occursin("min. distance to 20 nearest neighbors", output)

        # Verify the function returns nothing
        @test result === nothing
    end

    @testset "metrics with custom k" begin
        # Test with k=10
        output, result = capture_stdout(() -> metrics(cloud; k=10))

        @test output isa String
        @test occursin("Cloud Metrics", output)
        @test occursin("avg. distance to 10 nearest neighbors", output)
        @test occursin("std. distance to 10 nearest neighbors", output)
        @test occursin("max. distance to 10 nearest neighbors", output)
        @test occursin("min. distance to 10 nearest neighbors", output)

        @test result === nothing
    end

    @testset "metrics with small k" begin
        # Test with k=5 to ensure it works with smaller neighborhood
        output, result = capture_stdout(() -> metrics(cloud; k=5))

        @test output isa String
        @test occursin("avg. distance to 5 nearest neighbors", output)
        @test result === nothing
    end

    @testset "verify statistics are reasonable" begin
        # Capture output and parse to verify numerical values are reasonable
        output, _ = capture_stdout(() -> metrics(cloud; k=10))

        # Extract the lines containing numerical statistics
        lines = split(output, '\n')
        stats_lines = filter(line -> occursin("distance to", line), lines)

        # Should have 4 statistics (avg, std, max, min)
        @test length(stats_lines) == 4

        # Verify each line contains a numerical value (matches pattern like "1.23" or "1.23e-4")
        for line in stats_lines
            @test occursin(r"\d+\.?\d*(?:[eE][+-]?\d+)?", line)
        end
    end

    @testset "metrics with minimal cloud" begin
        # Test with a small manually created cloud
        points = Point.([(i * 1.0, 0.0, 0.0) for i in 1:25])
        small_cloud = PointCloud(PointBoundary(points))

        # Should work with k smaller than total points
        output, result = capture_stdout(() -> metrics(small_cloud; k=10))
        @test occursin("Cloud Metrics", output)
        @test result === nothing

        # Should work with k close to total points
        output2, result2 = capture_stdout(() -> metrics(small_cloud; k=20))
        @test occursin("Cloud Metrics", output2)
        @test result2 === nothing
    end
end
