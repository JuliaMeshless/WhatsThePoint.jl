@testitem "metrics with default k" setup = [TestData, CommonImports] begin
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

    cloud = PointCloud(TestData.BOX_PATH)

    output, result = capture_stdout(() -> metrics(cloud))

    @test output isa String
    @test occursin("Cloud Metrics", output)
    @test occursin("avg. distance to 20 nearest neighbors", output)
    @test occursin("std. distance to 20 nearest neighbors", output)
    @test occursin("max. distance to 20 nearest neighbors", output)
    @test occursin("min. distance to 20 nearest neighbors", output)

    @test result === nothing
end

@testitem "metrics with custom k" setup = [TestData, CommonImports] begin
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

    cloud = PointCloud(TestData.BOX_PATH)

    output, result = capture_stdout(() -> metrics(cloud; k = 10))

    @test output isa String
    @test occursin("Cloud Metrics", output)
    @test occursin("avg. distance to 10 nearest neighbors", output)
    @test occursin("std. distance to 10 nearest neighbors", output)
    @test occursin("max. distance to 10 nearest neighbors", output)
    @test occursin("min. distance to 10 nearest neighbors", output)

    @test result === nothing
end

@testitem "metrics with small k" setup = [TestData, CommonImports] begin
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

    cloud = PointCloud(TestData.BOX_PATH)

    output, result = capture_stdout(() -> metrics(cloud; k = 5))

    @test output isa String
    @test occursin("avg. distance to 5 nearest neighbors", output)
    @test result === nothing
end

@testitem "metrics statistics verification" setup = [TestData, CommonImports] begin
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

    cloud = PointCloud(TestData.BOX_PATH)

    output, _ = capture_stdout(() -> metrics(cloud; k = 10))

    lines = split(output, '\n')
    stats_lines = filter(line -> occursin("distance to", line), lines)

    @test length(stats_lines) == 4

    for line in stats_lines
        @test occursin(r"\d+\.?\d*(?:[eE][+-]?\d+)?", line)
    end
end

@testitem "metrics with minimal cloud" setup = [TestData, CommonImports] begin
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

    points = Point.([(i * 1.0, 0.0, 0.0) for i = 1:25])
    small_cloud = PointCloud(PointBoundary(points))

    output, result = capture_stdout(() -> metrics(small_cloud; k = 10))
    @test occursin("Cloud Metrics", output)
    @test result === nothing

    output2, result2 = capture_stdout(() -> metrics(small_cloud; k = 20))
    @test occursin("Cloud Metrics", output2)
    @test result2 === nothing
end
