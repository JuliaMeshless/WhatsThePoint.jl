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

    @test result isa NamedTuple
    @test haskey(result, :avg)
    @test haskey(result, :std)
    @test haskey(result, :max)
    @test haskey(result, :min)
    @test haskey(result, :k)
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

    @test result isa NamedTuple
    @test haskey(result, :avg)
    @test haskey(result, :std)
    @test haskey(result, :max)
    @test haskey(result, :min)
    @test haskey(result, :k)
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
    @test result isa NamedTuple
    @test haskey(result, :avg)
    @test haskey(result, :std)
    @test haskey(result, :max)
    @test haskey(result, :min)
    @test haskey(result, :k)
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

    points = Point.([(i * 1.0, 0.0, 0.0) for i in 1:25])
    small_cloud = PointCloud(PointBoundary(points))

    output, result = capture_stdout(() -> metrics(small_cloud; k = 10))
    @test occursin("Cloud Metrics", output)
    @test result isa NamedTuple
    @test haskey(result, :avg)
    @test haskey(result, :std)
    @test haskey(result, :max)
    @test haskey(result, :min)
    @test haskey(result, :k)

    output2, result2 = capture_stdout(() -> metrics(small_cloud; k = 20))
    @test occursin("Cloud Metrics", output2)
    @test result2 isa NamedTuple
    @test result2.k == 20
end

@testitem "spacing_metrics return shape" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    result = spacing_metrics(cloud, spacing)

    @test result isa NamedTuple
    @test haskey(result, :max_error)
    @test haskey(result, :mean_error)
    @test haskey(result, :std_error)
    @test haskey(result, :k)
    @test result.k == 20
    @test isfinite(result.max_error)
    @test isfinite(result.mean_error)
    @test isfinite(result.std_error)
    @test result.max_error >= result.mean_error
    @test result.mean_error >= 0
end

@testitem "spacing_metrics custom k" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    for k in (5, 10, 20)
        result = spacing_metrics(cloud, spacing; k = k)
        @test result.k == k
        @test isfinite(result.mean_error)
    end
end

@testitem "spacing_metrics penalizes mismatched target" setup = [TestData, CommonImports] begin
    using Unitful: m

    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing_correct = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing_correct; alg = SlakKosec(octree), max_points = 50)

    spacing_too_large = ConstantSpacing(100 * spacing_correct.Δx)
    spacing_too_small = ConstantSpacing(spacing_correct.Δx / 100)

    err_good = spacing_metrics(cloud, spacing_correct)
    err_large = spacing_metrics(cloud, spacing_too_large)
    err_small = spacing_metrics(cloud, spacing_too_small)

    @test err_large.mean_error > err_good.mean_error
    @test err_small.mean_error > err_good.mean_error
end

@testitem "spacing_metrics tracks repel" setup = [TestData, CommonImports] begin
    boundary = PointBoundary(TestData.BOX_PATH)
    octree = TriangleOctree(TestData.BOX_PATH; classify_leaves = true)
    spacing = _relative_spacing(boundary)
    cloud = discretize(boundary, spacing; alg = SlakKosec(octree), max_points = 50)

    pre = spacing_metrics(cloud, spacing)

    new_cloud = repel(cloud, spacing, octree; max_iters = 20)
    post = spacing_metrics(new_cloud, spacing)

    @test isfinite(pre.mean_error)
    @test isfinite(post.mean_error)
    # Sanity only — exact preservation is future work. Confirms repel doesn't
    # catastrophically destroy the spacing relationship.
    @test post.mean_error < 10 * pre.mean_error + 1
end
