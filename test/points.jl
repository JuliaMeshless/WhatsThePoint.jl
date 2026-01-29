@testitem "emptyspace" setup = [TestData, CommonImports] begin
    square2D = Point.([(0, 0), (1, 1), (0, 1), (1, 0)])
    square3D = Point.([
        (0, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (1, 0, 1),
    ])

    @test emptyspace(Point(0.5, 0.5), square2D, eps() * m) == true
    @test emptyspace(Point(0.5, 0.5), square2D, 1m) == false
    @test emptyspace(square2D, Point(0.5, 0.5), 1m) == false
end
