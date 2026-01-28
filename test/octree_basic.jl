# Basic tests for generic spatial octree implementation

using WhatsThePoint
using StaticArrays
using Test

@testset "SpatialOctree Construction" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    @test octree.num_boxes[] == 1
    @test octree.coords[1] == SVector(0, 0, 0, 1)
    @test octree.parent[1] == 0
    @test is_leaf(octree, 1)
    @test !has_children(octree, 1)
end

@testset "Box Geometry" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    center = box_center(octree, 1)
    @test center ≈ SVector(5.0, 5.0, 5.0)

    @test box_size(octree, 1) == 10.0

    min_corner, max_corner = box_bounds(octree, 1)
    @test min_corner ≈ SVector(0.0, 0.0, 0.0)
    @test max_corner ≈ SVector(10.0, 10.0, 10.0)
end

@testset "Subdivision" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    children = subdivide!(octree, 1)

    @test length(children) == 8
    @test octree.num_boxes[] == 9
    @test !is_leaf(octree, 1)
    @test has_children(octree, 1)
    @test all(is_leaf(octree, c) for c in children)

    # Check child coordinates
    @test octree.coords[children[1]] == SVector(0, 0, 0, 2)  # (0,0,0) at level 2
    @test octree.coords[children[8]] == SVector(1, 1, 1, 2)  # (1,1,1) at level 2

    # Check child sizes
    @test all(box_size(octree, c) ≈ 5.0 for c in children)

    # Check child centers
    center_first = box_center(octree, children[1])
    @test center_first ≈ SVector(2.5, 2.5, 2.5)

    center_last = box_center(octree, children[8])
    @test center_last ≈ SVector(7.5, 7.5, 7.5)
end

@testset "Point Query" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Before subdivision, all points map to root
    point = SVector(2.0, 2.0, 2.0)
    leaf = find_leaf(octree, point)
    @test leaf == 1

    # After subdivision
    children = subdivide!(octree, 1)

    # Query point in first octant (0,0,0)
    point = SVector(2.0, 2.0, 2.0)
    leaf = find_leaf(octree, point)
    @test octree.coords[leaf] == SVector(0, 0, 0, 2)

    # Query point in last octant (1,1,1)
    point = SVector(8.0, 8.0, 8.0)
    leaf = find_leaf(octree, point)
    @test octree.coords[leaf] == SVector(1, 1, 1, 2)

    # Query point on boundary (should go to one side consistently)
    point = SVector(5.0, 5.0, 5.0)
    leaf = find_leaf(octree, point)
    @test is_leaf(octree, leaf)
end

@testset "Neighbor Finding" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    children = subdivide!(octree, 1)

    # First child (0,0,0) at level 2
    first_child = children[1]

    # Neighbor in +x direction should be (1,0,0)
    neighbors_x = find_neighbor(octree, first_child, 2)  # +x direction
    @test length(neighbors_x) == 1
    @test octree.coords[neighbors_x[1]] == SVector(1, 0, 0, 2)

    # Neighbor in +y direction should be (0,1,0)
    neighbors_y = find_neighbor(octree, first_child, 4)  # +y direction
    @test length(neighbors_y) == 1
    @test octree.coords[neighbors_y[1]] == SVector(0, 1, 0, 2)

    # Neighbor in -x direction should be empty (boundary)
    neighbors_boundary = find_neighbor(octree, first_child, 1)  # -x direction
    @test isempty(neighbors_boundary)
end

@testset "Element Storage" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Add elements to root
    push!(octree.element_lists[1], 1)
    push!(octree.element_lists[1], 2)
    push!(octree.element_lists[1], 3)

    @test length(octree.element_lists[1]) == 3
    @test num_elements(octree) == 3

    # Subdivide and distribute elements
    children = subdivide!(octree, 1)

    # Manually distribute elements to children
    push!(octree.element_lists[children[1]], 1)
    push!(octree.element_lists[children[2]], 2)
    push!(octree.element_lists[children[3]], 3)

    # Clear parent
    empty!(octree.element_lists[1])

    @test num_elements(octree) == 3
    @test length(octree.element_lists[1]) == 0
end

@testset "Subdivision Criteria" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Add elements to root
    for i in 1:60
        push!(octree.element_lists[1], i)
    end

    # MaxElementsCriterion
    criterion = MaxElementsCriterion(50)
    @test should_subdivide(criterion, octree, 1) == true

    criterion2 = MaxElementsCriterion(100)
    @test should_subdivide(criterion2, octree, 1) == false

    # SizeCriterion
    criterion3 = SizeCriterion(15.0)
    @test should_subdivide(criterion3, octree, 1) == false

    criterion4 = SizeCriterion(5.0)
    @test should_subdivide(criterion4, octree, 1) == true

    # AndCriterion
    criterion5 = AndCriterion((MaxElementsCriterion(50), SizeCriterion(5.0)))
    @test should_subdivide(criterion5, octree, 1) == true

    criterion6 = AndCriterion((MaxElementsCriterion(100), SizeCriterion(5.0)))
    @test should_subdivide(criterion6, octree, 1) == false
end

@testset "Leaf Iteration" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Test 1: Root is a leaf
    leaves = all_leaves(octree)
    @test length(leaves) == 1
    @test leaves[1] == 1
    @test is_leaf(octree, leaves[1])

    # Test 2: After subdivision, 8 leaves
    subdivide!(octree, 1)
    leaves = all_leaves(octree)
    @test length(leaves) == 8
    @test all(is_leaf(octree, idx) for idx in leaves)

    # Test 3: After partial subdivision, mixed
    subdivide!(octree, leaves[1])  # Subdivide first leaf
    leaves = all_leaves(octree)
    @test length(leaves) == 15  # 7 original + 8 new = 15 total leaves
    @test all(is_leaf(octree, idx) for idx in leaves)
    @test !any(has_children(octree, idx) for idx in leaves)
end

@testset "Octree Balancing" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Create imbalanced tree
    # Subdivide root
    children_1 = subdivide!(octree, 1)

    # Subdivide first child
    children_2 = subdivide!(octree, children_1[1])

    # Subdivide one of those children (creating 3-level difference)
    subdivide!(octree, children_2[1])

    # Now we have boxes at levels 2, 4, 8 next to each other
    # Balance should fix this

    criterion = SizeCriterion(0.1)  # Allow small boxes
    balance_octree!(octree, criterion)

    # After balancing, check 2:1 constraint
    # All neighbors should differ by at most 1 level
    for leaf_idx in all_leaves(octree)
        @test !needs_balancing(octree, leaf_idx)
    end
end

@testset "Bounding Box" begin
    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    min_corner, max_corner = bounding_box(octree)
    @test min_corner ≈ SVector(0.0, 0.0, 0.0)
    @test max_corner ≈ SVector(10.0, 10.0, 10.0)
end

println("✓ All basic octree tests passed!")
