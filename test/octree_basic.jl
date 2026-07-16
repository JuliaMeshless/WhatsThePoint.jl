# Basic tests for generic spatial octree implementation

@testitem "SpatialOctree Construction" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, is_leaf, has_children

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    @test octree.num_boxes[] == 1
    @test octree.cell[1] == SVector(0, 0, 0) &&
        octree.level[1] == 1
    @test octree.parent[1] == 0
    @test is_leaf(octree, 1)
    @test !has_children(octree, 1)
end

@testitem "SpatialOctree Box Geometry" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, box_center, box_size, box_bounds

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    center = box_center(octree, 1)
    @test center ≈ SVector(5.0, 5.0, 5.0)

    @test box_size(octree, 1) == 10.0

    min_corner, max_corner = box_bounds(octree, 1)
    @test min_corner ≈ SVector(0.0, 0.0, 0.0)
    @test max_corner ≈ SVector(10.0, 10.0, 10.0)
end

@testitem "SpatialOctree Subdivision" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, is_leaf, has_children, box_size, box_center

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    children = subdivide!(octree, 1)

    @test length(children) == 8
    @test octree.num_boxes[] == 9
    @test !is_leaf(octree, 1)
    @test has_children(octree, 1)
    @test all(is_leaf(octree, c) for c in children)

    # Check child coordinates
    @test octree.cell[children[1]] == SVector(0, 0, 0) &&
        octree.level[children[1]] == 2  # (0,0,0) at level 2
    @test octree.cell[children[8]] == SVector(1, 1, 1) &&
        octree.level[children[8]] == 2  # (1,1,1) at level 2

    # Check child sizes
    @test all(box_size(octree, c) ≈ 5.0 for c in children)

    # Check child centers
    center_first = box_center(octree, children[1])
    @test center_first ≈ SVector(2.5, 2.5, 2.5)

    center_last = box_center(octree, children[8])
    @test center_last ≈ SVector(7.5, 7.5, 7.5)
end

@testitem "SpatialOctree Point Query" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, find_leaf, is_leaf

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
    @test octree.cell[leaf] == SVector(0, 0, 0) &&
        octree.level[leaf] == 2

    # Query point in last octant (1,1,1)
    point = SVector(8.0, 8.0, 8.0)
    leaf = find_leaf(octree, point)
    @test octree.cell[leaf] == SVector(1, 1, 1) &&
        octree.level[leaf] == 2

    # Query point on boundary (should go to one side consistently)
    point = SVector(5.0, 5.0, 5.0)
    leaf = find_leaf(octree, point)
    @test is_leaf(octree, leaf)
end

@testitem "SpatialOctree Neighbor Finding" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, find_neighbor

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    children = subdivide!(octree, 1)

    # First child (0,0,0) at level 2
    first_child = children[1]

    # Neighbor in +x direction should be (1,0,0)
    neighbors_x = find_neighbor(octree, first_child, 2)  # +x direction
    @test length(neighbors_x) == 1
    @test octree.cell[neighbors_x[1]] == SVector(1, 0, 0) &&
        octree.level[neighbors_x[1]] == 2

    # Neighbor in +y direction should be (0,1,0)
    neighbors_y = find_neighbor(octree, first_child, 4)  # +y direction
    @test length(neighbors_y) == 1
    @test octree.cell[neighbors_y[1]] == SVector(0, 1, 0) &&
        octree.level[neighbors_y[1]] == 2

    # Neighbor in -x direction should be empty (boundary)
    neighbors_boundary = find_neighbor(octree, first_child, 1)  # -x direction
    @test isempty(neighbors_boundary)
end

@testitem "SpatialOctree Element Storage" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, num_elements

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

@testitem "SpatialOctree Leaf Iteration" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, all_leaves, is_leaf, has_children

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

@testitem "SpatialOctree Balancing" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, all_leaves, balance_octree!,
        SubdivisionCriterion, should_subdivide, can_subdivide, needs_balancing, box_size

    # Minimal size-only criterion for the balancing test (no triangle index needed).
    struct SizeCriterion{T <: Real} <: SubdivisionCriterion
        h_min::T
    end
    WhatsThePoint.should_subdivide(c::SizeCriterion, tree, box_idx) = box_size(tree, box_idx) > c.h_min
    WhatsThePoint.can_subdivide(c::SizeCriterion, tree, box_idx) = box_size(tree, box_idx) > c.h_min

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Create imbalanced tree
    children_1 = subdivide!(octree, 1)
    children_2 = subdivide!(octree, children_1[1])
    subdivide!(octree, children_2[1])

    criterion = SizeCriterion(0.1)
    balance_octree!(octree, criterion)

    for leaf_idx in all_leaves(octree)
        @test !needs_balancing(octree, leaf_idx)
    end
end

@testitem "SpatialOctree Bounding Box" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, bounding_box

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    min_corner, max_corner = bounding_box(octree)
    @test min_corner ≈ SVector(0.0, 0.0, 0.0)
    @test max_corner ≈ SVector(10.0, 10.0, 10.0)
end

@testitem "SpatialOctree Capacity Growth" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, is_leaf

    origin = SVector(0.0, 0.0, 0.0)
    # initial_capacity=1 forces the resize path in add_box! (lines 242-256)
    octree = SpatialOctree{Int, Float64}(origin, 10.0; initial_capacity = 1)

    # Root uses slot 1; subdivision adds 8 children → must grow arrays
    children = subdivide!(octree, 1)
    @test octree.num_boxes[] == 9
    @test all(is_leaf(octree, c) for c in children)

    # Verify all arrays grew consistently
    @test length(octree.parent) >= 9
    @test length(octree.first_child) >= 9
    @test length(octree.cell) >= 9
    @test length(octree.element_lists) >= 9

    # Verify children have correct coordinates after resize
    @test octree.cell[children[1]] == SVector(0, 0, 0) &&
        octree.level[children[1]] == 2
    @test octree.cell[children[8]] == SVector(1, 1, 1) &&
        octree.level[children[8]] == 2

    # Second subdivision should also work (may trigger another resize)
    grandchildren = subdivide!(octree, children[1])
    @test octree.num_boxes[] == 17
    @test all(is_leaf(octree, gc) for gc in grandchildren)
    @test octree.cell[grandchildren[1]] == SVector(0, 0, 0) &&
        octree.level[grandchildren[1]] == 4
end

@testitem "SpatialOctree all_boxes" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, all_boxes

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    # Single root box
    @test all_boxes(octree) == [1]

    # After subdivision: root + 8 children
    subdivide!(octree, 1)
    boxes = all_boxes(octree)
    @test length(boxes) == 9
    @test boxes == collect(1:9)

    # After second subdivision: 9 + 8 = 17
    subdivide!(octree, 2)  # subdivide first child
    boxes = all_boxes(octree)
    @test length(boxes) == 17
    @test boxes == collect(1:17)
end

@testitem "SpatialOctree needs_balancing non-leaf" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, needs_balancing

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    subdivide!(octree, 1)
    # Root (box 1) is not a leaf after subdivision → early return false
    @test needs_balancing(octree, 1) == false
end

@testitem "SpatialOctree any_leaf_overlapping" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, any_leaf_overlapping

    origin = SVector(0.0, 0.0, 0.0)
    tree = SpatialOctree{Int, Float64}(origin, 10.0)
    subdivide!(tree, 1)

    # Query box overlaps the first octant → at least one leaf passes `true`
    @test any_leaf_overlapping(
        tree, SVector(0.5, 0.5, 0.5), SVector(1.0, 1.0, 1.0), _ -> true
    ) == true

    # Query box entirely outside the root → pruned at root
    @test any_leaf_overlapping(
        tree, SVector(100.0, 100.0, 100.0), SVector(101.0, 101.0, 101.0), _ -> true
    ) == false

    # Overlap exists but predicate rejects every leaf
    @test any_leaf_overlapping(
        tree, SVector(0.0, 0.0, 0.0), SVector(10.0, 10.0, 10.0), _ -> false
    ) == false

    # Predicate selects a single leaf by coords — true only when query overlaps it
    target = only(findall(i -> tree.cell[i] == SVector(1, 1, 1) && tree.level[i] == 2, 1:tree.num_boxes[]))
    predicate = idx -> idx == target
    @test any_leaf_overlapping(
        tree, SVector(6.0, 6.0, 6.0), SVector(9.0, 9.0, 9.0), predicate
    ) == true
    @test any_leaf_overlapping(
        tree, SVector(0.0, 0.0, 0.0), SVector(4.0, 4.0, 4.0), predicate
    ) == false
end

@testitem "SpatialOctree find_boxes_at_coords edge cases" setup = [CommonImports] begin
    using WhatsThePoint: SpatialOctree, subdivide!, find_neighbor, find_boxes_at_coords

    origin = SVector(0.0, 0.0, 0.0)
    octree = SpatialOctree{Int, Float64}(origin, 10.0)

    children_L1 = subdivide!(octree, 1)
    children_L2 = subdivide!(octree, children_L1[1])

    # --- Line 448: exact match at target level ---
    # Search for (1,0,0) at level 2 — navigates root→child 2 and matches
    result = find_boxes_at_coords(octree, 1, 0, 0, 2)
    @test result == [children_L1[2]]
    @test octree.cell[result[1]] == SVector(1, 0, 0) &&
        octree.level[result[1]] == 2

    # Also verify via find_neighbor: grandchild (1,0,0,4) looking -x → (0,0,0,4)
    result = find_neighbor(octree, children_L2[2], 1)  # (1,0,0,4), -x direction
    @test length(result) == 1
    @test octree.cell[result[1]] == SVector(0, 0, 0) &&
        octree.level[result[1]] == 4

    # --- Line 412: neighbor outside domain ---
    # Grandchild (0,0,0,4) looking -x: i_n = -1 → outside domain
    result = find_neighbor(octree, children_L2[1], 1)
    @test isempty(result)

    # Also test other boundary directions
    result = find_neighbor(octree, children_L2[1], 3)  # -y: j_n = -1
    @test isempty(result)
    result = find_neighbor(octree, children_L2[1], 5)  # -z: k_n = -1
    @test isempty(result)

    # --- Line 468: target coords outside root's subtree ---
    # Direct call with coords (2,0,0) at level 2 — root is (0,0,0,1),
    # scale_factor=2, i_scaled=2÷2=1 != root.i=0 → Int[]
    result = find_boxes_at_coords(octree, 2, 0, 0, 2)
    @test isempty(result)

    result = find_boxes_at_coords(octree, 0, 2, 0, 2)
    @test isempty(result)

    # --- Coarser neighbor: leaf covers target region (line 456) ---
    # Grandchild (1,0,0,4) looking +x: i_n=2, N=4, within domain (2 < 4)
    # Navigates to child (1,0,0,2) which is a leaf → returns coarser box
    result = find_neighbor(octree, children_L2[2], 2)  # +x
    @test length(result) == 1
    @test octree.cell[result[1]] == SVector(1, 0, 0) &&
        octree.level[result[1]] == 2
end

@testitem "SpatialTree quadtree (N=2) genericity" setup = [CommonImports] begin
    using WhatsThePoint: SpatialTree, subdivide!, find_leaf, is_leaf, box_bounds, n_children
    using StaticArrays

    tree = SpatialTree{2, Int, Float64}(SVector(0.0, 0.0), 1.0)
    @test n_children(tree) == 4                     # 2^2, the same generic code as the octree

    # complete depth-3 quadtree from the generic subdivide! (in a function to
    # avoid loop soft-scope on the reassigned frontier)
    function build_complete!(t, depth)
        frontier = [1]
        for _ in 1:depth
            nxt = Int[]
            for b in frontier
                append!(nxt, collect(subdivide!(t, b)))
            end
            frontier = nxt
        end
        return frontier
    end
    leaves = build_complete!(tree, 3)
    @test length(leaves) == 4^3                     # 64 leaves at depth 3

    # every located leaf must actually contain its query point
    for _ in 1:2000
        p = SVector(rand(), rand())
        b = find_leaf(tree, p)
        @test is_leaf(tree, b)
        lo, hi = box_bounds(tree, b)
        @test all(lo .<= p) && all(p .<= hi)
    end
end

@testitem "SpatialTree child-contiguity invariant" setup = [CommonImports] begin
    using WhatsThePoint: SpatialTree, subdivide!, children, n_children, is_leaf
    using StaticArrays

    # first_child assumes children are a contiguous 2^N block — guard it for N=2 and N=3.
    for N in (2, 3)
        tree = SpatialTree{N, Int, Float64}(zero(SVector{N, Float64}), 1.0)
        rng = subdivide!(tree, 1)
        fc = tree.first_child[1]
        @test collect(rng) == collect(fc:(fc + n_children(tree) - 1))
        @test collect(children(tree, 1)) == collect(rng)
        @test length(rng) == n_children(tree)
        @test !is_leaf(tree, 1)
        # a grandchild subdivision is also a contiguous block
        g = subdivide!(tree, first(rng))
        gfc = tree.first_child[first(rng)]
        @test collect(g) == collect(gfc:(gfc + n_children(tree) - 1))
    end
end
