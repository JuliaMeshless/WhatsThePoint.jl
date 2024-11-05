function findmin_turbo(x)
    indmin = 0
    minval = typemax(eltype(x))
    @turbo for (i, y) in enumerate(x)
        newmin = y < minval
        minval = newmin ? y : minval
        indmin = newmin ? i : indmin
    end
    return minval, indmin
end

function ranges_from_permutation(permutations::AbstractVector)
    num_elems = length.(permutations)
    # use cumsum here
    ends = map(i -> sum(num_elems[begin:i]), 1:length(num_elems))
    ranges = [(e - num_elems[i] + 1):e for (i, e) in enumerate(ends)]
    return ranges
end

function many_permute!(arr, permutations::AbstractVector, ranges::AbstractVector)
    perm_arr = similar(arr)
    for (range, perm) in zip(ranges, permutations)
        perm_arr[range] .= arr[perm]
    end
    arr .= perm_arr
    return nothing
end

function many_permute!(
    points::PointSet, permutations::AbstractVector, ranges::AbstractVector
)
    return many_permute!(parent(points), permutations, ranges)
end
