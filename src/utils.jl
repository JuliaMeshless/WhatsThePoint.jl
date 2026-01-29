function findmin_turbo(x)
    indmin = 0
    minval = typemax(eltype(x))
    for (i, y) in enumerate(x)
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
    ranges = [(e-num_elems[i]+1):e for (i, e) in enumerate(ends)]
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

function _angle(u::SVector{2}, v::SVector{2}) # preserve sign
    θ = atan(u × v, u ⋅ v) * u"rad"
    return θ == oftype(θ, -π) ? -θ : θ
end

_angle(u::SVector{3}, v::SVector{3}) = atan(norm(u × v), u ⋅ v) * u"rad" # discard sign
_angle(u::Vec, v::Vec) = ∠(u, v)
