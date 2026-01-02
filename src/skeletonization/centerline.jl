# Centerline types and backtracking for Antiga et al. (2003) algorithm

"""
    CenterlinePoint{T}

A point on a centerline with associated maximal inscribed sphere radius.

# Fields
- `position::SVector{3,T}` - Position in 3D space
- `radius::T` - Maximal inscribed sphere radius at this point
- `voronoi_idx::Int` - Index into the embedded Voronoi diagram
"""
struct CenterlinePoint{T<:AbstractFloat}
    position::SVector{3,T}
    radius::T
    voronoi_idx::Int
end

# Accessors
position(cp::CenterlinePoint) = cp.position
radius(cp::CenterlinePoint) = cp.radius

"""
    Centerline{T}

A centerline consisting of a sequence of points from source to target.

# Fields
- `points::Vector{CenterlinePoint{T}}` - Ordered points from source to target
- `arc_length::T` - Total arc length of the centerline
"""
struct Centerline{T<:AbstractFloat}
    points::Vector{CenterlinePoint{T}}
    arc_length::T
end

# Accessors
Base.length(cl::Centerline) = length(cl.points)
Base.getindex(cl::Centerline, i) = cl.points[i]
Base.iterate(cl::Centerline) = iterate(cl.points)
Base.iterate(cl::Centerline, state) = iterate(cl.points, state)

points(cl::Centerline) = cl.points
arc_length(cl::Centerline) = cl.arc_length

"""
    positions(cl::Centerline) -> Vector{SVector{3,T}}

Get all positions along the centerline.
"""
positions(cl::Centerline) = [p.position for p in cl.points]

"""
    radii(cl::Centerline) -> Vector{T}

Get all radii along the centerline.
"""
radii(cl::Centerline) = [p.radius for p in cl.points]

"""
    backtrack_centerline(state::FastMarchingState, voronoi::EmbeddedVoronoi, target_idx::Int) -> Centerline

Backtrack from a target vertex to the source following the predecessor chain.

# Arguments
- `state` - FastMarchingState after running fast_march!
- `voronoi` - Embedded Voronoi diagram
- `target_idx` - Index of target vertex (typically near outlet)

# Returns
- `Centerline` from source to target with radii at each point
"""
function backtrack_centerline(
    state::FastMarchingState,
    voronoi::EmbeddedVoronoi{T},
    target_idx::Int
) where {T}
    if !is_reachable(state, target_idx)
        error("Target vertex $target_idx is not reachable from source")
    end

    # Collect path indices by following predecessors
    path_indices = Int[]
    current = target_idx

    while current != 0
        push!(path_indices, current)
        current = state.predecessor[current]
    end

    # Reverse to get source -> target order
    reverse!(path_indices)

    # Convert to CenterlinePoints
    centerline_points = [
        CenterlinePoint(
            voronoi.vertices[i].position,
            voronoi.vertices[i].radius,
            i
        )
        for i in path_indices
    ]

    # Compute arc length
    total_length = zero(T)
    for i in 1:(length(centerline_points)-1)
        total_length += norm(centerline_points[i+1].position - centerline_points[i].position)
    end

    return Centerline(centerline_points, total_length)
end

"""
    arc_length_parameterize(cl::Centerline) -> Vector{T}

Compute arc length parameter s ∈ [0, arc_length] at each centerline point.
"""
function arc_length_parameterize(cl::Centerline{T}) where {T}
    n = length(cl.points)
    s = Vector{T}(undef, n)
    s[1] = zero(T)

    for i in 2:n
        s[i] = s[i-1] + norm(cl.points[i].position - cl.points[i-1].position)
    end

    return s
end

"""
    interpolate_centerline(cl::Centerline, n_points::Int) -> Centerline

Resample centerline to have uniform spacing with approximately n_points.
"""
function interpolate_centerline(cl::Centerline{T}, n_points::Int) where {T}
    if length(cl.points) < 2
        return cl
    end

    # Get arc length parameterization
    s = arc_length_parameterize(cl)
    total_length = s[end]

    # Target spacing
    ds = total_length / (n_points - 1)

    # Interpolate
    new_points = CenterlinePoint{T}[]
    push!(new_points, cl.points[1])

    target_s = ds
    for i in 1:(length(cl.points)-1)
        while target_s < s[i+1] && target_s < total_length
            # Linear interpolation factor
            t = (target_s - s[i]) / (s[i+1] - s[i])

            # Interpolate position
            pos = cl.points[i].position + t * (cl.points[i+1].position - cl.points[i].position)

            # Interpolate radius
            r = cl.points[i].radius + t * (cl.points[i+1].radius - cl.points[i].radius)

            push!(new_points, CenterlinePoint(pos, r, 0))  # voronoi_idx=0 for interpolated points
            target_s += ds
        end
    end

    # Ensure we include the last point
    if isempty(new_points) || norm(new_points[end].position - cl.points[end].position) > 1e-10
        push!(new_points, cl.points[end])
    end

    return Centerline(new_points, total_length)
end

"""
    mean_radius(cl::Centerline) -> T

Compute the mean radius along the centerline.
"""
function mean_radius(cl::Centerline{T}) where {T}
    return sum(p.radius for p in cl.points) / length(cl.points)
end

"""
    min_radius(cl::Centerline) -> T

Find the minimum radius along the centerline (stenosis location).
"""
min_radius(cl::Centerline) = minimum(p.radius for p in cl.points)

"""
    max_radius(cl::Centerline) -> T

Find the maximum radius along the centerline.
"""
max_radius(cl::Centerline) = maximum(p.radius for p in cl.points)

"""
    stenosis_index(cl::Centerline) -> Int

Find the index of the point with minimum radius (stenosis location).
"""
function stenosis_index(cl::Centerline)
    return argmin(p.radius for p in cl.points)
end

"""
    stenosis_grade(cl::Centerline; reference=:max) -> Float64

Compute stenosis grade as percentage of diameter reduction.

# Arguments
- `cl` - Centerline
- `reference` - Reference for normal diameter: :max (maximum radius) or :mean

# Returns
- Stenosis grade in range [0, 100] where 100 = complete occlusion
"""
function stenosis_grade(cl::Centerline; reference::Symbol=:max)
    r_min = min_radius(cl)

    if reference == :max
        r_ref = max_radius(cl)
    elseif reference == :mean
        r_ref = mean_radius(cl)
    else
        error("Unknown reference: $reference (use :max or :mean)")
    end

    if r_ref < 1e-12
        return 100.0
    end

    return 100.0 * (1.0 - r_min / r_ref)
end

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", cl::Centerline)
    println(io, "Centerline")
    println(io, "  $(length(cl.points)) points")
    println(io, "  arc length: $(cl.arc_length)")
    if length(cl.points) > 0
        println(io, "  radius range: $(min_radius(cl)) - $(max_radius(cl))")
        println(io, "  mean radius: $(mean_radius(cl))")
    end
end
