# Fast Marching Method on the embedded Voronoi diagram
# Solves the Eikonal equation |∇T| = F(x) = 1/R(x)

"""
    FastMarchingState{T}

State of the Fast Marching algorithm.

# Fields
- `arrival_time::Vector{T}` - Arrival time T(x) at each Voronoi vertex
- `status::Vector{Symbol}` - Status of each vertex: :far, :narrow_band, or :accepted
- `predecessor::Vector{Int}` - Predecessor vertex for backtracking (0 = source or unreached)
"""
mutable struct FastMarchingState{T<:AbstractFloat}
    arrival_time::Vector{T}
    status::Vector{Symbol}
    predecessor::Vector{Int}
end

"""
    FastMarchingState(voronoi::EmbeddedVoronoi) -> FastMarchingState

Create initial Fast Marching state for a Voronoi diagram.
All vertices start with infinite arrival time, :far status, and no predecessor.
"""
function FastMarchingState(voronoi::EmbeddedVoronoi{T}) where {T}
    n = length(voronoi)
    return FastMarchingState{T}(
        fill(T(Inf), n),
        fill(:far, n),
        zeros(Int, n)
    )
end

"""
    fast_march!(state::FastMarchingState, voronoi::EmbeddedVoronoi, source_idx::Int)

Run the Fast Marching Method from a source vertex.

Propagates a wavefront from `source_idx` with speed F(x) = R(x) (inverse of slowness 1/R).
The arrival time T(x) represents the weighted geodesic distance in the "radius metric".

After completion:
- `state.arrival_time[i]` = weighted distance from source to vertex i
- `state.predecessor[i]` = previous vertex on optimal path from source
- `state.status[i]` = :accepted for all reachable vertices

# Arguments
- `state` - FastMarchingState to update in place
- `voronoi` - Embedded Voronoi diagram
- `source_idx` - Index of source vertex (typically near inlet)
"""
function fast_march!(state::FastMarchingState, voronoi::EmbeddedVoronoi, source_idx::Int)
    if source_idx < 1 || source_idx > length(voronoi)
        error("Invalid source index: $source_idx (valid range: 1-$(length(voronoi)))")
    end

    # Initialize source
    state.arrival_time[source_idx] = 0.0
    state.status[source_idx] = :accepted
    state.predecessor[source_idx] = 0  # No predecessor

    # Priority queue: vertex index => arrival time
    pq = PriorityQueue{Int,Float64}()

    # Add neighbors of source to narrow band
    for neighbor in neighbors(voronoi, source_idx)
        _update_arrival_time!(state, voronoi, neighbor)
        state.status[neighbor] = :narrow_band
        enqueue!(pq, neighbor, state.arrival_time[neighbor])
    end

    # Main loop: process vertices in order of arrival time
    while !isempty(pq)
        # Get vertex with minimum arrival time
        current = dequeue!(pq)

        # Mark as accepted (frozen)
        state.status[current] = :accepted

        # Update all neighbors
        for neighbor in neighbors(voronoi, current)
            if state.status[neighbor] == :accepted
                continue  # Already frozen
            end

            old_time = state.arrival_time[neighbor]
            _update_arrival_time!(state, voronoi, neighbor)

            if state.status[neighbor] == :far
                # First time reaching this vertex
                state.status[neighbor] = :narrow_band
                enqueue!(pq, neighbor, state.arrival_time[neighbor])
            elseif state.arrival_time[neighbor] < old_time
                # Found shorter path - update priority
                pq[neighbor] = state.arrival_time[neighbor]
            end
        end
    end

    return state
end

"""
    _update_arrival_time!(state, voronoi, vertex_idx)

Update the arrival time at a vertex based on accepted neighbors.

Uses the Eikonal equation |∇T| = 1/R, where R is the inscribed sphere radius.
The arrival time is computed as:
    T[vertex] = min over accepted neighbors n of (T[n] + dist(n, vertex) / R[vertex])

Using 1/R as slowness means paths prefer larger radii (more central).
"""
function _update_arrival_time!(
    state::FastMarchingState,
    voronoi::EmbeddedVoronoi,
    vertex_idx::Int
)
    # Get the slowness at this vertex: F = 1/R
    # Smaller radius = slower propagation = higher cost
    R = voronoi.vertices[vertex_idx].radius
    slowness = 1.0 / max(R, 1e-12)  # Avoid division by zero

    # Find minimum arrival time from accepted neighbors
    min_time = state.arrival_time[vertex_idx]
    best_pred = state.predecessor[vertex_idx]

    for neighbor in neighbors(voronoi, vertex_idx)
        if state.status[neighbor] != :accepted
            continue
        end

        # Compute travel time from neighbor to vertex
        dist = edge_length(voronoi, neighbor, vertex_idx)

        # Average slowness along edge (simple approximation)
        neighbor_R = voronoi.vertices[neighbor].radius
        neighbor_slowness = 1.0 / max(neighbor_R, 1e-12)
        avg_slowness = (slowness + neighbor_slowness) / 2

        # Arrival time via this neighbor
        time_via_neighbor = state.arrival_time[neighbor] + avg_slowness * dist

        if time_via_neighbor < min_time
            min_time = time_via_neighbor
            best_pred = neighbor
        end
    end

    state.arrival_time[vertex_idx] = min_time
    state.predecessor[vertex_idx] = best_pred

    return nothing
end

"""
    fast_march_multi_source!(state::FastMarchingState, voronoi::EmbeddedVoronoi, source_idxs::Vector{Int})

Run Fast Marching from multiple source vertices simultaneously.

Useful for computing distances from a set of inlet points.
"""
function fast_march_multi_source!(
    state::FastMarchingState,
    voronoi::EmbeddedVoronoi,
    source_idxs::Vector{Int}
)
    pq = PriorityQueue{Int,Float64}()

    # Initialize all sources
    for source_idx in source_idxs
        if source_idx < 1 || source_idx > length(voronoi)
            error("Invalid source index: $source_idx")
        end
        state.arrival_time[source_idx] = 0.0
        state.status[source_idx] = :accepted
        state.predecessor[source_idx] = 0
    end

    # Add all neighbors of sources to narrow band
    for source_idx in source_idxs
        for neighbor in neighbors(voronoi, source_idx)
            if state.status[neighbor] == :accepted
                continue
            end
            _update_arrival_time!(state, voronoi, neighbor)
            if state.status[neighbor] == :far
                state.status[neighbor] = :narrow_band
                enqueue!(pq, neighbor, state.arrival_time[neighbor])
            elseif haskey(pq, neighbor)
                pq[neighbor] = min(pq[neighbor], state.arrival_time[neighbor])
            end
        end
    end

    # Main loop
    while !isempty(pq)
        current = dequeue!(pq)
        state.status[current] = :accepted

        for neighbor in neighbors(voronoi, current)
            if state.status[neighbor] == :accepted
                continue
            end

            old_time = state.arrival_time[neighbor]
            _update_arrival_time!(state, voronoi, neighbor)

            if state.status[neighbor] == :far
                state.status[neighbor] = :narrow_band
                enqueue!(pq, neighbor, state.arrival_time[neighbor])
            elseif state.arrival_time[neighbor] < old_time
                pq[neighbor] = state.arrival_time[neighbor]
            end
        end
    end

    return state
end

"""
    is_reachable(state::FastMarchingState, vertex_idx::Int) -> Bool

Check if a vertex was reached by the Fast Marching front.
"""
function is_reachable(state::FastMarchingState, vertex_idx::Int)
    return state.status[vertex_idx] == :accepted && isfinite(state.arrival_time[vertex_idx])
end

"""
    count_reached(state::FastMarchingState) -> Int

Count the number of vertices reached by Fast Marching.
"""
function count_reached(state::FastMarchingState)
    return count(s -> s == :accepted, state.status)
end
