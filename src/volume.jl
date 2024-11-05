struct PointVolume{M<:Manifold,C<:CRS} <: Domain{M,C}
    points::Domain{M,C}
end
PointVolume{M,C}() where {M<:Manifold,C<:CRS} = PointVolume(PointSet(Point{M,C}[]))

Base.length(vol::PointVolume) = length(vol.points)
Base.size(vol::PointVolume) = (length(vol),)
Base.getindex(vol::PointVolume, index) = vol.points[index]
function Base.iterate(vol::PointVolume, state=1)
    return state > length(vol) ? nothing : (vol[state], state + 1)
end
Base.isempty(vol::PointVolume) = isempty(vol.points)
Base.parent(vol::PointVolume) = vol.points
Base.filter!(f::Function, vol::PointVolume) = filter!(f, vol.points)

to(vol::PointVolume) = to.(vol.points)
centroid(vol::PointVolume) = centroid(PointSet(vol.points))
boundingbox(vol::PointVolume) = boundingbox(vol.points)

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", vol::PointVolume{Dim,T}) where {Dim,T}
    println(io, "PointVolume{$Dim, $T}")
    println(io, "└─Number of points: $(length(vol.points))")
    return nothing
end

Base.show(io::IO, ::PointVolume) = println(io, "PointVolume")
