using WhatsThePoint
using GLMakie
using GeoIO
using Unitful: m

mesh = GeoIO.load("bunny.stl").geometry
boundary = PointBoundary(mesh)

# OctreeSpacing with uniform spacing (simple, fast)
# Note: "Spacing-driven", not solution-adaptive
@time cloud = discretize(boundary, ConstantSpacing(1m); alg = OctreeSpacing(mesh), max_points = 20_000)

visualize(cloud; markersize = 0.3)
