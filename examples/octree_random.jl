using WhatsThePoint
using GLMakie
using GeoIO
using Unitful: m

mesh = GeoIO.load("bunny.stl").geometry
boundary = PointBoundary(mesh)

# OctreeRandom handles octree construction (auto min_ratio, leaf classification)
# Spacing is ignored by OctreeRandom (uniform random generation)
@time cloud = discretize(boundary, ConstantSpacing(1m); alg = OctreeRandom(mesh), max_points = 20_000)

visualize(cloud; markersize = 0.3)
