using WhatsThePoint
using GLMakie
using GeoIO

mesh = GeoIO.load("bunny.stl").geometry
boundary = PointBoundary(mesh)

# OctreeRandom handles octree construction (auto min_ratio, leaf classification)
@time cloud = discretize(boundary, OctreeRandom(mesh); max_points=20_000)

visualize(cloud; markersize=0.3)
