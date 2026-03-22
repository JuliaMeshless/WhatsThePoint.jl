using WhatsThePoint
using GLMakie
using GeoIO
using Unitful: m

mesh = GeoIO.load("bunny.stl").geometry
boundary = PointBoundary(mesh)

# AdaptiveOctree with uniform spacing (simple, fast)
@time cloud = discretize(boundary, ConstantSpacing(1m); alg=AdaptiveOctree(mesh), max_points=20_000, repel_iters=100)

visualize(cloud; markersize=0.3)
