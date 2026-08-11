using WhatsThePoint
using GLMakie
using GeoIO
using Unitful: m

mesh = import_mesh("bunny.stl", m)
boundary = PointBoundary(mesh)

# Orthtree with uniform spacing (simple, fast)
# Note: "Spacing-driven", not solution-adaptive
@time cloud = discretize(boundary, ConstantSpacing(1m); alg = Orthtree(mesh), max_points = 20_000)

visualize(cloud; markersize = 0.3)
