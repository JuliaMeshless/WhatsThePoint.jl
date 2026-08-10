# 2D Octree discretization on a starfish domain, with a VISUAL breakdown of
# the algorithm's stages — the 2D counterpart of octree_boundary_layer.jl.
#
# The same `Octree` algorithm that fills 3D meshes runs here on a closed 2D
# loop: a `SegmentQuadtree` indexes the boundary segments, a spacing-driven
# node quadtree subdivides the domain, its leaves are classified against the
# boundary, and a graded Bridson (Poisson-disk) front seeded from the boundary
# points fills the interior.
#
# ----------------------------------------------------------------------------
# RUN (from the repo root):
#
#     julia --project examples/octree_quadtree_2d.jl
#
# First run pays a one-time CairoMakie precompile; later runs are fast.
#
# OUTPUT: three PNGs you just open (the docs figures on the Octree page):
#   • quadtree_2d_boxes.png — boundary points and the spacing-driven node
#     quadtree, finer near the wall because the spacing is graded.
#   • quadtree_2d_dart.png — the Bridson front caught mid-run, with one
#     accepted candidate encircled by its spacing radius (engineering-style
#     radius callout): the acceptance rule in one picture.
#   • quadtree_2d_cloud.png — the finished cloud, boundary + graded interior.
# ----------------------------------------------------------------------------

using WhatsThePoint
using Unitful: m, ustrip
using CairoMakie   # already a WhatsThePoint dependency — activates plotting

# ---- knobs ----
const H_WALL = 0.035    # spacing at the boundary
const H_BULK = 0.14     # spacing in the core
const LAYER = 0.35      # boundary-layer depth (starfish is ~2.6 across)

# ---- boundary: starfish r(θ) = 1 + 0.3 cos(5θ), ~equal arc-length spacing ----
r(θ) = 1 + 0.3 * cos(5θ)
θf = range(0, 2π; length = 20_000)
xs, ys = r.(θf) .* cos.(θf), r.(θf) .* sin.(θf)
arclen = cumsum(hypot.(diff(xs), diff(ys)))
targets = range(0, arclen[end]; length = round(Int, arclen[end] / H_WALL) + 1)[1:(end - 1)]
idx = [searchsortedfirst(arclen, t) for t in targets]
pts = WhatsThePoint.Point.(collect(zip(xs[idx], ys[idx])))

# ---- the three lines from the docs ----
bnd = PointBoundary(pts)
spacing = BoundaryLayerSpacing(
    WhatsThePoint.points(bnd);
    at_wall = H_WALL * m, bulk = H_BULK * m, layer_thickness = LAYER * m,
)
alg = Octree(bnd; spacing)
cloud = discretize(bnd, spacing; alg)

# ---- gather what the figures need ----
# The stages figure peeks at two internals (the node quadtree and its leaf
# classification) purely for visualization; nothing below feeds back into the
# generated cloud.
node_tree = WhatsThePoint.build_node_octree(alg.geometry, spacing, alg.alpha, alg.node_min_ratio)
classification = WhatsThePoint.classify_node_octree(node_tree, alg.geometry)
leaves = WhatsThePoint.all_leaves(node_tree)

bx = [ustrip(p.coords.x) for p in WhatsThePoint.points(bnd)]
by = [ustrip(p.coords.y) for p in WhatsThePoint.points(bnd)]
vol = WhatsThePoint.volume(cloud)   # kept in Bridson insertion order
vx = [ustrip(p.coords.x) for p in vol]
vy = [ustrip(p.coords.y) for p in vol]
@info "cloud" n_boundary = length(bx) n_interior = length(vx) n_leaves = length(leaves)

leaf_rect(b) = begin
    lo, hi = WhatsThePoint.box_bounds(node_tree, b)
    Rect2f(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2])
end
rect_edges(b) = begin
    lo, hi = WhatsThePoint.box_bounds(node_tree, b)
    Point2f[
        (lo[1], lo[2]), (hi[1], lo[2]), (hi[1], lo[2]), (hi[1], hi[2]),
        (hi[1], hi[2]), (lo[1], hi[2]), (lo[1], hi[2]), (lo[1], lo[2]),
    ]
end
tinted(cls) = [leaf_rect(b) for b in leaves if classification[b] == cls]

# ---- figure 1: the star geometry and its node quadtree ----
# Only non-exterior boxes are drawn: the sampler never sees exterior leaves,
# and drawing them would wrongly suggest the tree samples outside the domain.
shown = [b for b in leaves if classification[b] != WhatsThePoint.LEAF_EXTERIOR]
fig = Figure(; size = (700, 700), backgroundcolor = :transparent)
ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = :transparent)
hidedecorations!(ax); hidespines!(ax)
linesegments!(
    ax, reduce(vcat, rect_edges.(shown));
    color = (:gray, 0.6), linewidth = 0.7,
)
scatter!(ax, bx, by; color = :red, markersize = 7)
save(joinpath(@__DIR__, "quadtree_2d_boxes.png"), fig)
@info "wrote quadtree_2d_boxes.png"

# ---- figure 2: the Bridson dart with its spacing radius ----
# Catch the front mid-run and encircle one accepted candidate with the disk
# it had to keep empty: radius r = bridson_factor · h(x). The candidate is
# picked in the coarse core so the circle is large enough to read.
hs = [ustrip(spacing(p)) for p in vol]
isempty(hs) && error("no interior points to illustrate — coarsen the knobs at the top")
# `2n÷3` is an empty range for n < 2 (coarsened knobs), and argmax throws on it.
ci = argmax(@view hs[1:max(1, 2 * length(hs) ÷ 3)])   # coarsest accepted point so far
rc = alg.bridson_factor * hs[ci]
cx, cy = vx[ci], vy[ci]

figd = Figure(; size = (700, 700), backgroundcolor = :transparent)
axd = Axis(figd[1, 1]; aspect = DataAspect(), backgroundcolor = :transparent)
hidedecorations!(axd); hidespines!(axd)
scatter!(axd, vx[1:(ci - 1)], vy[1:(ci - 1)]; color = :royalblue, markersize = 5)
scatter!(axd, bx, by; color = :red, markersize = 7)
# the accepted dart, highlighted
scatter!(axd, [cx], [cy]; color = :orange, strokecolor = :black, strokewidth = 1, markersize = 11)
# its empty disk
θc = range(0, 2π; length = 200)
lines!(axd, cx .+ rc .* cos.(θc), cy .+ rc .* sin.(θc); color = :red, linewidth = 1.5)
# Engineering-style radius callout: radius line from the center tick to the
# arc with an arrowhead, then a leader extending past the arc to a horizontal
# landing with the dimension label — like an R-dimension on a drawing.
ϕ = deg2rad(35)
tip = (cx + rc * cos(ϕ), cy + rc * sin(ϕ))
elbow = (cx + 1.8 * rc * cos(ϕ), cy + 1.8 * rc * sin(ϕ))
landing = (elbow[1] + 0.6 * rc, elbow[2])
lines!(axd, [cx, tip[1]], [cy, tip[2]]; color = :red, linewidth = 1.2)
scatter!(
    axd, [tip[1]], [tip[2]];
    color = :red, marker = :utriangle, markersize = 11, rotation = ϕ - π / 2,
)
scatter!(axd, [cx], [cy]; color = :red, marker = :+, markersize = 8)
lines!(
    axd, [tip[1], elbow[1], landing[1]], [tip[2], elbow[2], landing[2]];
    color = :red, linewidth = 1.0,
)
text!(
    axd, landing[1] + 0.06 * rc, landing[2];
    text = "r = 0.75 h(x)", color = :red, fontsize = 18, align = (:left, :center),
)
save(joinpath(@__DIR__, "quadtree_2d_dart.png"), figd)
@info "wrote quadtree_2d_dart.png"

# ---- figure 3: the finished cloud ----
fig2 = Figure(; size = (700, 700), backgroundcolor = :transparent)
ax2 = Axis(fig2[1, 1]; aspect = DataAspect(), backgroundcolor = :transparent)
hidedecorations!(ax2); hidespines!(ax2)
scatter!(ax2, vx, vy; color = :royalblue, markersize = 5)
scatter!(ax2, bx, by; color = :red, markersize = 7)
save(joinpath(@__DIR__, "quadtree_2d_cloud.png"), fig2)
@info "wrote quadtree_2d_cloud.png"
