# Hard-case node generation on the Stanford bunny ("the rabbit"), with a
# built-in VISUAL check so you can confirm the result is flawless by eye.
#
# Exercises the full direct pipeline at production scale with a STEEP boundary
# layer and gradient-limited spacing (`max_growth`) — the CFD-relevant case.
# Volume placement defaults to `:bridson` (global graded Poisson-disk) and
# `max_points` is auto-estimated from the spacing integral.
#
# ----------------------------------------------------------------------------
# RUN (from the repo root; bunny.stl already lives there):
#
#     julia -t auto --project examples/octree_boundary_layer.jl
#
# `-t auto` uses all cores (generation is heavily threaded). First run pays a
# one-time CairoMakie precompile; later runs are fast.
#
# OUTPUT: two PNGs you just open — no ParaView needed:
#   • bunny_slice.png  — a thin interior CROSS-SECTION, every point coloured by
#     its local target spacing h(x). This is the money shot: the density must
#     tighten SMOOTHLY from the fine wall layer into the coarse bulk, with no
#     abrupt jump at the layer edge, no clumps, and no voids.
#   • bunny_boundary.png — the Poisson-disk surface sampling, to confirm the
#     wall points are evenly spaced (no holes, no streaks).
# A ParaView-friendly bunny_hard.vtu is also written (set SAVE_VTK = false to
# skip). For an interactive 3D window instead of PNGs, see the note at the end.
# ----------------------------------------------------------------------------

using WhatsThePoint
using GeoIO
using Meshes
using Unitful: m, ustrip
using NearestNeighbors: KDTree, knn
using Printf
using Statistics: median
using CairoMakie   # already a WhatsThePoint dependency — activates plotting

# ---- knobs: make it harder by lowering AT_WALL or raising the BULK/AT_WALL ratio ----
const AT_WALL = 0.3m    # fine spacing at the wall
const BULK = 1.5m       # coarse spacing in the bulk  (5× wall:bulk ratio)
const LAYER_THICKNESS = 12.0m   # boundary-layer depth (bunny is ~86 m across)
const MAX_GROWTH = 0.15  # Lipschitz cap on |∇h| (0 disables); CFD growth ≈ 1.15
const SAVE_VTK = true    # also write bunny_hard.vtu for ParaView
const MESH_PATH = "bunny.stl"
# -------------------------------------------------------------------------------------

isfile(MESH_PATH) || error(
    "Mesh '$MESH_PATH' not found. Run this from the repo root (the bunny lives there), " *
        "or point MESH_PATH at your own STL.",
)

# A quick spacing sanity check before committing to a long run (the "step 0"):
suggest_spacing(MESH_PATH, m; name = MESH_PATH)
println()

println("Loading mesh...")
@time@time mesh = import_mesh(MESH_PATH, m)
println("Mesh: $(nelements(mesh)) triangles\n")

# Boundary-layer spacing keyed on the imported boundary points (KDTree-accelerated).
spacing = BoundaryLayerSpacing(
    WhatsThePoint.points(PointBoundary(mesh));
    at_wall = AT_WALL, bulk = BULK, layer_thickness = LAYER_THICKNESS,
)

println("Poisson-disk boundary sampling...")
@time boundary = PointBoundary(mesh, spacing)
println("Boundary: $(length(boundary)) points\n")

# placement defaults to :bridson; max_growth grades the steep layer smoothly.
alg = Octree(mesh; spacing, alpha = 1.0, max_growth = MAX_GROWTH)

println("Discretizing volume (max_points auto-estimated from the spacing integral)...")
@time cloud = discretize(boundary, spacing; alg)
nvol = length(WhatsThePoint.volume(cloud))
println("Total cloud: $(length(cloud)) points  ($(nvol) volume + $(length(boundary)) boundary)\n")

# ---- quality ----
println("--- quality ---")
sf = spacing_fidelity_metrics(cloud, spacing; k = 30)
cm = metrics(cloud; k = 20)
h_min = ustrip(minimum(spacing.(WhatsThePoint.points(cloud))))
@printf "spacing CV:                %.4f   (< 0.15 good)\n" sf.cv
@printf "sep / h_min:               %.3f   (≈ 0.75 ideal)\n" ustrip(cm.separation) / h_min
@printf "coordination:              %.1f    (12–14 ideal)\n" sf.coordination

# Smoothness probe: the worst spacing jump between a point and its nearest
# neighbour. The gradient limiter should keep this well below the raw value.
let pts = [Float64.(ustrip.(Meshes.to(p))) for p in WhatsThePoint.points(cloud)]
    tree = KDTree(pts)
    idxs, dists = knn(tree, pts, 2, true)
    dnn = [d[2] for d in dists]
    worst = maximum(
        max(dnn[i] / dnn[idxs[i][2]], dnn[idxs[i][2]] / dnn[i]) for i in eachindex(pts)
    )
    @printf "worst neighbour d_NN ratio: %.3f  (lower = smoother; max_growth=%.2f)\n" worst MAX_GROWTH
end

# ============================================================================
# VISUAL CHECK
# ============================================================================
# Helper: strip a point to a plain SVector-like Float64 tuple in metres.
_xyz(p) = Float64.(ustrip.(Meshes.to(p)))

vol_xyz = [_xyz(p) for p in WhatsThePoint.volume(cloud)]
bnd_xyz = [_xyz(p) for p in WhatsThePoint.points(WhatsThePoint.boundary(cloud))]

# Slice perpendicular to the THINNEST axis (broadest, most informative
# cross-section), through the middle, with a slab ~one bulk-spacing thick.
all_xyz = vcat(vol_xyz, bnd_xyz)
lo = reduce((a, b) -> min.(a, b), all_xyz)
hi = reduce((a, b) -> max.(a, b), all_xyz)
extent = hi .- lo
slice_axis = argmin(extent)
plane = (1:3)[1:3 .!= slice_axis]      # the two in-plane axes
u_ax, v_ax = plane[1], plane[2]
slab_center = median(getindex.(all_xyz, slice_axis))
slab_half = 0.75 * ustrip(BULK)        # thin slab → a clean single-layer slice

in_slab(p) = abs(p[slice_axis] - slab_center) <= slab_half
vol_slice = filter(in_slab, vol_xyz)
bnd_slice = filter(in_slab, bnd_xyz)
# Local target spacing at each interior slice point — the colour channel.
h_slice = [ustrip(spacing(Meshes.Point(p...))) for p in vol_slice]

println(
    "\nRendering visual check ($(length(vol_slice)) interior + " *
        "$(length(bnd_slice)) boundary points in the slice)..."
)

fig = Figure(; size = (1500, 760))

# (1) Interior cross-section coloured by spacing — the grading check.
axis_names = ("x", "y", "z")
ax1 = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    title = "interior slice ⟂ $(axis_names[slice_axis]) — colour = target spacing h(x)",
    xlabel = axis_names[u_ax] * " [m]", ylabel = axis_names[v_ax] * " [m]",
)
sc = scatter!(
    ax1,
    getindex.(vol_slice, u_ax), getindex.(vol_slice, v_ax);
    color = h_slice, colormap = :viridis, markersize = 4,
)
# wall outline in the same slab, in black, to frame the layer.
scatter!(
    ax1,
    getindex.(bnd_slice, u_ax), getindex.(bnd_slice, v_ax);
    color = :black, markersize = 3,
)
Colorbar(fig[1, 2], sc; label = "h(x) [m]")

# (2) Boundary Poisson-disk sampling (subsampled for a light render).
stride = max(1, length(bnd_xyz) ÷ 60_000)
bsub = bnd_xyz[1:stride:end]
ax2 = Axis3(
    fig[1, 3];
    aspect = :data, azimuth = 1.275π, elevation = π / 8,
    title = "boundary sampling ($(length(bsub)) of $(length(bnd_xyz)) shown)",
)
scatter!(
    ax2,
    getindex.(bsub, 1), getindex.(bsub, 2), getindex.(bsub, 3);
    color = :steelblue, markersize = 3,
)

save("bunny_slice.png", fig)
println("Saved bunny_slice.png  ← open this: density must grade SMOOTHLY, no clumps/voids.")

# Standalone boundary figure (full 3D, easier to rotate mentally).
figb = Figure(; size = (900, 900))
axb = Axis3(
    figb[1, 1]; aspect = :data, azimuth = 1.275π, elevation = π / 8,
    title = "boundary Poisson-disk sampling"
)
scatter!(
    axb, getindex.(bsub, 1), getindex.(bsub, 2), getindex.(bsub, 3);
    color = :steelblue, markersize = 3
)
save("bunny_boundary.png", figb)
println("Saved bunny_boundary.png ← wall points must be evenly spaced (no holes/streaks).")

if SAVE_VTK
    # ParaView .vtu of every point. Open it, set Representation = "Point Gaussian",
    # then colour by:
    #   • Solid Color           → plain grey point cloud
    #   • surface_id            → one colour per named boundary surface (0 = volume)
    #   • point_type            → boundary (1) vs volume (2)
    # `fields` carries a demo "solution" (here the local target spacing h(x)) so you
    # can see the post-solve workflow — swap in your solver output, ordered like
    # points(cloud) (boundary then volume), and re-export.
    h_field = ustrip.(spacing.(WhatsThePoint.points(cloud)))
    export_vtk("bunny_hard", cloud; fields = ("h_spacing" => h_field,))
    println("Saved bunny_hard.vtu (ParaView: Point Gaussian, colour by surface_id / point_type / h_spacing).")
end

println("\nDone. Open the PNGs to judge node quality, or bunny_hard.vtu in ParaView.")

# Want an interactive, rotatable 3D window instead of PNGs? Install GLMakie in
# this project (`] add GLMakie`) and, after running the script, do:
#     using GLMakie; GLMakie.activate!(); display(visualize(cloud; markersize = 1))
# `visualize` plots the whole cloud; for the interior you'll still want the
# slice above (a solid cloud hides its own interior).
