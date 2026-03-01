using WhatsThePoint
using GeoIO
using GLMakie
using Meshes
using Statistics: quantile
using Unitful: m, ustrip
using Random

# Density-aware discretization example with colorized point density.
#
# Run from project root:
#   julia --project=. examples/density_aware_discretization.jl

Random.seed!(42)

stl_file = "bunny.stl"
isfile(stl_file) || error("Bunny mesh not found at $stl_file")

println("Loading mesh: $stl_file")
mesh = GeoIO.load(stl_file).geometry
boundary = WhatsThePoint.PointBoundary(mesh)

println("Building spacing function (fine near boundary, coarse in interior)...")
spacing = SigmoidSpacing(
    WhatsThePoint.points(boundary),
    0.6m,   # h_boundary
    4.0m,   # h_interior
    8.0m,   # transition_center
    3.0m,   # transition_width
)

println("Building DensityAwareOctree and generating points...")
alg = DensityAwareOctree(mesh; placement=:jittered, boundary_oversampling=2.0)
@time cloud = discretize(boundary, spacing; alg=alg, max_points=120_000)

vol_pts = WhatsThePoint.points(WhatsThePoint.volume(cloud))
println("Generated volume points: $(length(vol_pts))")

# Compute a scalar density proxy rho = 1 / h^3 and color by log10(rho)
ρ_vals = Float64[]
xs = Float32[]
ys = Float32[]
zs = Float32[]
sizehint!(ρ_vals, length(vol_pts))
sizehint!(xs, length(vol_pts))
sizehint!(ys, length(vol_pts))
sizehint!(zs, length(vol_pts))

for p in vol_pts
    h = spacing(p)
    ρ = 1.0 / (ustrip(h)^3)
    c = Meshes.to(p)
    push!(ρ_vals, ρ)
    push!(xs, Float32(ustrip(c[1])))
    push!(ys, Float32(ustrip(c[2])))
    push!(zs, Float32(ustrip(c[3])))
end

logρ = log10.(ρ_vals)
q02, q98 = quantile(logρ, [0.02, 0.98])
println("log10(1/h^3) range: [$(minimum(logρ)), $(maximum(logρ))], display range: [$q02, $q98]")

# Boundary points for context in the density plot
bnd_pts = WhatsThePoint.points(boundary)
bx = Float32[]
by = Float32[]
bz = Float32[]
sizehint!(bx, length(bnd_pts))
sizehint!(by, length(bnd_pts))
sizehint!(bz, length(bnd_pts))
for p in bnd_pts
    c = Meshes.to(p)
    push!(bx, Float32(ustrip(c[1])))
    push!(by, Float32(ustrip(c[2])))
    push!(bz, Float32(ustrip(c[3])))
end

fig = Figure(size=(1200, 850))
ax = Axis3(fig[1, 1],
    title="Density-aware discretization (volume color = log10(1/h^3))",
    xlabel="x", ylabel="y", zlabel="z",
)

# Show boundary as neutral context
scatter!(ax, bx, by, bz;
    color=(:gray60, 0.10),
    markersize=1.1,
)

plt = scatter!(ax, xs, ys, zs;
    color=logρ,
    colormap=:viridis,
    colorrange=(q02, q98),
    markersize=2.2,
)

Colorbar(fig[1, 2], plt, label="log10(1 / h^3)")

display(fig)
