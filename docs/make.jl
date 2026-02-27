using WhatsThePoint
using Documenter

DocMeta.setdocmeta!(WhatsThePoint, :DocTestSetup, :(using WhatsThePoint); recursive = true)
remote = Documenter.Remotes.GitHub("JuliaMeshless", "WhatsThePoint.jl")
makedocs(;
    modules = [WhatsThePoint],
    authors = "Kyle Beggs",
    sitename = "WhatsThePoint.jl",
    repo = remote,
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaMeshless.github.io/WhatsThePoint.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Guide" => "guide.md",
        "Concepts" => "concepts.md",
        "Discretization" => "discretization.md",
        "Boundary & Normals" => "boundary_normals.md",
        "Point-in-Volume & Octree" => "isinside_octree.md",
        "Node Repulsion" => "repel.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaMeshless/WhatsThePoint.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "dev" => "dev"],
)
