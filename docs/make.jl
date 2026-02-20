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
        "Getting Started" => "getting_started.md",
        "Discretization" => "discretization.md",
        "Octree" => "octree.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaMeshless/WhatsThePoint.jl",
    devbranch = "main",
    versions = ["stable" => "v^", "dev" => "dev"],
)
