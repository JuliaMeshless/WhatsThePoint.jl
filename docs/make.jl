using WhatsThePoint
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(WhatsThePoint, :DocTestSetup, :(using WhatsThePoint); recursive = true)

makedocs(;
    modules = [WhatsThePoint],
    authors = "Kyle Beggs",
    sitename = "WhatsThePoint.jl",
    remotes = nothing,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/JuliaMeshless/WhatsThePoint.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Getting Started" => [
            "Quick Start" => "quickstart.md",
            "Guide" => "guide.md",
        ],
        "Manual" => [
            "Concepts" => "concepts.md",
            "Discretization" => "discretization.md",
            "Octree Algorithm" => "octree.md",
            "Boundary & Normals" => "boundary_normals.md",
            "Point-in-Volume & Octree" => "isinside_octree.md",
            "Node Repulsion" => "repel.md",
        ],
        "API Reference" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaMeshless/WhatsThePoint.jl",
    devbranch = "main",
    push_preview = true,
)
