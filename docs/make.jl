using PointClouds
using Documenter

DocMeta.setdocmeta!(PointClouds, :DocTestSetup, :(using PointClouds); recursive=true)
remote = Documenter.Remotes.GitHub("kylebeggs", "PointClouds.jl")
makedocs(;
    modules=[PointClouds],
    authors="Kyle Beggs",
    sitename="PointClouds.jl",
    repo=remote,
    remotes=nothing,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kylebeggs.github.io/PointClouds.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md", "Getting Started" => "getting_started.md", "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/kylebeggs/PointClouds.jl",
    devbranch="main",
    versions=["stable" => "v^", "dev" => "dev"],
)
