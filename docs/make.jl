using PointClouds
using Documenter

DocMeta.setdocmeta!(PointClouds, :DocTestSetup, :(using PointClouds); recursive=true)

makedocs(;
    modules=[PointClouds],
    authors="Kyle Beggs",
    repo="https://github.com/kylebeggs/PointClouds.jl/blob/{commit}{path}#{line}",
    sitename="PointClouds.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kylebeggs.github.io/PointClouds.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Getting Started" => "getting_started.md"],
)

deploydocs(; repo="github.com/kylebeggs/PointClouds.jl", devbranch="main")
