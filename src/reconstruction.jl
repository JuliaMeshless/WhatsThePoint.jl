# intent of this source file is to provide surface reconstruction capabilities, such as in the paper "Implicit surface reconstruction with radial basis functions via PDEs" by Liu (2019)

# use RBFs or b-splines to paramterize a surface

"""
    struct ParameterizedSurface{Dim, T} <: AbstractSurface{Dim,T}
A surface that has been parameterized from points into a continous function representation.
Contains views (data, not visual) of a surface of a [PointBoundary](@ref).

## Note
This is a data view, so mutations to this data will change the data in the [PointBoundary](@ref)
       which this surface resides as well.
"""
struct ParameterizedSurface{Dim,T} <: AbstractSurface{Dim,T} end
