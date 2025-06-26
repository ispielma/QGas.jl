"""
    ArrayDimensions

A module to deal with the basic information needed to create and maintain 
dimensioning information in a simple way

One confusing language thing:
    values: the actual variables to be scaled along each direction

    coords: the coordinate from 1...n (integer) along each direction of a 
        specific value
    
    index : the index in a 1D array that could be used to represent a vector
        of values (x0...xn) or of coords (c0...cn)

thus the functions

    CoordsFromValues
    ValuesFromCoords

Transform between scaled vectors (Values) and integer valued index vectors (Coords)

Then these functions

    IndexFromCoords
    CoordsFromIndex

move between these different indexing representations:  given an 
index i, return the (x,y,z,...) coordinates associated with it and the reverse.
"""
module ArrayDimensions

    _VectorOrTuple{T} = Union{Vector{T}, NTuple{N,T}} where N

    # Import the functions that I plan to add methods to
    import Base: ndims, size, length, copy, reshape, getindex, setindex!, show, +, *, /, ^
    import Base: CartesianIndices, LinearIndices, convert

    convert(::Type{Vector}, a::Tuple) = [x for x in a]

    """
    Dimension

    defines the dimensional properties of an axis of an N-dimensional array
    """ 

    abstract type AbstractDimension end

    Base.@kwdef mutable struct Dimension <: AbstractDimension
        x0::Float64 = 0.0
        dx::Float64 = 1.0
        npnts::Int = 0
        unit::String = ""
        symmetric::Bool = false
        periodic::Bool = false
    end 

    mutable struct Dimensions
        dims::Vector{AbstractDimension}
    end
    Dimensions() = Dimensions(AbstractDimension[])
    Dimensions(adims::Dimensions) = copy(adims)
    Dimensions(adims::NTuple{N, AbstractDimension}) where N = Dimensions(collect(adims))
    Dimensions(adims::AbstractVector{AbstractDimension}) = Dimensions(collect(adims))

    Dimensions(adims::AbstractDimension...) = Dimensions(adims)
    Dimensions(adim::AbstractDimension) = Dimensions([adim])

    # #############################################################################
    #
    # Extend builtin methods
    #
    # #############################################################################

    """
    copy

    new method for builtin copy
    """
    copy(adims::Dimensions) = Dimensions( copy(adims.dims) )

    """
    ndims

    new method for builtin ndims, returns dimension of the object
    """
    ndims(adims::Dimensions) = length(adims.dims)

    """
        size(adims::Dimensions)
        size(T::Type, adims::Dimensions)
        size(adims::Dimensions, i::Integer)

    returns the size of an array described by dimensions.

        * T::Type if present the dimensions will be stored in a contained of this type.  Default is Tuple
        * i::Integer if present return the size of the i-th dimension.
    """
    size(::Type{Vector}, adims::Dimensions) = [d.npnts for d in adims.dims]
    size(::Type{T}, adims::Dimensions) where T = T(size(Vector, adims))

    size(adims::Dimensions) = size(Tuple, adims::Dimensions)
    size(adims::Dimensions, i::Integer) = adims.dims[i].npnts

    """
        length(adims::Dimensions)

    new method for builtin length, returns total number of elements associated with the Dimensions
    """
    length(adims::Dimensions) = prod(size(Vector, adims)) # Vector because prod is faster

    """
        LinearIndices(adims::Dimensions)

    new method for builtin LinearIndices for Dimensions type
    """
    LinearIndices(adims::Dimensions) = LinearIndices(coord_ranges(adims))

    """
        CartesianIndices(adims::Dimensions)

    new method for builtin CartesianIndices for Dimensions type
    """
    CartesianIndices(adims::Dimensions) = CartesianIndices(coord_ranges(adims))

    # #############################################################################
    #
    # Code ported from python
    #
    # #############################################################################

    """
        deltas(adims::Dimensions)
        deltas(T::Type, adims::Dimensions)

    returns an array of the dx values in the array

        * T::Type if present the dx values will be stored in a contained of this type.  Default is Tuple

    """
    deltas(::Type{Vector}, adims::Dimensions) = [d.dx for d in adims.dims]
    deltas(::Type{T}, adims::Dimensions) where T = T(deltas(Vector, adims))

    deltas(adims::Dimensions) = deltas(Tuple, adims)

    """
        deltas_prod(adims::Dimensions)

    returns the product of the dx values in the array.  Usually this is for an integration measure.
    """
    deltas_prod(adims::Dimensions) = prod(deltas(Vector, adims))

    """
        update_from_array!(adims::Dimensions, arr::AbstractArray)

    Update dimensions to have the correct size based based on the array
    """
    function update_from_array!(adims::Dimensions, arr::AbstractArray)

        if size(arr) != size(adims)
            error("array size $(size(arr)) does not match dimensions size $(size(adims))")
        end

        for i in eachindex(adims.dims)
            adims.dims[i].npnts = size(arr, i)
        end

        return adims
    end

    """
        value_ranges(adims::Dimensions)
        value_ranges(T::Type, adims::Dimensions)

    ranges of coordinate values for each axes

        * T::Type if present the values will be stored in a contained of this type.  Default is Tuple

    """
    value_ranges(::Type{Vector}, adims::Dimensions) = [range(dim.x0, dim.x0+dim.dx*(dim.npnts-1), step=dim.dx) for dim in adims.dims]
    value_ranges(::Type{T}, adims::Dimensions) where T = T(value_ranges(Vector, adims))
    value_ranges(adims::Dimensions) = value_ranges(Tuple, adims)

    """
        coord_ranges(adims::Dimensions)
        coord_ranges(T::Type, adims::Dimensions)

    ranges of index vectors for each axes

        * T::Type if present the values will be stored in a contained of this type.  Default is Tuple

    """
    coord_ranges(::Type{Vector}, adims::Dimensions) = [range(1, dim.npnts) for dim in adims.dims]
    coord_ranges(::Type{T}, adims::Dimensions) where T = T(coord_ranges(Vector, adims))
    coord_ranges(adims::Dimensions) = coord_ranges(Tuple, adims)

    #
    # Tools to reshape arrays which are described be these dimensions
    # 
    """
        flatten(arr::AbstractArray, adims::Dimensions)

    Returns a view to a flattened version of the passed array.  Verifies
    that the array could be described by the dimensions
    """
    function flatten(arr::AbstractArray, adims::Dimensions)

        if size(arr) != size(adims)
            throw( DimensionMismatch("size of arr $(size(arr)) not equal to size of ad $(size(adims))") )
        end

        return reshape(arr, length(arr))

    end

    """
        reshape(arr::AbstractArray, adims::Dimensions)

    Returns a view to the passed array reshaped in accordance with the
    dimensions.  Verifies that the array could be described by the dimensions.
    """
    function reshape(arr::AbstractArray, adims::Dimensions)

        if length(arr) != length(adims)
            msg = "arr length $(length(arr)) must match dimension length = $(length(adims))"
            throw( DimensionMismatch(msg) )
        end
            
        return reshape(arr, size(adims)...)
    end

    #    
    # Tools to control the dimensions and extract information from them
    # 

    """
        valid_coods_length(adims::Dimensions, values::_VectorOrTuple{<:Any})

    helper function raises an error if the number of coords is not consistent with Dimensions
    """
    function valid_coods_length(adims::Dimensions, values::_VectorOrTuple{<:Any})
        d = ndims(adims)
        if length(values) != d
            msg = "values/coords length = $(length(values)) must match the number of dimensions = $(d)"
            throw( DimensionMismatch(msg) )
        end

        return d
    end

    """
        center!(adims::Dimensions)

    Centers the Dimensions
    """
    function center!(adims::Dimensions)

        for dim in adims.dims
            dim.x0 = -0.5*dim.dx * (dim.npnts-1)
        end

        return adims
    end

    """
        _values_to_coords_inner(dim, value, crop)

    inner loop of `values_to_coords` to allow an efficient comprehension to be written
    """
    function _values_to_coords_inner(dim, value, crop)
        if dim.dx == 0
            throw(DivideError("Invalid dx = 0 giving division by zero") )
        end
        
        coord = round(Int, (value - dim.x0)/dim.dx)

        if crop
            if dim.periodic
                coord = mod(coord, dim.npnts)
            else
                if coord < 0
                    coord = 0
                elseif coord > dim.npnts-1
                    coord = dim.npnts-1
                end
            end
        end

        return coord + 1 # Julia has 1-based indexing
    end

    """
        values_to_coords(adims::Dimensions, values::_VectorOrTuple{<:Number}; crop=true)
        values_to_coords(T::Type, adims::Dimensions, values::_VectorOrTuple{<:Number}; crop=true)

    returns the integer index with scaled value closest to x using
    the list of dimensions
    
        * crop:
            true : crop the return to the range `[1, npnts]`
            false : no cropping or wrapping.

        * T::Type if present the values will be stored in a contained of this type.  Default is Tuple
    """
    function values_to_coords(::Type{Vector}, adims::Dimensions, values::_VectorOrTuple{<:Number}; crop=true)

        valid_coods_length(adims, values)

        return [_values_to_coords_inner(dim, value, crop) for (value, dim) in zip(values, adims.dims)]
    end
    values_to_coords(::Type{T}, adims::Dimensions, values::_VectorOrTuple{<:Number}; kwargs...) where  T = T(values_to_coords(Vector, adims, values; kwargs...))

    values_to_coords(adims, values; kwargs...) = values_to_coords(Tuple, adims, values; kwargs...)

    """
        coords_to_values(adims::Dimensions, coords::_VectorOrTuple{<:Integer})
        coords_to_values(T::Type, adims::Dimensions, coords::_VectorOrTuple{<:Integer})

    returns the scaled value associated with the i,j,k, ... coords, where we 
    are finding a scaled value for each of the dimensions

        * T::Type if present the values will be stored in a contained of this type.  Default is Tuple

    """
    function coords_to_values(::Type{Vector}, adims::Dimensions, coords::_VectorOrTuple{<:Integer})
        
        valid_coods_length(adims, coords)

        # Julia has 1-based indexing
        return [dim.x0 .+ dim.dx .* (coord-1) for (dim, coord) in zip(adims.dims, coords)]
    end
    coords_to_values(::Type{T}, adims::Dimensions, coords::_VectorOrTuple{<:Integer}) where T = T(coords_to_values(Vector, adims, coords))

    coords_to_values(adims, coords) = coords_to_values(Tuple, adims, coords)

    """
        valid_coords(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

    Checks to see if Coords in a valid coord residing in the bounds
    defined by Dimensions

    If the boundary conditions are periodic, all coords are valid.

    """
    function valid_coords(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

        valid_coods_length(adims, coords)

        for (dim, coord) in zip(adims.dims, coords)

            if (!dim.periodic) && ((coord < 1) || (coord > dim.npnts))
                return false
            end
        end
            
        return true
    end

    """
        coords_to_index(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

    Given the physical i,j,k, ... coordinates, return the 1D index associated with this.
    """
    function coords_to_index(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

        valid_coods_length(adims, coords)
        
        # perform wrap-around for parodic boundary conditions case
        coords = [(dim.periodic ? mod(coord, Base.OneTo(dim.npnts)) : coord) for (coord, dim) in zip(coords, adims.dims)]

        return LinearIndices(adims)[coords...] 
    end
    """
        index_to_coords(adims::Dimensions, i::Integer; Values=False)
        index_to_coords(T::Type, adims::Dimensions, Index::<:Integer; Values=False)

    Given the index for a point in a vector, return the integer index along all dimensions directions 
    from that index.

        * i::Integer if present return only the coordinates for the i-th index.
        * T::Type if present the coordinates will be stored in a contained of this type.  Default is Tuple.

    """
    # Cannot convert CartesianIndices directly to vector
    index_to_coords(::Type{Tuple}, adims::Dimensions) = [Tuple(coord) for coord in CartesianIndices(adims)][:] 
    index_to_coords(::Type{Vector}, adims::Dimensions) = [convert(Vector, coord) for coord in index_to_coords(Tuple, adims)]
    index_to_coords(::Type{T}, adims::Dimensions) where T = [T(coord) for coord in index_to_coords(Vector, adims)]

    index_to_coords(adims::Dimensions) = index_to_coords(Tuple, adims)

    index_to_coords(::Type{Tuple}, adims::Dimensions, i::Integer) =  Tuple(CartesianIndices(adims)[i])
    index_to_coords(::Type{Vector}, adims::Dimensions, i::Integer) =  convert(Vector, index_to_coords(Tuple, adims, i))
    index_to_coords(::Type{T}, adims::Dimensions, i::Integer) where T =  T(index_to_coords(Vector, adims, i))


    index_to_coords(adims::Dimensions, i) =  index_to_coords(Tuple, adims, i)

    """
        index_to_values(adims::Dimensions, Index::Integer; Values=False)

    Given the index for a point in a vector, return the physical coordinates of that position.

        * i::Integer if present return only the values for the i-th index.
        * T::Type if present the values will be stored in a contained of this type.  Default is Tuple.
    """
    index_to_values(::Type{T}, adims::Dimensions) where T = [coords_to_values(T, adims, c) for c in index_to_coords(adims)]
    index_to_values(adims::Dimensions) = index_to_values(Tuple, adims)

    index_to_values(::Type{T}, adims::Dimensions, i::Integer) where T = coords_to_values(T, adims, index_to_coords(adims, i))
    index_to_values(adims::Dimensions, i::Integer) = index_to_values(Tuple, adims, i)

    # #############################################################################
    #
    # End of ArrayDimensions
    #
    # #############################################################################

    # #############################################################################
    #
    # Supporting functions for range utilities
    #
    # #############################################################################

    """
        NDRange
    
    A multidimensional range iterator with behavior similar to meshgrid.  This is almost implemented
    by Base.product, but this is not a subtype of AbstractArray, so it does not behave as one would expect
    """
    struct NDRange{dim} <: AbstractArray{Any, dim}
        ranges::Tuple{Vararg{<:AbstractVector, dim}}
        function NDRange(ranges)
            len = length(ranges)
            if len == 0
                len = 1
                ranges = tuple([])
            end
            new{len}(ranges)
        end
    end
    NDRange(ranges...) = NDRange{length(ranges)}(ranges)
    NDRange() = NDRange([]) # default setup for no range

    # Overload methods needed for AbstractArray

    size(ndrange::NDRange) = Tuple(length(x) for x in ndrange.ranges)

    size(ndrange::NDRange, d) = d::Integer <= length(ndrange.ranges) ? length(ndrange.ranges[d]) : 1

    # we might be passed a single index in which case we convert to a cartesian index
    function getindex(ndrange::NDRange, i::Integer) 
        ci = Base.CartesianIndices(ndrange)[i]
        return getindex(ndrange, ci)
    end
    function getindex(ndrange::NDRange, i1::Union{Integer, Base.CartesianIndex}, I::Union{Integer, Base.CartesianIndex}...)
        indices = Base.to_indices(ndrange, (i1, I...))
        return Tuple(x[i] for (x,i) in zip(ndrange.ranges, indices))
    end

    length(ndrange::NDRange) = prod(size(ndrange))

    ndims(ndrange::NDRange{dim}) where dim = dim::Integer

    show(io::IO, ndrange::NDRange) = print(io, ndrange.ranges)

    +(ndrange::NDRange, x::Number) = NDRange(Tuple(y + x for y in ndrange.ranges))
    +(x::Number, ndrange::NDRange) = +(ndrange::NDRange, y)

    *(ndrange::NDRange, x::Number) = NDRange(Tuple(y * x for y in ndrange.ranges))
    *(x::Number, ndrange::NDRange) = *(ndrange::NDRange, y)

    /(ndrange::NDRange, x::Number) = NDRange(Tuple(y / x for y in ndrange.ranges))
    /(x::Number, ndrange::NDRange) = NDRange(Tuple(x / y for y in ndrange.ranges))

    """
    Returns an tuple with length ndims(NDRange) filled with the contents of the range

    The Julia forums claimed that using iterators made meshgrid un-needed, but my experience was that the 
    performance suffered 100x fold.
    """
    meshgrid(ndrange::NDRange) =  Tuple([x[j[i]] for j in CartesianIndices(ndrange)] for (i, x) in enumerate(ndrange.ranges) )


end # ArrayDimensions