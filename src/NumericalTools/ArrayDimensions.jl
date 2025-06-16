"""
    ArrayDimensions

A module to deal with the basic information needed to create and maintain 
dimensioning information in a simple way

One confusing language thing:
    values: the actual variables to be scaled 

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

    # Import the functions that I plan to extend
    import Base: ndims, size, length, copy, reshape, getindex, setindex!, show, +, *, /, ^
    import Base: CartesianIndices, LinearIndices

    """
    Dimension

    defines the dimensional properties of an axis of an N-dimensional array
    """ 
    Base.@kwdef mutable struct Dimension
        x0::Float64 = 0.0
        dx::Float64 = 1.0
        npnts::Int64 = 0
        unit::String = ""
        symmetric::Bool = false
        periodic::Bool = false
    end 

    Base.@kwdef mutable struct Dimensions
        dims::Vector{Dimension} = Dimension[]
    end
    Dimensions(arr::AbstractArray) = update_from_array(arr)
    Dimensions(adims::Dimensions) = copy(adims)
    Dimensions(ndims::Integer; x0=0.0, dx=1.0, symmetric=false, periodic=periodic) = [Dimension(;x0=x0, dx=dx, symmetric=symmetric, periodic=periodic) for j in 1:ndims] 

    _VectorOrTuple{T} = Union{Vector{T},Tuple{T}}

    # #############################################################################
    #
    # Overloaded builtin methods
    #
    # #############################################################################

    """
    copy

    new method for builtin copy
    """
    copy(adims::Dimensions) = Dimensions( dims=copy(adims.dims) )

    """
    ndims

    new method for builtin ndims, returns dimension of the object
    """
    ndims(adims::Dimensions) =length(adims.dims)

    """
    size

    new method for builtin size, returns array of sizes for each axis
    """
    size(adims::Dimensions) = tuple([d.npnts for d in adims.dims]...)
    size(adims::Dimensions, d::Integer) = adims.dims[d]

    """
    length(adims::Dimensions)

    new method for builtin length, returns total number of elements associated with the Dimensions
    """
    length(adims::Dimensions) = prod(size(adims))

    """
        LinearIndices(adims::Dimensions)

    new method for builtin LinearIndices for Dimensions type
    """
    LinearIndices(adims::Dimensions) = LinearIndices(CoordRanges(adims))

    """
        CartesianIndices(adims::Dimensions)

    new method for builtin CartesianIndices for Dimensions type
    """
    CartesianIndices(adims::Dimensions) =CartesianIndices(CoordRanges(adims))

    # #############################################################################
    #
    # Code ported from python
    #
    # #############################################################################

    """
        Deltas(adims::Dimensions)

    returns an array of the dx values in the array
    """
    deltas(adims::Dimensions) = [d.dx for d in adims.dims]

    """
        DeltasProd(adims::Dimensions)

    returns the product of the dx values in the array.  Usually this is for an integration measure
    """
    deltas_prod(adims::Dimensions) = prod(Deltas(adims))

    """
        update_from_dims!(adims::Dimensions, dims::Dimension)

    Update dimensions from dims to have the correct size
    """
    function update_from_dims!(adims::Dimensions, dims::Dimension)
        adims.dims = copy(dims)
        return adims
    end

    """
        update_from_array(arr::AbstractArray)

    Update dimensions from array to have the correct size based on a template array
    """
    update_from_array(arr::AbstractArray) = [Dimension(;npnts=x) for x in size(arr)]

    """
        update_from_array!(adims::Dimensions, arr::AbstractArray)

    Update dimensions from array to have the correct size
    """
    function update_from_array!(adims::Dimensions, arr::AbstractArray)

        adims.dims = update_from_array(arr)
        return adims
    end

    """
        value_ranges(adims::Dimensions)

    ranges of coordinate values for each axes
    """
    value_ranges(adims::Dimensions) = tuple(range(dim.x0, dim.x0+dim.dx*(dim.npnts-1), step=dim.dx) for dim in adims.dims)

    """
        coord_ranges(adims::Dimensions)

    ranges of index vectors for each axes
    """
    coord_ranges(adims::Dimensions) = tuple(range(1, dim.npnts) for dim in adims.dims)

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
    dimensions.  Verifies that the array could be described by the
    dimensions.
    """
    function reshape(arr::AbstractArray, adims::Dimensions)

        if length(arr) != length(adims)
            msg = "arr length $(length(arr)) must match dimension length = $(length(adims))"
            throw( DimensionMismatch(msg) )
        end
            
        return reshape(arr, size(arr)...)
    end

    # I am not making the range arrays function because "apparently" it is not needed in
    # julia.  I will see.

    #
    # Tools to control the dimensions and extract information from them
    # 

    """
        _validcoodslength(adims::Dimensions, values::_VectorOrTuple{<:Any})

    helper function raises an error if the number of coords is not consistant with an Dimensions
    """
    function _validcoodslength(adims::Dimensions, values::_VectorOrTuple{<:Any})
        if length(values) != ndims(adims)
            msg = "values/coords length = $(length(values)) must match the number of dimensions = $(ndims(adims))"
            throw( DimensionMismatch(msg) )
        end
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
        values_to_coords(adims::Dimensions, values::_VectorOrTuple{<:Number}; crop=true)

    returns the integer index with scaled value closest to x using
    the list of dimensions
    crop:
        True directs the program to crop the return to the range
        [1,npnts], axes that are periodic wrap-around instead
        
        False directs no cropping        
    """
    function values_to_coords(adims::Dimensions, values::_VectorOrTuple{<:Number}; crop=true)

        _validcoodslength(adims, values)

        coords = zeros(Int, ndims(adims))
        for (i, (value, dim)) in enumerate(zip(values, adims.dims))

            if dim.dx == 0
                throw(DivideError("Invalid dx = 0 giving division by zero") )
            end
            
            coord = round(Int, (value - dim.x0)/dim.dx)

            if crop
                if dim.periodic
                    coord = mod(cord, dim.npnts)
                else
                    if coord < 0
                        coord = 0
                    elseif coord > dim.npnts-1
                        coord = dim.npnts-1
                    end
                end
            end

            coords[i] = coord + 1 # Recall that julia has 1-based indexing
        end

        return coords
    end

    """
        coords_to_values(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

    returns the scaled value associated with the indices, where we 
    are finding a scaled value for each of the dimensions
    """
    function coords_to_values(adims::Dimensions, coords::_VectorOrTuple{<:Integer})
        
        _validcoodslength(adims, coords)

        return [dim.x0 + dim.dx * coord for (dim, coord) in zip(adims.dims, coords)]
    end

    """
        valid_coords(adims::Dimensions, coords::_VectorOrTuple{<:Integer})::Bool

    Checks to see if Coords in a valid coord residing in the bounds
    defined by Dimensions

    If the boundary conditions are periodic, all Coords are valid.
    """
    function valid_coords(adims::Dimensions, coords::_VectorOrTuple{<:Integer})::Bool

        _validcoodslength(adims, coords)

        for (dim, coord) in zip(adims.dims, coords)

            if (!dim.periodic) && ((coord < 1) || (coord > dim.npnts))
                return false
            end
        end
            
        return true
    end

    # LinearIndices

    """
        coords_to_index(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

    Given the physical coordinates a point in a vector,
    return the vector index associated with this using
    the information provided in Dimensions
    """
    function coords_to_index(adims::Dimensions, coords::_VectorOrTuple{<:Integer})

        _validcoodslength(adims, coords)
        
        # perform wrap-around for perodic boundary conditions case
        coordsperiodic = zeros(Int, length(coords))
        for i in 1:length(coords)
            r = Base.OneTo(adims.dimsArray[i])
            if adims.dims[i].periodic
                coordsperiodic[i] = mod(coords[i], Base.OneTo(adims.dimsArray[i]))
            end
        end

        coordsperiodic = [(dim.periodic ? mod(coord, Base.OneTo(dim.npnts)) : coord) 
                            for (coord, dim) in zip(coords, adims.dims)]

        return LinearIndices(adims)[coordsperiodic...] 
    end
    """
        index_to_coords(adims::Dimensions, Index::<:Integer; Values=False)

    Given the index for a point in a vector, return the integer index along all dimensions directions 
    from that index.  Depending on how this is used, I might want to just used the overloaded CartesianIndices
    that I have introduced
    """
    index_to_coords(adims::Dimensions, Index::Integer) = Tuple(CartesianIndices(adims)[Index])

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