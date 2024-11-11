"""
ImageProcessing

This module is as grab-bag of tools ported from python for image analysis, typically of
cold atom data
"""
module ImageProcessing

    import LinearAlgebra as LA
    import FFTW
    import Distributions as Dist
    import Statistics as Stat
    import PaddedViews
    using DataFrames



    #=
     ######  ######## ########  ##     ##  ######  ########  ######
    ##    ##    ##    ##     ## ##     ## ##    ##    ##    ##    ##
    ##          ##    ##     ## ##     ## ##          ##    ##
     ######     ##    ########  ##     ## ##          ##     ######
          ##    ##    ##   ##   ##     ## ##          ##          ##
    ##    ##    ##    ##    ##  ##     ## ##    ##    ##    ##    ##
     ######     ##    ##     ##  #######   ######     ##     ######
    =#
 
    #=
       ##      ## #### ##    ## ########   #######  ##      ##  ######
       ##  ##  ##  ##  ###   ## ##     ## ##     ## ##  ##  ## ##    ##
       ##  ##  ##  ##  ####  ## ##     ## ##     ## ##  ##  ## ##
       ##  ##  ##  ##  ## ## ## ##     ## ##     ## ##  ##  ##  ######
       ##  ##  ##  ##  ##  #### ##     ## ##     ## ##  ##  ##       ##
       ##  ##  ##  ##  ##   ### ##     ## ##     ## ##  ##  ## ##    ##
        ###  ###  #### ##    ## ########   #######   ###  ###   ######
    =#

    # Here I have a series of distributions and windows.  The key distinction between the 
    # two functions are that windows have a peak value of 1 (i.e. inside the window), while
    # distributions are normalized to 1

    # Plan: for each item the "base" function will be the distribution as it will be the most
    # simple to express, and the normed version will be a different function that calls that 
    # and then appends the norm.
    # I will be careful to minimize the number of times the norm is computed in the case of vector
    # operations (i.e don't comput it inside the scalar method if it is called from the vector method)

    """
        shift_window(f, min, max)

    shifts a function that has a domain [0,1] to be in the domain [min,max]
    """
    shift_window(f, min, max) = (max-min).*f .+ min

    """
        tukey_window(f, a)

    f : defines the quantity on which the window is evaluated where the window cuts off
        for |f| ≥ 1/2.  Note that often the tukey is defined instead from 0 to 1 rather
        than -1/2 to 1/2

    a : tukey window parameter with 0 being a box window and 1 being a cos window

    w : x axis width.  I use this for radial tukeys that make a window starting 
        at zero and naturally ending at 1.  In this case one would set width = 2

    """
    function tukey_window(f, a)
        f = abs(f)
        if f > 0.5
            window = 0.0
        elseif f < 0.5*(1 - a) 
            window = 1.0
        else
            window = ( 1-cos(pi*(1-2*f)/a) ) / 2.0
        end

        return window
    end 

    tukey_window(f, a, w) = tukey_window.(f./w, a)
    tukey_window(f, a, w, cen) = tukey_window(f .- cen, a, w)
    
    # Multi dimension methods
    tukey_window(fs::Tuple{Vararg{T, dim}}, a) where {T, dim} = tukey_window(sqrt(sum(fs.^2)), a)
    tukey_window(fs::Tuple{Vararg{T, dim}}, a, w::Number) where {T, dim} = tukey_window(fs ./ w, a)
    tukey_window(fs::Tuple{Vararg{T, dim}}, a, ws) where {T, dim} = tukey_window(Tuple( f / w for (f, w) in zip(fs, ws)), a)
    tukey_window(fs::Tuple{Vararg{T, dim}}, a, ws, cens) where {T, dim} = tukey_window(Tuple( f - c for (f, c) in zip(fs, cens)), a, ws)

    # version for many points
    tukey_window(f::AbstractArray, args...) = [tukey_window(f[i], args...) for i in CartesianIndices(f) ]


    """
        gauss_window(w, a)

    f : defines the quantity on which the window is evaluated.

    w : gaussian width

    """

    gauss_window(f::Number) = exp(-0.5*f^2 )

    gauss_window(f, w) = gauss_window(f./w)
    gauss_window(f, w, cen) = gauss_window(f-cen, w)
    
    # Multi dimension methods
    gauss_window(fs::Tuple{Vararg{T, dim}}) where {T, dim} = gauss_window(sqrt(sum(fs.^2)))
    gauss_window(fs::Tuple{Vararg{T, dim}}, w::Number) where {T, dim} = gauss_window(fs ./ w)
    gauss_window(fs::Tuple{Vararg{T, dim}}, ws) where {T, dim} = gauss_window(Tuple( f / w for (f, w) in zip(fs, ws)))
    gauss_window(fs::Tuple{Vararg{T, dim}}, ws, cens) where {T, dim} = gauss_window(Tuple( f - c for (f, c) in zip(fs, cens)), ws)

    # version for many points
    gauss_window(f::AbstractArray, args...) = [gauss_window(f[i], args...) for i in CartesianIndices(f) ]

    """
    MaskConfig

    A struct with a constructor that creates one of the known mask/window types

    shift (bool): if we apply fftshift to the window (useful if it will be applied after
        a FFT)
    """
    struct WindowConfig
        type::String
        radius::Vector{Float64}
        center::Vector{Float64}
        power::Float64
        tukey_parameter::Float64
        window::Array{Float64}
    end
    function WindowConfig(grids, type, radius, center, power, tukey_parameter, invert; shift=false)
        dims = size(grids[1])
        
        # Define a scaled radial array
        grd2 = zeros(dims)
        for (grd, c, r) in zip(grids, center, radius)
            grd2 .+= (abs.(grd .- c)./r).^power
        end

        # Now convert this to a mask
        # Use whatever window function you want here, but for standard windows    
        if type=="Tukey"
            grd2 = grd2.^(1/power)
            window = tukey_window(grd2, tukey_parameter, 2)
        elseif type=="SuperGauss"
            window = exp.(-grd2.^tukey_parameter)
        elseif type=="Gauss" # gauss
            window = exp.(-0.5.*grd2)
        else
            window = ones(dims)
        end
    
        if invert
            window = 1 .- window
        end
        
        if shift
            window = FFTW.fftshift(window)
        end

        return WindowConfig(type, radius, center, power, tukey_parameter, window)
    end
    
    #=
       ########  #######   #######  ##        ######
          ##    ##     ## ##     ## ##       ##    ##
          ##    ##     ## ##     ## ##       ##
          ##    ##     ## ##     ## ##        ######
          ##    ##     ## ##     ## ##             ##
          ##    ##     ## ##     ## ##       ##    ##
          ##     #######   #######  ########  ######
    =#

    """
    downsample_array 

    Downsamples an array by a specified factor, will average elements

    This is implemented with CartesianIndices math, and there may be a better way to do this
    """
    downsample_array(arr::Missing, factor) = missing
    function downsample_array(arr::AbstractArray{M, dim}, factor::Tuple{Vararg{Int, dim}}) where {M, dim}

        if all(factor .== 1) # No change needed of scale is 1
            return arr
        elseif any(factor .< 1)
            error("scale $(factor) must be positive")
        end
    
        # Initilize downsampled array
        down_size = div.(size(arr), factor)
        array_size = down_size .* factor # size of intial array, but scaled if factor is not commensurate.
        arr_down = zeros(M, down_size)
    
        # Downsample
    
        # Make an array of the required ranges with steps
        ranges = Tuple(1:skip:step for (skip, step) in zip(factor, array_size) )
        skip_indices = CartesianIndices(ranges)
    
        ranges = Tuple(1:step for step in down_size )
        final_indices = CartesianIndices(ranges)
        
        step_index = CartesianIndex(factor .- 1)
    
    
        return _downsample_array_innerloop(arr, arr_down, final_indices, skip_indices, step_index)
    end

    """
    Wrap inner loop for type stability.
    """
    function _downsample_array_innerloop(arr, arr_down, final_indices, skip_indices, step_index)

        for (i, i_skip) in zip(final_indices, skip_indices)
            s = zero(eltype(arr))
            n = 0
            for j in i_skip:(i_skip .+ step_index)
                s += arr[j]
                n += 1
            end
    
            arr_down[i] = s / n
        end

        return arr_down
    end

    """
        conv_with_weights(data, w, f)
    
    computes the convilution of a dataset with a function including the weight factors w.  The norm of data will be 
        conserved.

    data: array to be convolved
    w : weight factor (equal to 1/σ, with uncertainty σ)
    f : filter array (equal in shape to data), does not need to be normalized 
    """
    function conv_with_weights(data::AbstractArray, w::AbstractArray, f::AbstractArray)

        f_fft = FFTW.rfft(f)


        # Compute numerator
        d = size(data)[1] # needed for irfft
        ans = FFTW.irfft( FFTW.rfft(data .* w.^2) .* f_fft, d)

        # Compute denominator
        ans ./= FFTW.irfft( FFTW.rfft(w.^2) .* f_fft, d)

        return ans
    end

    """
    I think that "bad" is equal to 1 where the data is good!
    """
    function fix_bad!(data, bad, filter; pad=true)

        initial_size = size(data)
    
        if pad
            pad_data = PaddedViews.PaddedView(0, data, initial_size .* 2)
            pad_bad = PaddedViews.PaddedView(0, bad, initial_size .* 2)
    
            # This one needs padded in such a way that FFTshift will work
            pad_filter = PaddedViews.PaddedView(
                0,
                filter.window,
                initial_size .* 2,
                initial_size .÷ 2 .+ 1
            )
        else
            pad_data = data
            pad_bad = bad
            pad_filter = filter
        end
    
        smooth = conv_with_weights(pad_data, pad_bad, FFTW.fftshift(pad_filter))
    
        if pad
            smooth = smooth[1:initial_size[1], 1:initial_size[2]]
        end
    
        # got the desired smoothed array!
        data .= data .* bad + smooth .* (1 .- bad)
    end

    function PSD(Field; shift=false)
        psd = abs.(FFTW.rfft(Field)).^2
    
        if shift
            psd = FFTW.fftshift(psd)
        end
    
        return psd
    end

    #=
        ######   #######  ##       ########        ###    ########  #######  ##     ##  ######
       ##    ## ##     ## ##       ##     ##      ## ##      ##    ##     ## ###   ### ##    ##
       ##       ##     ## ##       ##     ##     ##   ##     ##    ##     ## #### #### ##
       ##       ##     ## ##       ##     ##    ##     ##    ##    ##     ## ## ### ##  ######
       ##       ##     ## ##       ##     ##    #########    ##    ##     ## ##     ##       ##
       ##    ## ##     ## ##       ##     ##    ##     ##    ##    ##     ## ##     ## ##    ##
        ######   #######  ######## ########     ##     ##    ##     #######  ##     ##  ######
    =#

    """
    preprocess_probe(probe, dark, filter)

    preprocesses a probe beam

    probe : probe image
    dark : dark image
    filter : the fancy version convolves the data with filter to make a smoothed version of the data (weighted by the uncertanties)
        and fills in the invalid (negative) data with the filtered data.
    """
    function preprocess_probe(probe::Array{Float64}, dark::Array{Float64})
        probe -= dark

        return probe
    end
    # These methods convert to Float64 only if needed
    preprocess_probe(probe::Array{Float64}, dark::AbstractArray) = preprocess_probe(probe, convert(Array{Float64}, dark))
    preprocess_probe(probe::AbstractArray, dark::Array{Float64}) = preprocess_probe(convert(Array{Float64}, probe), dark)
    preprocess_probe(probe::AbstractArray, dark::AbstractArray) = preprocess_probe(convert(Array{Float64}, probe), convert(Array{Float64}, dark))

    function filter_probe(probe, filter)

        weights = [p <= 0.0 ? 0.0 : sqrt(p) for p in probe]
        probe_smooth = conv_with_weights(probe, weights, filter)
        mask = weights .== 0
        probe[mask] .= probe_smooth[mask]

        return probe
    end

    preprocess_probe(probe, dark, filter::AbstractArray) = filter_probe(preprocess_probe(probe, dark), filter)

    """
        average_dark(darks)

    return the average of the dark frames as well as the average noise
    """
    function average_dark(darks::Array{Float64}; remove=[])
        dim = ndims(darks)
        mean = Stat.mean(darks; dims=dim)
        std = Stat.std(darks; mean=mean, dims=dim)

        # For some reason mean and std do not reduce the dimension of the array and have
        # a 1 sized dimension on the last index
        mean = reshape(mean, size(mean)[1:(end-1)] )
        std =  reshape(std, size(std)[1:(end-1)] )

        for image in remove
            image .-= mean
        end

        return mean, std
    end
    average_dark(darks::AbstractArray) = average_dark(convert(Array{Float64}, darks))

    function average_dark(data::AbstractDataFrame, dark_key; remove=[])
        mean = Stat.mean(data[!, dark_key])
        std = Stat.std(data[!, dark_key]; mean=mean)

        for k in remove
            for d in data[!, k] 
                d .-= mean
            end
        end

        return mean, std
    end

    r"""
        probe_basis(probes)

    construct a PCA/SVD basis from a set of probe images.  

    Currently always does SVD in the way that assumes that the number of probe images is small
    as compared to the number of pixels $N$ in each images.  For the opposite approach one would
    make the covariance matrix be $N\times N$.
    """
    function probe_basis(probes::Array{Float64})
        probes_flat = reshape(probes, :, size(probes)[end])

        factor = LA.svd(probes_flat);

        # Get eigenvalues
        pvs = factor.S .^2
        pvs ./= sum(pvs)

        return reshape(factor.U, size(probes)[1:end-1]..., :), pvs
    end
    probe_basis(probes::AbstractArray) = probe_basis(convert(Array{Float64}, probes))

#=
             #######  ######## ##     ## ######## ########     ########  #######  ##    ##  ######
            ##     ##    ##    ##     ## ##       ##     ##       ##    ##     ##  ##  ##  ##    ##
            ##     ##    ##    ##     ## ##       ##     ##       ##    ##     ##   ####   ##
            ##     ##    ##    ######### ######   ########        ##    ##     ##    ##     ######
            ##     ##    ##    ##     ## ##       ##   ##         ##    ##     ##    ##          ##
            ##     ##    ##    ##     ## ##       ##    ##        ##    ##     ##    ##    ##    ##
             #######     ##    ##     ## ######## ##     ##       ##     #######     ##     ######
=#
    """
    α_realization(I, I_bar, ΔI2)

    Generates a single realization of a field consistant with:

    I   The observed intensity
    I_bar   The known mean intensity
    ΔI2     The pixel-by-pixel variance

    """
    function α_realization(I, I_bar, ΔI2; addnoise=false)
        κ_min = 0.01

        # Field variance
        Δα = sqrt.(2 .* I_bar .* ((1 .+ ΔI2 ./ (4 .* I_bar.^2)).^0.5 .- 1))

        α = sqrt.(I_bar) .+ 0im
        α .+= (I - I_bar) ./ (2 .* α)
        
        # Now add noise  
        if addnoise
            α .+= 1.0im * Dist.rand.(Dist.Normal.(0, Δα./2))
        end

        return α
    end
end