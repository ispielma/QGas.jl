"""
Analysis

This module is as grab-bag of general analysis tools
"""
module Analysis

    """
    Create a binned average

    if normalize is true, then the average is normalized.  If not we are looking at a binned
        sum.

    if uniform is true then the bins are uniformly spaced, otherwise we also average the x values within each bin.  It turned out that
    for some distributions this was the best choice (rather than simply making a grid)

    returns: (avg, counts, xvals, uncertanties)
    """
    make_avgs(bins, data_x, data_y; kwargs...) = make_avgs(bins, data_x, data_y, extrema(data_x)...; kwargs...)

    function make_avgs(bins::Integer, data_x::AbstractArray{T}, data_y, min::T, max::T; normalize=true, uniform=false) where T
        # Convert to 1D arrays if needed
        data_x = vec(data_x)
        data_y = vec(data_y)

        counts = zeros(Int64, bins)
        avg = zeros(Float64, bins)
        avg2 = zeros(Float64, bins)
        xvals = zeros(Float64, bins)

        dx = (max - min) / bins


        for i in eachindex(data_x, data_y)
            if data_x[i] > min && data_x[i] < max
                idx = Int64(ceil( (data_x[i] - min) / dx))
                idx = idx == 0 ? 1 : idx # shift first point into the box

                counts[idx] += 1
                avg[idx] += data_y[i]
                avg2[idx] += data_y[i].^2
                xvals[idx] += data_x[i]
            end
            
        end

        for i in eachindex(counts, avg, xvals)
            if counts[i] == 0
                avg[i] = 0
                avg2[i] = 0
                xvals[i] = NaN
            else
                if normalize
                    avg[i] /= counts[i]
                    avg2[i] /= counts[i]
                end
                xvals[i] /= counts[i]
            end
        end

        if uniform # replace xvals with the center of the boxes.
            # the left edge of the first box is at min and the right edge of the last box is at max
            xvals = range(min + dx/2, max-dx/2, bins)
        end

        if normalize
            sigma = sqrt.(avg2 .- avg.^2)
        else
            sigma = missing
        end

        avg, counts, xvals, sigma
    end

    """
    Create a histogram average

    Notice that we also average the x values within each bin.  It turned out that
    for some distrubtions this was the best choice (rather than simply making a grid)
    """
    histogram(bins, data) = histogram(bins, data, extrema(data)...)

    function histogram(bins, data, min, max)
        counts = zeros(Int64, bins)
        xvals = zeros(Float64, bins)

        dx = (max - min) / bins

        for i in eachindex(data)
            if data[i] > min && data[i] < max
                idx = Int64(ceil( (data[i] - min) / dx))
                idx = idx == 0 ? 1 : idx

                counts[idx] += 1
                xvals[idx] += data[i]
            end
        end

        for i in eachindex(counts, xvals)
            if counts[i] == 0
                xvals[i] = NaN
            else
                xvals[i] /= counts[i]
            end
        end

        xvals, counts 
    end

end