"""
Analysis

This module is as grab-bag of general analysis tools
"""
module Analysis

    """
    Create a binned average

    Notice that we also average the x values within each bin.  It turned out that
    for some distrubtions this was the best choice (rather than simply making a grid)
    """
    function make_avgs(bins, data_x, data_y)
        min, max = extrema(data_x)
        counts = zeros(Int64, bins)
        avg = zeros(Float64, bins)
        xvals = zeros(Float64, bins)

        dx = (max - min) / bins


        for i in eachindex(data_x, data_y)
            idx = Int64(ceil( (data_x[i] - min) / dx))
            idx = idx == 0 ? 1 : idx

            counts[idx] += 1
            avg[idx] += data_y[i]
            xvals[idx] += data_x[i]
            
        end

        for i in eachindex(counts, avg, xvals)
            if counts[i] == 0
                avg[i] = 0
                xvals[i] = NaN
            else
                avg[i] /= counts[i]
                xvals[i] /= counts[i]
            end
        end

        avg, counts, xvals
    end

    """
    Create a histogram average

    Notice that we also average the x values within each bin.  It turned out that
    for some distrubtions this was the best choice (rather than simply making a grid)
    """
    function histogram(bins, data)
        min, max = extrema(data)
        counts = zeros(Int64, bins)
        avg = zeros(Float64, bins)
        xvals = zeros(Float64, bins)

        dx = (max - min) / bins

        for i in eachindex(data)
            idx = Int64(ceil( (data[i] - min) / dx))
            idx = idx == 0 ? 1 : idx

            counts[idx] += 1
            xvals[idx] += data[i]
            
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