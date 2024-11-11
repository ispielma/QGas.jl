"""
Functions

This module contains special functions that I have written over the years
"""
module Functions

    import SpecialFunctions as SF
    """
    HermiteGauss

    This struct creates a callable order n HermiteGauss function with the call signature
    
    HermiteGauss(x, coefs)
        where coefs are the n coefficients and x is the horizontal coordinate in units of the gaussian width
    """

    struct HermiteGauss
        n::Integer
        prefactors::Vector{Float64}
    end

    function HermiteGauss(n)

        # Generate the prefactors # not complete
        prefactors = [1 for i = 0:n] ./ ( (2^n * factorial(n))^0.5 * (pi^0.25)) 

        # function HermiteGauss(n, prefactors)
    end

    # (f::HermiteGauss)(x, coefs) = sum([coefs[i]*f.prefactors[i]*x.^(i-1) for i in eachindex(coefs, f.prefactors)])

end