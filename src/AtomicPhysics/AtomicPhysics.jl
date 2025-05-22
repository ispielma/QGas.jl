"""
Created on Sat Dec  17 2022

This package provides Constants and functions required for atomic physics experiments.

@author: ispielma

Copyright: Copyright 2005, Ian B. Spielman

History: 
    Version 0.1 (Mathematica) Ian Spielman (2005) 
    Version 0.2 (Mathematica) Ian Spielman (2008) : Added Lithium constants
    Version 0.3 (Mathematica) Ian Spielman (2009) : Added Potassium constants
    Version 1.0 (Python) Ian Spielman (2014): Ported to Python
    Version 2.0 (Julia) Ian Spielman (2022): Ported to Julia
"""

module AtomicPhysics

    export AtomicConstants
    export BreitRabi

    include("AtomicConstants.jl")
    include("BreitRabi.jl")

end # module AtomicPhysics