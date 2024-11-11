module QGas
    export FileIO
    export AtomicPhysics    
    export AtomicConstants 
    export NumericalTools    
    export Simulation    

    include("AtomicPhysics.jl")

    include("AtomicConstants.jl")

    include("NumericalTools/NumericalTools.jl")

    include("Simulation/Simulation.jl")

    include("FileIO.jl")

    # Check to see if this is being run from the command line
    if abspath(PROGRAM_FILE) == @__FILE__
        print("Running from command line")
    end
end
