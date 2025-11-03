module FileIO

import ..NumericalTools.ImageProcessing as ImProc

import HDF5
import Format
import LazyGrids
import FFTW


"""
    A helper function that returns a range that always includes zero.  Note that I need to do an even / odd check
"""
_range_with_zero(npnt::Integer, dx) = (npnt % 2 == 0 ? range(-dx*(npnt)/2, dx*(npnt-2)/2, length=npnt) : range(-dx*(npnt-1)/2, dx*(npnt-1)/2, length=npnt) )
   

#=
#### ##     ##    ###     ######   ########    ########  ########   #######  ########  ######## ########  ######## #### ########  ######
 ##  ###   ###   ## ##   ##    ##  ##          ##     ## ##     ## ##     ## ##     ## ##       ##     ##    ##     ##  ##       ##    ##
 ##  #### ####  ##   ##  ##        ##          ##     ## ##     ## ##     ## ##     ## ##       ##     ##    ##     ##  ##       ##
 ##  ## ### ## ##     ## ##   #### ######      ########  ########  ##     ## ########  ######   ########     ##     ##  ######    ######
 ##  ##     ## ######### ##    ##  ##          ##        ##   ##   ##     ## ##        ##       ##   ##      ##     ##  ##             ##
 ##  ##     ## ##     ## ##    ##  ##          ##        ##    ##  ##     ## ##        ##       ##    ##     ##     ##  ##       ##    ##
#### ##     ## ##     ##  ######   ########    ##        ##     ##  #######  ##        ######## ##     ##    ##    #### ########  ######
=#

"""
    CameraInfo

Describes the physical properties of an imaging system as well as a 
downsampling factor.
"""
struct CameraInfo{dim}
    sensor_size::Tuple{Vararg{Int, dim}}
    pixel_size::Tuple{Vararg{Float64, dim}}
    magnification::Tuple{Vararg{Float64, dim}}
    downsample::Tuple{Vararg{Int, dim}}
    units::Tuple{Vararg{String, dim}}
    datatype::Type # Type data will be cast to upon loading
    function CameraInfo(sensor_size, pixel_size, magnification, downsample, units, datatype)

        # Convert sensor size to a Tuple a
        sensor_size = Tuple(Int(j) for j in sensor_size)
        dim = length(sensor_size)

        # Expand singletons to tuples if needed
        pixel_size = _tuple_length_fix(pixel_size, dim)
        magnification = _tuple_length_fix(magnification, dim)
        downsample = _tuple_length_fix(downsample, dim)
        units = _tuple_length_fix(units, dim; exclude=AbstractString)

        new{dim}(sensor_size, pixel_size, magnification, downsample, units, datatype)
    end
end
CameraInfo(sensor_size, pixel_size, magnification, downsample, units) = CameraInfo(sensor_size, pixel_size, magnification, downsample, units, Float64)
CameraInfo(sensor_size, pixel_size, magnification, units, datatype::Type) = CameraInfo(sensor_size, pixel_size, magnification, 1, units, datatype)
CameraInfo(sensor_size, pixel_size, magnification, units) = CameraInfo(sensor_size, pixel_size, magnification, 1, units, Float64)

# This is designed for when we crop
CameraInfo(camera_info::CameraInfo, sensor_size) = CameraInfo(sensor_size .* camera_info.downsample, camera_info.pixel_size, camera_info.magnification, camera_info.downsample, camera_info.units, camera_info.datatype)

"""
_length_fix

Helper function to help easily convert singletons to tuples if desired

If we are of type "exclude" it is assumed that the object is a singleton
"""
function _tuple_length_fix(item, dim; exclude::Type=Nothing)
    if length(item) == 1 || typeof(item) <: exclude
        item = Tuple(item for _ in 1:dim)
    else
        if length(item) != dim
            error("length of quantity $(item) not equal to target length $(dim)")
        end
        item = Tuple(item[j] for j in 1:dim)
    end
    item
end


"""
ImageInfo

contains the most basic information commonly required to describe images
""" 
struct ImageInfo{dim}
    npnts::Tuple{Vararg{Int, dim}}
    dxs::Tuple{Vararg{Float64, dim}}
    dks::Tuple{Vararg{Float64, dim}}
    units::Tuple{Vararg{String, dim}}
    index_ranges::Tuple{Vararg{AbstractVector, dim}} # Array of 1D ranges  (to access array elements)
    ranges::Tuple{Vararg{AbstractVector, dim}} # Array of 1D ranges in real units
    ndrange::Tuple{Vararg{LazyGrids.AbstractGrid, dim}}  # Mesh-grid like object hm... several types possible here... GridSL vs GridUR
    k_ranges::Tuple{Vararg{AbstractVector, dim}} # Array of 1D ranges in real units
    k_ndrange::Tuple{Vararg{LazyGrids.AbstractGrid, dim}}  # Mesh-grid like object hm... several types possible here... GridSL vs GridUR
end 
function ImageInfo(
        npnts::Tuple{Vararg{Int, dim}}, 
        dxs::Tuple{Vararg{Float64, dim}}, 
        units::Tuple{Vararg{String, dim}}
    ) where dim

    dks = (2 * π) ./ (npnts .* dxs) 

    index_ranges = Tuple(1:npnt for npnt in npnts)
    # Because I always want 0,0 to be in the range I do need to distinguish odd versus even.
    ranges = Tuple(_range_with_zero(npnt, dx) for (dx, npnt) in zip(dxs, npnts)) 
    ndrange = LazyGrids.ndgrid(ranges...)

    k_ranges = Tuple(FFTW.fftfreq(n,  (2 * π) / dx) for (n, dx) in zip(npnts, dxs) )
    k_ndrange = LazyGrids.ndgrid(k_ranges...)

    # return ndrange
    return ImageInfo{dim}(npnts, dxs, dks, units, index_ranges, ranges, ndrange, k_ranges, k_ndrange)
end
function ImageInfo(
        npnts::Tuple{Vararg{Int, dim}},
        dxs::Tuple{Vararg{Float64, dim}}) where dim
    units = Tuple("" for i in 1:dim)
    return ImageInfo(npnts, dxs, units)
end
function ImageInfo(npnts::Tuple{Vararg{Int, dim}}) where dim
    dxs = Tuple(1.0 for i in 1:dim)
    return ImageInfo(npnts, dxs)
end
function ImageInfo(info::CameraInfo)
    ImageInfo(
        info.sensor_size .÷ info.downsample, 
        info.pixel_size .* info.downsample ./ info.magnification, 
        info.units);
end

"""
k_ranges_lab

Convert the computationally useful ranges into lab units
"""
k_ranges_lab(info::ImageInfo, scale) = (FFTW.fftshift(k_range) .* (scale ./ (2*pi)) for k_range in info.k_ranges) 
k_ranges_lab(info::ImageInfo) = k_ranges_lab(info, 1)

"""
find the index associated with a set of coordinates
"""
index_from_coords(info::ImageInfo, coords) = Tuple(argmin( (range .- coord).^2) for (coord, range) in zip(coords, info.ranges))

#=
##          ###    ########   ######   ######  ########  #### ########  ########
##         ## ##   ##     ## ##    ## ##    ## ##     ##  ##  ##     ##    ##
##        ##   ##  ##     ## ##       ##       ##     ##  ##  ##     ##    ##
##       ##     ## ########   ######  ##       ########   ##  ########     ##
##       ######### ##     ##       ## ##       ##   ##    ##  ##           ##
##       ##     ## ##     ## ##    ## ##    ## ##    ##   ##  ##           ##
######## ##     ## ########   ######   ######  ##     ## #### ##           ##
=#

struct LabscriptConfig
    experiment_shot_storage::String
    output_folder_format::String
    filename_prefix_format::String
    extension::String
    file_fstring::String
    extra_fstring::Dict{Symbol, String}
end 
LabscriptConfig(experiment_shot_storage, output_folder_format, filename_prefix_format; extension=".h5", kwargs...) = LabscriptConfig(
    experiment_shot_storage, 
    output_folder_format, 
    filename_prefix_format,
    extension, 
    labscript_file_fstring(experiment_shot_storage, output_folder_format, filename_prefix_format; extension=extension),
    Dict(k => labscript_file_fstring(v; extension="") for (k, v) in kwargs)
)

"""
    LabscriptSequence

Basic information used to define an individual sequence consisting of a collection of shots
"""
struct LabscriptSequence
    script_basename::String
    year::Int
    month::Int
    day::Int
    index::Int
    shots::Vector{Int}
end 
# LabscriptSequence(script_basename, year, month, day, index, shots) = LabscriptSequence(script_basename, year, month, day, index, Vector(shots))
Base.String(seq::LabscriptSequence) = "$(seq.script_basename)_$(seq.shots[1])...$(seq.shots[end])"
Base.show(io::IO, seq::LabscriptSequence) = print(io, String(seq))

"""
    LabscriptImage
    
Information needed to a single image from specific camera
"""
struct LabscriptImage
    orientation::AbstractString # Orientation label for saved image.
    label::AbstractString # Label of saved image (ignore if empty)
    image::AbstractString # Labscript identifier
end

"""
    ImageGroup
    
A collection of related images
"""
struct ImageGroup
    images::Dict{Symbol, LabscriptImage} # dictionary where key is the name that will be used in the analysis
    camera_info::CameraInfo
    image_info::ImageInfo
    image_process::Function # Run after loading each image.  Can do anything, but is designed to check for bad data
    group_process::Function # Run after loading the whole image group.  Can do anything, but is designed to label the group if it contains bad data
end
ImageGroup(camera_info::CameraInfo, image_info::ImageInfo, image_process::Function, group_process::Function; kwargs...) = ImageGroup(Dict(k=>v for (k, v) in kwargs), camera_info, image_info, image_process, group_process)
ImageGroup(camera_info::CameraInfo, image_info::ImageInfo; kwargs...) = ImageGroup(camera_info, image_info, identity, identity; kwargs...)

ImageGroup(camera_info, image_process, group_process; kwargs...) = ImageGroup(camera_info, ImageInfo(camera_info), image_process, group_process; kwargs...)
ImageGroup(info; kwargs...) = ImageGroup(info, identity, identity; kwargs...)


raw"""
    labscript_file

Creates a typical labscript style prefix for a file including the path

An example file might be:
    "2024-03-22_0010_BEC_elongatedTrap_FK_PEP_fieldMonitor_066.h5"

and for the path:
    "~/Documents/data/RbChip/Labscript/BEC_elongatedTrap_FK_PEP_fieldMonitor/2024/03/22/0010"

Notice the partly redundent information where in this example we had:
    experiment_shot_storage = "~/Documents/data/RbChip/Labscript"
    script_basename = "BEC_elongatedTrap_FK_PEP_fieldMonitor"
    year = 2024
    month = 03
    day = 22
    sequence_index = 10
    shot = 66

It is important to use format strings such as "0>4d" that will make strings like 0111 or 0001 to correspond
to the configuration information usually in the labscript config such as:

    output_folder_format = %%Y\%%m\%%d\{sequence_index:04d}
    filename_prefix_format = %%Y-%%m-%%d_{sequence_index:04d}_{script_basename}

Here we will use a slightly different format based on the python f-string.  For the example above we would have

    experiment_shot_storage = "~/Documents/data/RbChip/Labscript"
    output_folder_format = "{year:04d}/{month:02d}/{day:02d}/{sequence_index:04d}"
    filename_prefix_format = "{year:04d}-{month:02d}-{day:02d}_{sequence_index:04d}_{script_basename}_{shot:04d}"
    script_basename = "BEC_elongatedTrap_FK_PEP_fieldMonitor"
    year = 2024
    month = 03
    day = 22
    sequence_index = 10
    shot=66
"""
function labscript_file_fstring(args...; extension::String=".h5")

    ks = (
        "script_basename",
        "year",
        "month",
        "day",
        "sequence_index",
        "shot"
    )

    file_fstring = ""

    for arg in args[1:end-1]
        arg_string = _fstring_key_to_index(arg, 0, ks)
        file_fstring *= "$(arg_string)/"
    end
    arg_string = _fstring_key_to_index(args[end], 0, ks)
    file_fstring *= "$(arg_string)$(extension)"

    return file_fstring
end

"""
fstring_key_to_index

Takes a string formatted like a python f-string using the kwarg format and replaces with numerical indices like the args format.
"""
fstring_key_to_index(s::String, args...; kwargs...) =  _fstring_key_to_index(s, length(args),  keys(kwargs))

"""
    _fstring_key_to_index

Helper function

    n_args : Number of arguments
    ks : tuple or vector of strings
"""
function _fstring_key_to_index(s::String, n_args::Int, ks)
    n_ks = length(ks)
    rules = Tuple("{"*string(ks[idx]) => "{"*string(idx+n_args) for idx in 1:n_ks)
    replace(s, rules...)
end

"""
labscript_file_name

generate a single labscript file from a f-string
"""
labscript_file_name(file_fstring, sequence::LabscriptSequence, args...) = Format.format(
    file_fstring, 
    sequence.script_basename, 
    sequence.year,
    sequence.month,
    sequence.day,
    sequence.index,
    args...)

"""
file_name_array : this function returns an array of files that should be evaulated
    Takes as parameters a list of DataFile structs
    with keyword parameters
    extension: the filename
    formatstring: a python style format string directing how to format the index
        "0>4d" will make strings like 0111 or 0001
"""
function file_name_array(sequences::Vector{LabscriptSequence}, labconfig::LabscriptConfig)

    # Now build the array of file names
    filelist = Array{String}(undef, 0)

    for sequence in sequences	
        # And here is the list for this element of files.  
        newfiles = [labscript_file_name(labconfig.file_fstring, sequence, i) for i in sequence.shots]
        append!(filelist, newfiles)
    end
    
    return filelist
end
file_name_array(file::LabscriptSequence, labconfig::LabscriptConfig) = file_name_array([file], labconfig)

"""Get previously saved image from the h5 file.

h5_file : h5 reference or a file name

Args:
    orientation (str): Orientation label for saved image.
    label (str): Label of saved image.
    image (str): Identifier of saved image.

KWargs:
    cast = type to convert to if not nothing

Raises:
    Exception: If the image or paths do not exist.

Returns:
    image array.
"""
function get_image(h5_file::HDF5.File, orientation::String, label::String, image::String, datatype::Type)

    h5path = "images"
    if ~haskey(h5_file, h5path)
        throw( ErrorException("File does not contain any images") )
    end

    if ~haskey(h5_file[h5path], orientation)
        throw( ErrorException("File does not contain any images with orientation $(orientation)") )
    end
    h5path = h5path*"/"*orientation

    if ~haskey(h5_file[h5path], label) # no label is OK
        throw( ErrorException("File does not contain any images with label $(label)") )
    end
    h5path = h5path*"/"*label

    if ~haskey(h5_file[h5path], image)
        throw( ErrorException("Image '$(image)' not found in file '$(h5_file)' in group '$h5path'") )
    end

    return convert(Array{datatype}, HDF5.read(h5_file[h5path][image]))
end
function get_image(filename::String, args...)
    image = HDF5.h5open(filename, "r") do h5_file
        get_image(h5_file, args...)
    end
    return image
end

"""Get previously saved image from the h5 file.

h5_file : h5 file name

KWargs:
    shot_process (Function): a function to run after each shot

Returns:
    Dict of image arrays.
"""
function get_images(h5_file::HDF5.File, imagegroups::Vector{ImageGroup}; shot_process::Union{Function,Nothing} = nothing)
    local imagedict = Dict{Symbol, Any}()
    imagedict[:file_name] = splitpath(h5_file.filename)[end]

    for imagegroup in imagegroups
        
        local imagedict_group = Dict{Symbol, Any}()
        for (name, image) in imagegroup.images
            img = get_image(h5_file, image.orientation, image.label, image.image, imagegroup.camera_info.datatype)
            img = imagegroup.image_process(img)

            # Place the downsampled image into the dict
            imagedict_group[name] = ImProc.downsample_array(img, imagegroup.camera_info.downsample)
        end
        imagedict_group = imagegroup.group_process(imagedict_group)
        merge!(imagedict, imagedict_group)
    end

    if shot_process === nothing
        return imagedict
    end

    return shot_process(h5_file, imagedict)
end
function get_images(filename::String, imagegroups::Vector{ImageGroup}; kwargs...)

    imagedict = HDF5.h5open(filename, "r") do h5_file
        get_images(h5_file, imagegroups; kwargs...)
    end

    return imagedict
end
get_images(filename, imagegroup::ImageGroup; kwargs...) = get_images(filename, [imagegroup]; kwargs...)

#=
 ######   ######## ##    ## ######## ########     ###    ##
##    ##  ##       ###   ## ##       ##     ##   ## ##   ##
##        ##       ####  ## ##       ##     ##  ##   ##  ##
##   #### ######   ## ## ## ######   ########  ##     ## ##
##    ##  ##       ##  #### ##       ##   ##   ######### ##
##    ##  ##       ##   ### ##       ##    ##  ##     ## ##
 ######   ######## ##    ## ######## ##     ## ##     ## ########
=#


"""
dict_to_h5

recursivly iterates through d and saves contents into the h5 reference
"""
function dict_to_h5(h5ref::HDF5.H5DataStore, d::Dict{String, Any})
    for (key, item) in d

        # Clear out old data
        if key in keys(h5ref)
            HDF5.delete_object(h5ref, key)
        end

        # recurse if needed
        if typeof(item) == Dict{String, Any}
            HDF5.create_group(h5ref, key)

            dict_to_h5(h5ref[key], item)
        else
            HDF5.write_dataset(h5ref, key, item)
        end
        
    end
end
dict_to_h5(filename::AbstractString, d::Dict{String, Any}) = HDF5.h5open( f -> dict_to_h5(f, d), filename, "cw")

"""
h5_to_dict

Returns a dictionary containg the data from the h5 file
"""
h5_to_dict(filename) = HDF5.h5open(h5_to_dict, filename, "r")
h5_to_dict(h5ref::HDF5.H5DataStore) = read(h5ref)


function write_dataset_compress(grp, key, data, args...; kwargs...)
    if ndims(data) == 0
        pop!(kwargs, :chunk, nothing)
        pop!(kwargs, :deflate, nothing)
    else
        get!(kwargs, :shuffle, true)
        get!(kwargs, :deflate, 9)
        get!(kwargs, :chunk, FileIO.guess_chunk(data))
    end

    HDF5.write_dataset(grp, key, data, args...; kwargs...)
end


"""
overwrite_dataset overwrites a dataset!
"""
function overwrite_dataset(f, k, args...; kwargs...)
    if  k in  keys(f)
        HDF5.delete_object(f,  k)
    end
    HDF5.write_dataset(f, k, args...; kwargs...)
end

"""
overwrite_group overwrites a group
"""
function overwrite_group(f, k, args...; kwargs...)
    if  k in  keys(f)
        HDF5.delete_object(f,  k)
    end
    HDF5.create_group(f, k, args...; kwargs...)
end

"""
This autochunking code was converted from python
"""

# Define constants similar to the Python code
const CHUNK_BASE = 16 * 1024      # 16 KB
const CHUNK_MIN = 8 * 1024        # 8 KB
const CHUNK_MAX = 1024 * 1024     # 1 MB

"""
Guess an appropriate chunk layout for a dataset, given its shape and
the size of each element in bytes. Will allocate chunks only as large
as MAX_SIZE. Chunks are generally close to some power-of-2 fraction of
each axis, slightly favoring bigger values for the last index.

This function is a direct translation from Python to Julia.
"""
function guess_chunk(shape, typesize)

    # For unlimited dimensions (size 0), we guess 1024
    shape = [x != 0 ? x : 1024 for x in shape]

    ndims = length(shape)
    if ndims == 0
        throw(ArgumentError("Chunks not allowed for scalar datasets."))
    end

    chunks = Float64.(shape)
    if !all(isfinite.(chunks))
        throw(ArgumentError("Illegal value in chunk tuple"))
    end

    # Determine the optimal chunk size in bytes using a PyTables-like expression.
    # This is kept as a float.
    dset_size = prod(chunks) * typesize
    target_size = CHUNK_BASE * (2.0 ^ log10(dset_size / (1024.0 * 1024.0)))

    if target_size > CHUNK_MAX
        target_size = CHUNK_MAX
    elseif target_size < CHUNK_MIN
        target_size = CHUNK_MIN
    end

    idx = 0
    while true
        # Repeatedly loop over the axes, dividing them by 2. Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        # 2. The chunk is smaller than the maximum chunk size

        chunk_bytes = prod(chunks) * typesize

        if ((chunk_bytes < target_size || abs(chunk_bytes - target_size) / target_size < 0.5) &&
            chunk_bytes < CHUNK_MAX)
            break
        end

        if prod(chunks) == 1
            break  # Element size larger than CHUNK_MAX
        end

        # Modulo indexing with 1-based indexing in Julia
        idx_mod = mod1(idx, ndims)
        chunks[idx_mod] = ceil(chunks[idx_mod] / 2.0)
        idx += 1
    end

    return Tuple(Int.(chunks))
end
guess_chunk(data) = guess_chunk(size(data), sizeof(eltype(data)))

#=
 ######  ##     ##  #######  ##     ## ##       ########     ########  ########    ######## ##        ######  ######## ##      ## ##     ## ######## ########  ########
##    ## ##     ## ##     ## ##     ## ##       ##     ##    ##     ## ##          ##       ##       ##    ## ##       ##  ##  ## ##     ## ##       ##     ## ##
##       ##     ## ##     ## ##     ## ##       ##     ##    ##     ## ##          ##       ##       ##       ##       ##  ##  ## ##     ## ##       ##     ## ##
 ######  ######### ##     ## ##     ## ##       ##     ##    ########  ######      ######   ##        ######  ######   ##  ##  ## ######### ######   ########  ######
      ## ##     ## ##     ## ##     ## ##       ##     ##    ##     ## ##          ##       ##             ## ##       ##  ##  ## ##     ## ##       ##   ##   ##
##    ## ##     ## ##     ## ##     ## ##       ##     ##    ##     ## ##          ##       ##       ##    ## ##       ##  ##  ## ##     ## ##       ##    ##  ##
 ######  ##     ##  #######   #######  ######## ########     ########  ########    ######## ########  ######  ########  ###  ###  ##     ## ######## ##     ## ########
=#
#
# Pulled from the Chip lab file loaders.
#

function group_postprocess_pci(images)
    images[:good_pci] = !any(ismissing.(values(images)))
    return images
end

function group_postprocess_abs(images)
    images[:good_abs] = !any(ismissing.(values(images)))
    return images
end

"""
initial_load

This is a data loader. It returns a named tuple of data where
each entry is a stack of 2D images.
"""
function initial_load(file_names::Vector, image_group_pci, image_group_abs)
    
    # First get data as a vector of named tuples.
    data = [get_images(file_name, [image_group_pci, image_group_abs]) for file_name in file_names]

    # Strip out clearly bad data
    data = [d for d in data if d[:good_pci] && d[:good_abs]]

    # Now re-organize into a named-tuple of stacks of images
    nimages = length(data)
    ks = keys(data[1])
    rng = 1:nimages

    return (NamedTuple( (k, stack([data[i][k] for i in rng]) ) for k in ks), nimages)
end

end # module FileIO