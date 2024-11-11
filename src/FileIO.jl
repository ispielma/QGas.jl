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
end 
LabscriptConfig(experiment_shot_storage, output_folder_format, filename_prefix_format; extension=".h5") = LabscriptConfig(
    experiment_shot_storage, 
    output_folder_format, 
    filename_prefix_format,
    extension, 
    labscript_file_fstring(experiment_shot_storage, output_folder_format, filename_prefix_format; extension=extension)
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
ImageGroup(camera_info, image_process, group_process; kwargs...) = ImageGroup(Dict(k=>v for (k, v) in kwargs), camera_info, ImageInfo(camera_info), image_process, group_process)
ImageGroup(camera_info; kwargs...) = ImageGroup(camera_info, Identity, Identity; kwargs...)


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
function labscript_file_fstring(
    experiment_shot_storage::String,
    output_folder_format::String,
    filename_prefix_format::String;
    extension::String=".h5"
    )

    ks = (
        "script_basename",
        "year",
        "month",
        "day",
        "sequence_index",
        "shot"
    )

    experiment_shot_storage = _fstring_key_to_index(experiment_shot_storage, 0, ks)

    output_folder_format = _fstring_key_to_index(output_folder_format, 0, ks)

    filename_prefix_format = _fstring_key_to_index(filename_prefix_format, 0, ks)

    return "$(experiment_shot_storage)/$(output_folder_format)/$(filename_prefix_format)$(extension)"
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
        newfiles = [Format.format(
            labconfig.file_fstring, 
            sequence.script_basename, 
            sequence.year,
            sequence.month,
            sequence.day,
            sequence.index,
            i) for i in sequence.shots]
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
        throw( KeyError("File does not contain any images") )
    end

    if ~haskey(h5_file[h5path], orientation)
        throw( KeyError("File does not contain any images with orientation $(orientation)") )
    end
    h5path = h5path*"/"*orientation

    if ~haskey(h5_file[h5path], label) # no label is OK
        throw( GenericException("File does not contain any images with label $(label)") )
    end
    h5path = h5path*"/"*label

    if ~haskey(h5_file[h5path], image)
        throw( GenericException("Image $(image) not found in file") )
    end

    data = HDF5.read(h5_file[h5path][image]) 

    return convert(Array{datatype}, data)
end
function get_image(filename::String, args...)
    image = HDF5.h5open(filename, "r") do h5_file
        get_image(h5_file, args...)
    end
    return image
end

"""Get previously saved image from the h5 file.

h5_file : h5 file name

Args:
    orientation (str): Orientation label for saved image.
    label (str): Label of saved image.
    image (str): Identifier of saved image.

Returns:
    Dict of image arrays.
"""
function get_images(h5_file::HDF5.File, imagegroups::Vector{ImageGroup})
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

    return imagedict
end
function get_images(filename::String, imagegroups::Vector{ImageGroup})

    imagedict = HDF5.h5open(filename, "r") do h5_file
        get_images(h5_file, imagegroups)
    end

    return imagedict
end
get_images(filename, imagegroup::ImageGroup) = get_images(filename, [imagegroup])

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

end # module FileIO