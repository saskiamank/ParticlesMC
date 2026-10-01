module IO

using ..ParticlesMC: Particles, Atoms, Molecules, System
using ..ParticlesMC: fold_back, volume_sphere
using ..ParticlesMC: EmptyList, LinkedList, CellList, VerletList
using ..ParticlesMC: Model, GeneralKG, JBB, BHHP, SoftSpheres, KobAndersen, Trimer, LennardJones
using Arianna
using Distributions, LinearAlgebra, StaticArrays, Printf
using DataStructures: OrderedDict
export XYZ, EXYZ, LAMMPS
export load_configuration, load_chains

include("xyz.jl")
include("exyz.jl")
include("lammps.jl")

function Arianna.write_system(io, system::Particles)
    println(io, "\tNumber of particles: $(length(system))")
    println(io, "\tDimensions: $(system.d)")
    println(io, "\tCell: $(system.box)")
    println(io, "\tDensity: $(system.density)")
    println(io, "\tTemperature: $(system.temperature)")
    println(io, "\tNeighbour list: " * replace(string(typeof(system.neighbour_list)), r"\{.*" => ""))
    return nothing
end


# ------------------
# function to extract the format of the given filename and check whether it is valid
# then calls the actual function to load_configuration while specifying the format  
function load_configuration(filename::String)
    io = open(filename, "r")  # Open file as IOStream
    if endswith(filename, ".xyz")
        return load_configuration(io, XYZ())
    elseif endswith(filename, ".exyz")
        return load_configuration(io, EXYZ())
    elseif endswith(filename, ".lmp") || endswith(filename, ".lammpstrj") || endswith(filename, ".lammps")
        return load_configuration(io, LAMMPS())
    else
        error("Unsupported file format: $filename")
        return nothing
    end
end
# NO CHANGE


# -----------------
# input:      trajectory or snaptshot, 
#             the format of this file, 
#             m = {Int} specifying the starting frame (m=1 for molecules enforce )  
# 
# output:     dictionary, content is specified at the end of function, section OUTPUT
function load_configuration(io, format::Arianna.Format; m=1)
    #region
    data = readlines(io) 
    # loads file into an array called data, each entry corresponding to one row in data(?) 

    N, box, column_info, metadata = read_header(data, format)   
    # read_header() is defined in each .jl file for the different file formats...                                                       
        # extract the information from the header in the file
        # N saves the number of atoms/entries per frame(???) 
        # box is now a d-dimensional static Vector containing the length 
        # of the cell in each spatial direction, where d is the dimension of space (in most cases 3)
        # column_info stores an ordered dictionary containing pairs of 
        # "name" -> [dimension of the stored information, 
        # index indicating the position of the information stored in "name"] (I think...), 
        # because column_info basically provides the header for all data stored in the next lines, 
        # so we need to know which column stores what and how many colums of the name "name" exist 
        # (e.g. there are d colums storing the "position")
        # metadata is just split into an array by read_header, and contains additional 
        # information like temperature or density or timestep
    #
    

    selrow = get_selrow(format, N, m)       # selrow = number of line where the positions start (for the selected frame m)
    frame = data[selrow:selrow+N-1]         # array with N entries, each entry one row of positions 
                                            # (as well as all other information specified in column_info, 
                                            # like molecul or species)
                                            # QUESTION: which size does frame have? Is it a 1D array of strings, 
                                            # each string being one row of the file? (???) 
    bool_molecule = "molecule" in keys(column_info)     # check whether there is a column specifying which molecule the 
                                                        # line corresponds to 
    bool_species = "species" in keys(column_info)       # check whether there is a column specifying which specy of atom  - " - 
    


    # start ---------------------------------------------------------------------------------------
    # for molecules: check that we start with first frame, check that there is the right amount 
    # of data for molecules in the file create an empty array called molecules and store an array 
    # of N Vectors which contain indices of all atoms an atom is bonded with in bond
    if bool_molecule
        if m != 1
            error("For molecular systems the frame index has to be equal to 1")
        end
        molecule_d, molecule_index = column_info["molecule"]    # molecule_d is the number of columns with header "molecule", 
                                                                # molecule_index specifies which column(s) that are
        if molecule_d != 1
            error("molecule dimension must be 1")   # means that there can only be one column 
                                                    # in which the current molecule is indexed 
        end
        molecule = Vector{Int}(undef, N)    # Vector{Type}(undef, N) creates an uninitialized array 
                                            # of dimension N with elements of type Type 

        bond = read_bonds(data, N, format)  # this function returns a an array of N Vectors, 
                                            # where each Vector corresponds to one atom and 
                                            # stores the index of every atom it has a bond with 
    end
    # end ---------------------------------------------------------------------------------------



    # initialize an array called species, which is either an N-dimensional array filled with integer 1s if species is not given,
    # or an unitialized array of dimension N with elements of type type(species) (or should be I think)
    if bool_species
        species_d, species_index = column_info["species"]
        if species_d != 1
            error("Species dimension must be 1")
        end
        sT = typeof(eval(Meta.parse(split(frame[1], " ")[1])))      # sT is the type of the first entry in the first row of frame, 
                                                                    # which is determined by making it a string and then evaluating it, such that "35" 
                                                                    # would become an integer 35 and sT would be Int 
                                                                    # -> QUESTION: shouldn't the index 1 be species_index instead of 1? (???)
        species = Vector{sT}(undef, N)                              # create Vector of N uninitialized entries of type sT
    else
        species = ones(Int, N)              # species is an N dimensional array filled with integer 1s if species 
                                            # is not given, so in the case of an atomistic system, where all atoms are of the same species
    end




    # start -----------------------------------------------------------------------------
    # some tests and create position: an unitialized vector of dimension N with elements: 
    # SVectors of dimension d(spatial dimension) and type float
    # adjust box according to dimension of positions 
    if "pos" in keys(column_info)
        pos_d, pos_index = column_info["pos"]       # pos_d is the dimension of the system, since each column 
                                                    # specifies the position with respect to one axis 
        position = Vector{SVector{pos_d,Float64}}(undef, N)         # position is an unitialized vector of dimension N 
                                                                    # where its elements are static vectors of dimension pos_d
                                                                    # with entries of type float64
    else
        missing_key_error("pos")
    end
    
    if pos_d < length(box)
        box = box[1:pos_d]      # if there are more values in box than spatial dimensions, 
                                # only the first d values are kept 
    end
    # end --------------------------------------------------------------------------

    # for loop to fill species, molecule, position with data from trajectory line by line (atom by atom) 
    # -> i is basically index of atom
    for i in eachindex(frame)
        split_line = split(frame[i], " ")
        if bool_species
            species[i] = eval(Meta.parse.(split_line[species_index]))       # eval(Meta.parse.(...)) is used to convert the string representation 
                                                                            # of ... into its actual type (e.g., Int, Float, etc.), such that now 
                                                                            # species is an array with N entries, each entry being the species of 
                                                                            # the corresponding atom (as an integer)
        end
        if bool_molecule
            molecule[i] = parse.(Int64, split_line[molecule_index])         # same for molecule 
        end
        position[i] = SVector{pos_d}(parse.(Float64, split_line[pos_index:pos_index+pos_d-1]))  # position[i] is now a static vector of dimension pos_d, 
                                                                                                # where the entries are the position of atom i in each 
                                                                                                # spatial dimension (as float64)
                                                                                                # so position is an array of N static vectors of dimension pos_d
    end
    #endregion

    # start OUTPUT ---------------------------------------------------------------------------------
    # create a dictionary to store all data 
    config_dict = Dict(:N => N,         # number of atoms 
        :d => pos_d,                    # integer (float?) specifying spatial dimension 
        :box => box,                    # d-dimensional static Vector containing the length 
        :species => species,            # in case of species not defined: N dimensional array filled with integer 1s
        :position => position,          # unitialized vector of dimension N, elements:
                                        # SVectors of dimension d(spatial dimension) and type float 
        :metadata => metadata           # array containing additional information 
    )
    if bool_molecule
        config_dict[:molecule] = molecule   # N-dimensional Vector with elements of type Int, containing the index of the molecule the atom belongs to
        config_dict[:bond] = bond           # array of N Vectors which contain indices of all atoms an atom is bonded with
    end
    # end OUTPUT ------------------------------------------------------------------------------------------
    return config_dict
end
# NO CHANGE


# -----------------
function read_bonds(filename::String, N)
    io = open(filename, "r")
    N_bonds = parse(Int, io[1])
    return construct_bonds_array(io[2:end], N_bonds, N)
end
# NO CHANGE, I think 



# ------------------
# function that takes the input file and returns an array called bond of N Vectors, 
# where each Vector corresponds to one atom and stores the index of every atom it has a bond with
function construct_bonds_array(io, N_bonds, N)
    bond = [Vector{Int}() for _ in 1:N]
    bond_index = 1
    for i in 1:N_bonds
        line = split(io[i], " ")
        if length(line) != 2
            error("Invalid bond format in line $i: expected two integers.")
        end
        try
            atom_i, atom_j = parse.(Int, line[bond_index:bond_index+1])             # tries to assign 2 integers belonging to one bond to atom_i and atom_j, 
                                                                                    # which are the indices of the two atoms forming a bond
        catch
            error("Invalid bond format in line $i: Could not parse integers.")
        end
        push!(bond[atom_i], atom_j)                                                 # bond[atom_i] is a vector containing the indices of all atoms 
                                                                                    # that are bonded to atom_i, atom_j is added to this vector 
        push!(bond[atom_j], atom_i)
    end
    return bond
end
# NO CHANGE


# -------------------
filter_kwargs(pairs...) = (; (k => v for (k, v) in pairs if v !== nothing)...) 
# filters all key-value pairs where the value is not empty, 
# for simple functions consisting only of one expression, you don't need to put function ... end 


#-------------------
# function extracts the given parameters for the model and returns the specified model in [model], name = ""
# this function is called in load_chains() which is called in ParticlesMC  
# returns a struct of the model specified in the input file, which is then stored in the model_matrix
function get_model(data, i::Int, j::Int)
    key = i <= j ? "$i-$j" : "$j-$i"                            # select interaction between species i and j, since the interaction is symmetric
    m = data[key]
    if m["name"] == "GeneralKG"
        rcut = get(m, "rcut", nothing)

        return GeneralKG(m["epsilon"], m["sigma"], m["k"], m["r0"];
            filter_kwargs(
                :rcut => get(m, "rcut", nothing),
                :ϵbond => get(m, "epsilonbond", nothing),
                :σbond => get(m, "sigmabond", nothing),
                :rcutbond => get(m, "rcutbond", nothing),
            )...)
    elseif m["name"] == "SmoothLennardJones"
        return SmoothLennardJones(m["epsilon"], m["sigma"];
            filter_kwargs(
                :rcut => get(m, "rcut", nothing))...)
    elseif m["name"] == "LennardJones"
        return LennardJones(m["epsilon"], m["sigma"];
            filter_kwargs(
                :rcut => get(m, "rcut", nothing),
                :shift_potential => get(m, "shift_potential", true),
            )...)
    else
        error("Model $(m["name"]) is not implemented")
        return nothing
    end
end
# NO CHANGE, I think 


# ----------------------
# this function returns a an array of N Vectors, where each Vector corresponds to one atom and stores the index of every atom it has a bond with 
# this also means that the Vectors do not necessarily have the same length, since the length of the vector corresponds to 
# the amount of atoms that have a bond with the current atom 

# also enables different bond types, don't think this has been used so far (???) 
function read_bonds(data, N, format::Arianna.Format)
    selrow = get_selrow(format, N, 1)
    bonds_data = data[N+selrow:end]     # the bond data is stored after the positions 
                                        # each row only contains to integer numbers indexing which two atoms form a bond 

    if length(bonds_data) == 0
        error("No bonds found in the file")
    else
        N_bonds, column_info = read_bonds_header(bonds_data, format)
    end
    bool_btype = "btype" in keys(column_info)  # is there a column providing information about the type of the bonds 
    bool_bond = "bond" in keys(column_info)    # is there a column/are there columns for the bonds? 
    if bool_btype
        btype_d, btype_index = column_info["btype"]
        if btype_d != 1
            error("Bond type dimension must be 1")
        end
        btype = fill(Vector{Int}(), N)
    end
    if bool_bond                                    # if there are columns for the bonds 
        bond_d, bond_index = column_info["bond"]
        if bond_d != 2                              
            error("Bond dimension must be 2. Found $bond_d.")    # there must be exactly 2 columns for the bonds (two atoms form a bond and the columns just have the number of the 2 atoms)
        end
        row_bonds = get_row_bonds(selrow, N, format)    # just returns 2 always (for xyz), i think should index the row where the entries start (???) 
        bond = [Vector{Int}() for _ in 1:N]             # bond is an array of N empty vectors with elements of type Int (???)  
                                                        # [Vector{Int}() for _ in 1:N] is better than bond = [Int[]] * N because in the latter version all entries could point to the same vector 
                                                        # (AI, haven't really understood it, but something I should maybe pay attention to)


        for i in 1:N_bonds # going through all rows, each row belonging to one bond between to atoms 
            atom_i, atom_j = parse.(Int, split(bonds_data[row_bonds+i], " ")[bond_index:bond_index+1])
            # atom_i and atom_j store both an integer values corresponding to the entries in the current row 
            push!(bond[atom_i], atom_j)  # save the bond i,j for atom_i
            push!(bond[atom_j], atom_i)  # save the bond i,j for atom_j
            # this basically creates for every atom a list with atoms that are bonded 

            if bool_btype
                btype_ij = parse.(Int, split(bonds_data[row_bonds+i], " ")[btype_index])
            else
                btype_ij = 1
            end
            #push!(btype[atom_i], btype_ij)
            #push!(btype[atom_j], btype_ij)
        end
    else
        error("Bond array is not written in the $format file")
    end
    return bond
end
# NO CHANGE, I think 


function missing_key_error(key)
    error(error("$key array has not been found in metadata or is not defined. Define the $key in the args Dict"))
end
# NO CHANGE

function broadcast_dict(dicts, key)
    return [dict[key] for dict in dicts]
end
# NO CHANGE 


# --------------
# Input: init_path is the path to the input file for the initial configuration 
# args is a dictionary containing the parameters for the simulation, like temperature, density, model, extracted 
# from the .toml file
# 
# the output of this function is an array of System structs, where each struct corresponds to one input file (???)
function load_chains(init_path; args=Dict(), filename="", verbose=false)

    input_files = Vector{String}()          # initializes a Vector whose entries are of the type String 
    if isfile(init_path)                    # checks whether the first argument of the function is a file 
        push!(input_files, init_path)       # if so, puts the file in init_path inside the Vector input_files
    elseif isdir(init_path)                 # if the first argument is not a file, but a directory
        for (root, dirs, files) in walkdir(init_path)
            for file in files
                if occursin(filename, file)
                    push!(input_files, joinpath(root, file))   
                end
            end
        end
    end
    # now input_files is a Vector of all files, with type String(???) 


    # using a && b, the subexpression b is only evaluated if a evaluates to true 
    # using a || b, the subexpression b is only evaluated if a evaluates to false 
    verbose && println("Processing $(length(input_files)) configuration file(s)")  # prints how many input files where given 
    verbose && @show input_files                                                   # prints which input files where given 


    # all data in the files is loaded in config_dict, 
    # the entries are specified in the comments of the function  
    config_dict = load_configuration.(input_files)
    # one dictionary in config_dict per file (???) 

    # start------------------------------------------------------------------------------
    # creating arrays, where each entry corresponds to one dict, where each dict corresponds to one file in input_files (???) 
    # broadcast_dict(a, b) just creates an array, each element being the value assigned to b["a"]
    initial_species_array = broadcast_dict(config_dict, :species)
    initial_position_array = broadcast_dict(config_dict, :position)      
    initial_box_array = broadcast_dict(config_dict, :box)               # box is a d-dimensional array (each position vector is of dimension d, 
                                                                        # so d is the dimension of the system) containing the width/length of the 
                                                                        # cell in each spatial dimension 
    metadata_array = broadcast_dict(config_dict, :metadata)
    # end --------------------------------------------------------------------------

    # some tests, assigning N the number of atoms, d the spatial dimension of the system 
    N, d = config_dict[1][:N], config_dict[1][:d]
    @assert all(isequal(N), length.(initial_position_array))
    @assert all(isequal(d), vcat([length.(X) for X in initial_position_array]...))


    initial_density_array = length.(initial_position_array) ./ prod.(initial_box_array)     # initial_position_array is an array of all Ns (one N for each dictionary, why do we have several?) while initial_box_array is an array of d dimensional svectors containing the length of the box in each spatial dimension, such that the prod.() function just gives an array of volumes for each dict 
    # initial density_array is an array for each dict containing the density 

    # all() tests whether all values among the given dimensions of an array are true, all(element -> condition, array)
    # any() tests whether any values along the given dimensions of an array are true, any(element -> condition, array)
    has_temp = all(m -> any(x -> occursin("T:", x), m), metadata_array)             # test whether each dict/file has a 
                                                                                    # temperature specified in the metadata (???)
    has_model = all(m -> any(x -> occursin("model:", x), m), metadata_array)        # same for model 


    # start ---------------------------------------------------------------------------------------------------
    # for model and temperature: if there is a temperature specified in this function take it, 
    # otherwise use the temperature specified in the file, 
    # otherwise error (???)
    # update density 
    # -------------------

    # if there are initial temps for each file, creates an array with stores 
    # a float value for the initial temperature for each dict/file
    if length(metadata_array) ≥ 1 && has_temp
        initial_temperature_array = [parse(Float64, split(filter(x -> occursin("T:", x), m)[1], ":")[2]) for m in metadata_array]
    else
        initial_temperature_array = nothing
    end
    
    # same for model 
    if length(metadata_array) ≥ 1 && has_model
        input_models = [split(filter(x -> occursin("model:", x), m)[1], ":")[2] for m in metadata_array]
        @assert all(isequal(input_models[1]), input_models)
    else
        input_models = nothing
    end

    # Update density if needed
    # Basically just determines the ratio of densities to then rescale positions and box accordingly
    if haskey(args, "density") && !isnothing(args["density"])       # testing whether the function load_chains 
                                                                    # gets the argument density with a valid value 
        λs = (initial_density_array ./ args["density"]) .^ (1 / d)  # this is basically (now_density/desired density)^(1/d) 
                                                                    # but an array of it for each dict/file 
        initial_density_array .= args["density"]
        initial_position_array .= [X .* λ for (X, λ) in zip(initial_position_array, λs)]    # scales all positions
        initial_box_array .= [box .* λ for (box, λ) in zip(initial_box_array, λs)]          # scales the box length/width
    end
    # QUESTION: what is the resulting configuration is non-physical (particle positions overlap...)? 
    # QUESTION: rescaling of positions of ALL atoms should always lead to huge energies, since the 
    # distance between atoms in molecule is almost fixed (very narrow potential)... (???) 


    # Safely overriding arrays without type or dimension conflicts
    # if there is a temperature specified in this function take it, 
    # otherwise use the temperature specified in the file, 
    # otherwise error (???) 
    if haskey(args, "temperature") && !isnothing(args["temperature"])   # testing whether the function load_chains 
                                                                        # received valid value for new temperature 
        if args["temperature"] isa AbstractVector
            initial_temperature_array = args["temperature"]
        else
            initial_temperature_array = fill(args["temperature"], length(input_files))
        end
    elseif isnothing(initial_temperature_array)
        missing_key_error("temperature")
    end


    # same for model 
    if haskey(args, "model") && !isnothing(args["model"])
        if args["model"] isa AbstractVector
            input_models = args["model"]
        else
            input_models = fill(args["model"], length(input_files))
        end
    elseif isnothing(input_models)
        missing_key_error("model")
    end
    # end -----------------------------------------------------------------------------------------------------

    # Fold back into the box
    # fold_back(x, box) reduces x by y*box, where y is the amount of times box fits in x 
    # When would that happen (???) the density is scaled such that this should never occur or not??? 
    initial_position_array .= [[fold_back(x, box) for x in X] for (X, box) in zip(initial_position_array, initial_box_array)]


    # Copy configurations nsim times (replicas)
    # vcat(A...) concatenates along dimension 1 
    # i don't think that is ever called (???), load_chains() is only called in ParticlesMC.jl, where nsim is never specified
    if haskey(args, "nsim") && !isnothing(args["nsim"]) && args["nsim"] > 1
        nsim = args["nsim"]
        verbose && println("Generating $nsim replicas per input file")
        initial_position_array = vcat([[copy(x) for _ in 1:nsim] for x in initial_position_array]...)
        initial_species_array = vcat([[copy(x) for _ in 1:nsim] for x in initial_species_array]...)
        initial_density_array = vcat([[copy(x) for _ in 1:nsim] for x in initial_density_array]...)
        initial_temperature_array = vcat([[copy(x) for _ in 1:nsim] for x in initial_temperature_array]...)
    end

    # Parse model
    # start -----------------------------------------------------------------------------------------------
    # create an AbstractArray model_matrix
    # haven't really understood it yet, need to take a closer look still  (???)

    available_species = unique(vcat(initial_species_array...)) # save each available species once 
    n_species = length(available_species)
    if input_models[1] isa Dict
        model_matrix = SMatrix{n_species,n_species}([get_model(input_models[1], i, j) for i in 1:n_species, j in 1:n_species])    
                                                # I think(???) the get_model() function returns a struct of the model specified 
    elseif occursin(r"\(", input_models[1]) && occursin(r"\)", input_models[1])
        model_matrix = eval(Meta.parse(input_models[1]))
    else
        model_matrix = eval(Meta.parse(input_models[1] * "()"))
    end
    @assert isa(model_matrix, AbstractArray)
    # end -------------------------------------------------------------------------------------------------


    # start -----------------------------------------------------------------------------------------------
    # set list_type to LinkedList only if atoms on average interact with less than 10% of all atoms due to chosen cut-off
    # otherwise choose EmptyList
    # override it with the value for list_type passed to the function (???)

    maxcut = maximum([m.rcut for m in model_matrix])            # extract the maximum cut-off in model-matrix 
    Z = mean(initial_density_array) * volume_sphere(maxcut, d)  # Z is the average number of atoms interacting, 
                                                                # mean density times the volume of the sphere in d 
                                                                # dimensions with the cut-off radius
    list_type = Z / N < 0.1 ? LinkedList : EmptyList            # test whether the atoms interact with less then 10% of the 
                                                                # total number of atoms and choose LinkedList as list_type only if true

    if haskey(args, "list_type") && !isnothing(args["list_type"])
        list_type = eval(Meta.parse(args["list_type"]))
    end
    # end ----------------------------------------------------------------------------------------------

    list_parameters = get(args, "list_parameters", nothing)
    verbose && println("Using $list_type as cell list type")

    # Create independent chains (Preserves V2 System constraints)
    bool_molecule = :molecule in keys(config_dict[1])
    if bool_molecule
        initial_molecule_array = broadcast_dict(config_dict, :molecule)
        initial_bond_array = broadcast_dict(config_dict, :bond)
        masses = get(args, "masses", nothing)
        chains = [System(initial_position_array[k], initial_species_array[k], initial_molecule_array[k], initial_density_array[k], initial_temperature_array[k], model_matrix, initial_bond_array[k], masses=masses, list_type=list_type, list_parameters=list_parameters) for k in eachindex(initial_position_array)]
        # System() can be found in atoms.jl or molecules.lj depending on the arguments...
    else
        chains = [System(initial_position_array[k], initial_species_array[k], initial_density_array[k], initial_temperature_array[k], model_matrix, list_type=list_type, list_parameters=list_parameters) for k in eachindex(initial_position_array)]
    end

    verbose && println("$(length(chains)) chains created")
    return chains
end
# NO CHANGE, I think


# ----------------
# this function is used to format the position of the atoms in the output file
function formatted_string(num::Real, digits::Integer)
    fmtstr = "%." * string(digits) * "f"
    fmt = Printf.Format(fmtstr)
    return Printf.format(fmt, num)
end
# NO CHANGE


# ----------------
function write_position(io, position, digits::Int)
    for position_i in position
        formatted_position_i = formatted_string(position_i, digits)
        print(io, " ")
        print(io, formatted_position_i)
    end
    println(io)
    return nothing
end
# NO CHANGE

# ----------------
function store_bonds(io, system::Molecules, format::Arianna.Format)
    s = 0
    for bond in system.bonds                    # s should be the total number of bonds times 2, since each bond is stored twice in bonds 
        s += length(bond)
    end
    println(io, s ÷ 2)                          # devide s by 2 to get total number of bonds, and write it in io, 
                                                # which is the output file, as the first line of the bond section 
    write_bonds_header(io, format)              # function specified in the file for the given format, writes the header for the bond section
    for i in 1:system.N
        for j in system.bonds[i]
            if i < j
                println(io, "$i $j")            # writes in each colum the indices of the two atoms forming a bond, but only if i<j, 
                                                # since each bond is stored twice in bonds
            end
        end
    end
    return nothing
end
# NO CHANGE


# ----------------
function Arianna.store_trajectory(io, system::Atoms, t, format::Arianna.Format; digits::Integer=6)
    write_header(io, system, t, format, digits)                             # function specified in the file for the given format, 
                                                                            # writes the header for the trajectory section
    for (species, position) in zip(system.species, system.position)         # zip for multiple iterators, returns an iterator of tuples, 
                                                                            # where each tuple contains one entry from each of the iterators
        print(io, "$species")
        write_position(io, position, digits)
    end
    return nothing
end
# NO CHANGE


# ----------------
function Arianna.store_trajectory(io, system::Molecules, t, format::Arianna.Format; digits::Integer=6)
    write_header(io, system, t, format, digits)
    for (molecule, species, position) in zip(system.molecule, system.species, system.position)
        print(io, "$molecule $species")
        write_position(io, position, digits)
    end
    return nothing
end
# NO CHANGE



function Arianna.store_lastframe(io, system::Molecules, t, format::Arianna.Format; digits::Integer=6)  
    # QUESTION: Why does this function only exist for molecules and not for atoms? (???)
    # Possible answer: to only store the bonds in the last frame and for atoms there are no bonds 
    write_header(io, system, t, format, digits)
    for (molecule, species, position) in zip(system.molecule, system.species, system.position)
        print(io, "$molecule $species")
        write_position(io, position, digits)
    end
    store_bonds(io, system, format)
    return nothing
end
# NO CHANGE

end # module IO





## This part should be left unchanged 