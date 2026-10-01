struct XYZ <: Arianna.Format
    extension::String
    function XYZ()
        return new(".xyz")
    end
end

# a ? b : c means if a is true, b is executed and otherwise c is executed 
function get_selrow(::XYZ, N, m)
    return m ≥ 0 ? (N + 2) * m - N + 1 : length(data) + m * (N + 2) + 3
    # data is the file storing the trajectory (in some cases anyway, don't know if there are others...)
    # (N + 2) should be the number lines containing header and positions of one snapshot of a trajectory
    # m should be the index of the frame, positive m counts from beginning, negative m counts from end  
    # m = 0 returns a negative number which should be prohibite (???)  
end
# function returns the number of the line in the file where the actual entries of the first snapshot to be evaluated starts 
# NO CHANGE


function parse_column_string(column_str::AbstractString, ::XYZ; d::Int=3)
    columns = split(column_str, ",")
    column_info = OrderedDict{String,Vector}() # Use OrderedDict to maintain order
    index = 1
    for column_name in columns
        if column_name == "molecule"
            dimension = 1
            column_info[column_name] = [dimension, index]
        elseif column_name == "species"
            dimension = 1
            column_info[column_name] = [dimension, index]
        elseif column_name == "position"
            column_info["pos"] = [d, index]
        elseif column_name == "bond"
            dimension = 2
            column_info["bond"] = [2, index]
        elseif column_name == "btype"
            dimension = 1
            column_info[column_name] = [dimension, index]
        else
            error("$column_name is not supported")
        end
        index += 1
    end
    return column_info
    # returns an ordered dictionary containing pairs of 
    # "name" -> [dimension of the stored information, index indicating the position of the information stored in "name"]
end
# NO CHANGE

function read_header(data, format::XYZ)
    N = parse(Int, data[1])  # Number of atoms or entries
    metadata = split(data[2], " ")  # Metadata split into an array

    # Extract cell vector from metadata
    cell_str = replace(metadata[findfirst(startswith("cell:"), metadata)], "cell:" => "")   # delete the "cell:" part of the string, so that only the numbers remain
    cell_vector = parse.(Float64, split(cell_str, ","))                                     # in the data file the first line contains something like 
                                                                                            # cell:13.572088082974531,13.572088082974531,13.572088082974531, 
                                                                                            # so this line takes this and splits the entries at the , 
                                                                                            # so that this is now a 3d array of the lengths of the cell boundaries
    d = length(cell_vector)                                                                 # the dimension of the cell
    box = SVector{d}(cell_vector)                                                           # box is now a d-dimensional static Vector containing the length of the 
                                                                                            # cell in each spatial direction 
    column_str = replace(metadata[findfirst(startswith("columns:"), metadata)], "columns:" => "")
    column_info = parse_column_string(column_str, format; d=d)
    return N, box, column_info, metadata
end
# NO CHANGE

function get_system_column(::Atoms, ::XYZ)
    return ""
end
# NO CHANGE

function get_system_column(::Molecules, ::XYZ)
    return "molecule,"
end
# NO CHANGE

function get_row_bonds(selrow, N, ::XYZ)
    return 2
end
# NO CHANGE

function read_bonds_header(bonds, format::XYZ)
    N_bonds = parse(Int, bonds[1])
    column_line = split(bonds[2], " ")
    column_line isa String ? [column_line] : column_line
    column_str = replace(column_line[findfirst(startswith("columns:"), column_line)], "columns:" => "")
    column_info = parse_column_string(column_str, format)       # column_info is an ordered dictionary containing pairs of "name" -> 
                                                                # [dimension of the stored information, index indicating the position 
                                                                # of the information stored in "name"]
                                                                # so in this case mostely "bond" -> [2, 1] and "btype" -> [1, 2]
    return N_bonds, column_info
end
# NO CHANGE

function write_bonds_header(io, ::XYZ)
    println(io, "columns:bond")
    return nothing
end
# NO CHANGE

function write_header(io, system::Particles, t, format::XYZ, digits::Integer)
    println(io, length(system))
    box = replace(replace(string(system.box), r"[\[\]]" => ""), r",\s+" => ",")
    println(io, "step:$t columns:$(get_system_column(system, format))species,position dt:1 cell:$(box) rho:$(system.density) T:$(system.temperature)")
    return nothing
end
# NO CHANGE

