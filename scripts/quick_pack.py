import subprocess
import argparse

def generate_packmol_input(pdb_file, box_size):
    """
    Generate a Packmol input file to pack a given pdb into the specified box size.
    
    :param pdb_file: Input PDB file
    :param box_size: A tuple (x, y, z) defining the box dimensions
    :param output_file: Path to the output file (e.g., 'packmol_input.inp')
    :param center_molecule: Flag to determine if the molecule should be fixed at the box center
    """
    # Box dimensions (x, y, z)
    x_dim, y_dim, z_dim = box_size

    # Creating the Packmol input file
    with open('pack.inp', 'w') as f:
        f.write(f"tolerance 2.0\n")
        f.write(f"filetype pdb\n")
        f.write(f"output conf.pdb\n")
        f.write(f"seed 1234567\n")
        f.write(f"randominitialpoint\n")

        f.write(f"structure {pdb_file}\n")
        f.write(f"  number 1\n")  # Number of molecules to pack
        f.write(f"  center\n")
        f.write(f"  fixed {x_dim/2} {y_dim/2} {z_dim/2} 0. 0. 0.\n")
        f.write(f"end structure\n")

    print(f"Packmol input file generated: pack.inp")

def run_packmol(pdb_file, box_size):
    """
    Run Packmol with the specified input file.
    
    :param packmol_input_file: Path to the Packmol input file
    """
    generate_packmol_input(pdb_file, box_size)
    try:
        # Run the Packmol executable
        subprocess.run("packmol < pack.inp", shell=True)
        print("Packmol ran successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Packmol: {e}")
        return False
    return True

def parse_arguments():
    """
    Parse command line arguments for PDB file and box size.
    """
    parser = argparse.ArgumentParser(description="Pack a PDB file into a specified box using Packmol.")
    
    # Required arguments
    parser.add_argument('pdb_file', type=str, help="Path to the PDB file")
    
    # Box size arguments
    parser.add_argument('x', type=float, help="Box size along X-axis (in Nanometers)")
    parser.add_argument('y', type=float, help="Box size along Y-axis (in Nanometers)")
    parser.add_argument('z', type=float, help="Box size along Z-axis (in Nanometers)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Box size as a tuple
    box_size = (args.x*10, args.y*10, args.z*10)

    # Pack the PDB into the box with one molecule fixed at the center
    run_packmol(args.pdb_file, box_size)

    # add the pbc box line
    with open('conf.pdb', 'r') as f:
        lines = f.readlines()
    line_idx = 0
    for line in lines:
        if line.startswith("ATOM"):
            break
        line_idx += 1
    box_line = "CRYST1{:>9.2f}{:>9.2f}{:>9.2f}  90.00  90.00  90.00 P 1           1\n".format(*box_size)
    lines.insert(line_idx, box_line)
    with open("conf.pdb", 'w') as f:
        lines = "".join(lines)
        f.write(lines)
    
    print("Finished!")
