# Generate 2D voxel grid for MOF projected along given direction
#
# Developed by: Saad Aldin Mohamed
# Date: November 11, 2025
#
#######################################################################

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from ase.io import read
from ase.build import make_supercell
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ase.spacegroup")

#--------------------- vdW atomic radii = 1/2 sigma ----------------------#
sigma = {
    'Ac':3.099, 'Ag':2.805, 'Al':4.008, 'Am':3.012, 'Ar':3.446, 'As':3.769, 'At':4.232,
    'Au':2.934, 'B':3.638, 'Ba':3.299, 'Be':2.446, 'Bi':3.893, 'Bk':2.975, 'Br':3.732, 
    'C':3.431, 'Ca':3.028, 'Cd':2.537, 'Ce':3.168, 'Cf':2.952, 'Cl':3.516, 'Cm':2.963, 
    'Co':2.559, 'Cr':2.693, 'Cs':4.024,'Cu':3.114, 'Dy':3.054, 'Er':3.021, 'Es':2.939,
    'Eu':3.112, 'F':2.997, 'Fe':2.594, 'Fm':2.927, 'Fr':4.365, 'Ga':3.905, 'Gd':3.001,
    'Ge':3.813, 'H':2.571, 'He':2.104, 'Hf':2.798, 'Hg':2.410, 'Ho':3.037, 'I':4.009,
    'In':3.976, 'Ir':2.530, 'K':3.396, 'Kr':3.689, 'La':3.138, 'Li':2.184, 'Lu':3.243,
    'Lr':2.883, 'Md':2.917, 'Mg':2.691, 'Mn':2.638, 'Mo':2.719, 'N':3.261, 'Na':2.658,
    'Nb':2.820, 'Nd':3.185, 'Ne':2.889, 'Ni':2.525, 'No':2.894, 'Np':3.050, 'O':3.118,
    'Os':2.780, 'P':3.695, 'Pa':3.050, 'Pb':3.828, 'Pd':2.583, 'Pm':3.160, 'Po':4.195,
    'Pr':3.213, 'Pt':2.454, 'Pu':3.050, 'Ra':3.276, 'Rb':3.665, 'Re':2.632, 'Rh':2.609,
    'Rn':4.245, 'Ru':2.640, 'S':3.595, 'Sb':3.938, 'Sc':2.936, 'Se':3.746, 'Si':3.826,
    'Sm':3.136,'Sn':3.913, 'Sr':3.244, 'Ta':2.824, 'Tb':3.074, 'Tc':2.671, 'Te':3.982,
    'Th':3.025, 'Ti':2.829, 'Tl':3.870, 'Tm':3.006, 'U':3.025, 'V':2.801, 'W':2.734,
    'Xe':3.924, 'Y':2.980, 'Yb':2.989, 'Zn':2.462, 'Zr':2.783
}
#------------------------ Covalent atomic radii -----------------------#
covalent_radii = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71,
    'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.70,
    'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24,
    'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20,
    'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42,
    'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
    'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98,
    'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87,
    'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41,
    'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
    'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00,
    'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69
}

#------------------------ Main code -----------------------#
def triclinic_vectors(a, b, c, alpha, beta, gamma):
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    # a vector
    ax, ay, az = a, 0, 0
    # b vector
    bx = b * np.cos(gamma_r)
    by = b * np.sin(gamma_r)
    bz = 0
    # c vector
    cx = c * np.cos(beta_r)
    cy = c * (np.cos(alpha_r) - np.cos(beta_r)*np.cos(gamma_r)) / np.sin(gamma_r)
    cz = c * np.sqrt(1 - np.cos(beta_r)**2 - (cy/c)**2)
    vector = np.array([[ax, ay, az],
                       [bx, by, bz],
                       [cx, cy, cz]])
    return vector 

def solve_2d_replication(vec1, vec2, grid_size, extra=2):
    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)
    
    n1 = int(np.ceil(grid_size / len1) * extra)
    n2 = int(np.ceil(grid_size / len2) * extra)
    
    return n1, n2
	
def compute_replications_plane(vectors, grid_size, plane="ab"):
    a_vec, b_vec, c_vec = vectors

    if plane == "ab":
        nx, ny = solve_2d_replication(a_vec[:2], b_vec[:2], grid_size)
        nz = 1
    elif plane == "ac":
        nx, nz = solve_2d_replication(a_vec[[0,2]], c_vec[[0,2]], grid_size)
        ny = 1
    elif plane == "bc":
        ny, nz = solve_2d_replication(b_vec[[1,2]], c_vec[[1,2]], grid_size)
        nx = 1
		
    return nx, ny, nz

def read_and_supercell(cif_path, plane="ab", grid_size=50.0):
    structure = read(cif_path)
    a, b, c = structure.get_cell().lengths()
    alpha, beta, gamma = structure.get_cell().angles()

    vectors = triclinic_vectors(a, b, c, alpha, beta, gamma)
    nx, ny, nz = compute_replications_plane(vectors, grid_size, plane=plane)

    P = np.diag([nx, ny, nz])
    supercell = make_supercell(structure, P)

    return structure, supercell, P

def flatten_positions(supercell, plane="ab"):
    frac = supercell.get_scaled_positions()
    cell = supercell.get_cell().array

    if plane == "ab":
        v1 = cell[0]; f1 = frac[:,0]
        v2 = cell[1]; f2 = frac[:,1]

    elif plane == "ac":
        v1 = cell[0]; f1 = frac[:,0]
        v2 = cell[2]; f2 = frac[:,2]

    elif plane == "bc":
        v1 = cell[1]; f1 = frac[:,1]
        v2 = cell[2];f2 = frac[:,2]
        
    positions_2d = np.outer(f1, v1[:3]) + np.outer(f2, v2[:3])

    e1 = v1 / np.linalg.norm(v1)
    v2p = v2 - np.dot(v2, e1) * e1
    e2 = v2p / np.linalg.norm(v2p)

    x = np.dot(positions_2d, e1)
    y = np.dot(positions_2d, e2)

    flattened = np.column_stack((x, y))
    return flattened

def get_replicated_elements(structure, P):
    orig_elements = [atom.symbol for atom in structure]
    nx, ny, nz = P.diagonal().astype(int)
    return np.tile(orig_elements, nx * ny * nz)

def center_and_cut(positions, elements, r='vdw', grid_size=50.0):
    pos_2d = positions[:, :2]
    if r=='vdw':
        r_vdw_array = 0.5*np.array([sigma[el] for el in elements])
    elif r=='cov':
        r_vdw_array = np.array([covalent_radii[el] for el in elements])	
    else:
        r_vdw_array = np.full(pos_2d.shape[0], r)
    center = pos_2d.mean(axis=0)
    positions_centered = pos_2d - center
    cut_half = grid_size / 2
    mask = (
        (positions_centered[:,0]+r_vdw_array >= -cut_half) &
        (positions_centered[:,0]-r_vdw_array <= cut_half) &
        (positions_centered[:,1]+r_vdw_array >= -cut_half) &
        (positions_centered[:,1]-r_vdw_array <= cut_half)
    )
    positions_selected = positions_centered[mask] + cut_half
    elements_selected = np.array(elements)[mask]
    return positions_selected, elements_selected

def calculate_2d_grid(positions, elements, r='vdw', grid_size=50.0, resolution=500):
    dx = dy = grid_size / resolution
    grid_2d = np.ones((resolution,resolution), dtype=int)
    for idx, pos in enumerate(positions):
        if r=='vdw': 
            r_atom = 0.5*sigma[elements[idx]]
        elif r=='cov': 
            r_atom = covalent_radii[elements[idx]]	
        else: 
            r_atom = r
        r_x = int(np.ceil(r_atom/dx))
        r_y = int(np.ceil(r_atom/dy))
        i_center, j_center = int(np.floor(pos[0]/dx)), int(np.floor(pos[1]/dy))
        i_min, i_max = max(0,i_center-r_x), min(resolution-1,i_center+r_x)
        j_min, j_max = max(0,j_center-r_y), min(resolution-1,j_center+r_y)
        for i in range(i_min,i_max+1):
            for j in range(j_min,j_max+1):
                x_vox = (i+0.5)*dx
                y_vox = (j+0.5)*dy
                if (x_vox - pos[0])**2 + (y_vox - pos[1])**2 <= r_atom**2:
                    grid_2d[i,j] = 0
    return grid_2d

def save_plot_grid(grid, cifname, out_dir, plane="ab", grid_size=50.0, resolution=500, r='vdw'):
    save_dir = os.path.join(out_dir, f"{cifname}_2D_grid_{plane}_{r}")
	
    plt.figure(figsize=(4,4))
    plt.imshow(grid.T, origin='lower', cmap='gray', interpolation='nearest', extent=[0,grid_size,0,grid_size])
    #plt.xlabel("X (Å)"); plt.ylabel("Y (Å)")
    plt.yticks([])
    plt.xticks([])
    plt.axis("off")
	
    plt.savefig(save_dir+".png", dpi=resolution, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    grid_oriented = np.flipud(grid.T)  
    np.save(save_dir+".npy", grid_oriented)

def main(cif_path, out_dir, plane="ab", resolution=500, grid_size=50.0, r='vdw', save_img=True):
    structure, supercell, P = read_and_supercell(cif_path, plane, grid_size)
    positions_cart = flatten_positions(supercell, plane)
    elements_replicated = get_replicated_elements(structure, P)
    positions_shifted, elements_selected = center_and_cut(positions_cart, elements_replicated, r, grid_size)
    grid = calculate_2d_grid(positions_shifted, elements_selected, r, grid_size, resolution)
    if save_img:
        cifname = os.path.splitext(os.path.basename(cif_path))[0]
        save_plot_grid(grid, cifname, out_dir, plane, grid_size, resolution, r)
    return grid

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate 2D voxelized images from CIFs")

    parser.add_argument("--cifdir", default=".", help="Directory containing CIF files (default: current directory)")
    parser.add_argument("--outdir", default=None, help="Directory for generated images (default: same as --cifdir)")
    parser.add_argument("--r", default="vdw", help="Atomic radii: vdw (default), cov, or a float value in Å")
    parser.add_argument("--res", type=int, default=500, help="Image resolution in pixels (default: 500)")
    parser.add_argument("--grid", type=float, default=50.0, help="Grid size in Å (default: 50)")

    args = parser.parse_args()

    in_dir = os.path.abspath(args.cifdir)
    if not os.path.isdir(in_dir): in_dir = os.getcwd()
    out_dir = os.path.abspath(args.outdir) if args.outdir else in_dir
	
    cif_files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".cif")])

    if len(cif_files) == 0:
        raise RuntimeError("No CIF files found in directory")

    r_input = args.r.lower() if isinstance(args.r, str) else args.r
    
    if r_input in ["vdw", "cov"]:
        r_value = r_input
    else:
        try:
            r_value = float(r_input)
        except ValueError:
            print(f"Warning: Invalid --r value '{args.r}', using default 'vdw'")
            r_value = "vdw"

    print(f"\nFound {len(cif_files)} CIF files")
    print(f"Input: {in_dir}")
    print(f"Output: {out_dir}")

    failed = []

    for cif in cif_files:
        cif_path = os.path.join(in_dir, cif)
        for plane in ["ab", "ac", "bc"]:
            try:
                print(f"Processing {cif} → plane {plane}")
                main(cif_path, out_dir=out_dir, plane=plane, resolution=args.res, 
						grid_size=args.grid, r=r_value)
            except Exception as e:
                print(f"failed: {cif} → plane {plane} : {e}")
                failed.append(f"{cif} ({plane})")	