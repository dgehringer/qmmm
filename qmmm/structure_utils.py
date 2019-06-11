from math import cos, pi, radians, sin
from pymatgen import Structure, Lattice
import numpy as np

def orthogonalize_hexagonal(structure, atol=0.1):
    structure = structure.copy()
    hexagonal_lattice = structure.lattice
    structure.make_supercell([2, 2, 1])
    print(hexagonal_lattice.b * 2.0 * sin(radians(hexagonal_lattice.gamma)), hexagonal_lattice.gamma, hexagonal_lattice.b, sin(radians(hexagonal_lattice.gamma)))
    orthogonal_lattice = Lattice.from_parameters(
        hexagonal_lattice.a,
        hexagonal_lattice.b * 2.0 * sin(radians(hexagonal_lattice.gamma)),
        hexagonal_lattice.c,
        90, 90, 90
    )
    species = [s.species_string for s in structure]
    frac_coords = [orthogonal_lattice.get_fractional_coords(s.coords) for s in structure]
    site_properties = structure.site_properties

    orthogonal =  Structure(orthogonal_lattice, species, frac_coords, site_properties=site_properties)
    d = orthogonal.distance_matrix
    close_indices = np.argwhere(d <= atol)
    # Eliminate main diogonal
    close_indices = close_indices[np.argwhere(close_indices[:,0] < close_indices[:,1])][:, 0,:]
    # Take just one of the indices
    indices_to_remove = [j for i, j in close_indices]
    orthogonal.remove_sites(indices_to_remove)
    return orthogonal

def layers(structure, atol=None):
    if not atol:
        atol = 0.1/structure.lattice.c
    height_classes = {}
    for s in structure.sites:
        fa, fb, fc = s.frac_coords
        class_found = False
        for k in height_classes.keys():
            if fc - atol <= k <= fc + atol:
                height_classes[k].append(s)
                class_found = True
                break
        if not class_found:
            height_classes[fc] = [s]
    return height_classes

def spacing(lattice, plane):
    return  2.0 * pi / np.linalg.norm(sum([miller * vec  for vec, miller in zip(lattice.reciprocal_lattice.matrix, plane)]))


def build_shell(structure, seed_sites, distance, include_index=True):
    shell_sites = {}
    for seed_site in seed_sites:
        neighbors = structure.get_sites_in_sphere(seed_site.coords, distance, include_index=True)
        for site, _, index in neighbors:
            if index not in shell_sites:
                shell_sites[index] = site
    return [(k, v) for k, v in shell_sites.items()] if include_index else list(shell_sites.values())

