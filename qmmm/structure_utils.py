from math import cos, pi, radians, sin
from pymatgen import Structure, Lattice
from ase.utils.geometry import wrap_positions as ase_wrap_positions
from ase.data import covalent_radii
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NeighborList
from qmmm.core.utils import ensure_iterable
from ase import Atoms
import numpy as np


def pymatgen_to_ase(structure):
    return AseAtomsAdaptor.get_atoms(structure)

def ase_to_pymatgen(atoms):
    return AseAtomsAdaptor.get_structure(atoms)

def view(structure, spacefill=True, show_cell=True, camera='perspective', particle_size=0.5, background='white', color_scheme='element', show_axes=True):
    try:
        import nglview
    except ImportError:
        raise ImportError('nglview is needed')
    if isinstance(structure, Atoms):
        atoms = structure
    elif isinstance(structure, Structure):
        atoms = pymatgen_to_ase(structure)
    else:
        raise TypeError
    view_  = nglview.show_ase(atoms)
    if spacefill:
        view_.add_spacefill(radius_type='vdw', color_scheme=color_scheme, radius=particle_size)
        # view.add_spacefill(radius=1.0)
        view_.remove_ball_and_stick()
    else:
        view_.add_ball_and_stick()
    if show_cell:
        if atoms.cell is not None:
            view_.add_unitcell()
    if show_axes:
        view_.shape.add_arrow([-2, -2, -2], [2, -2, -2], [1, 0, 0], 0.5)
        view_.shape.add_arrow([-2, -2, -2], [-2, 2, -2], [0, 1, 0], 0.5)
        view_.shape.add_arrow([-2, -2, -2], [-2, -2, 2], [0, 0, 1], 0.5)
    if camera != 'perspective' and camera != 'orthographic':
        print('Only perspective or orthographic is permitted')
        return None
    view_.camera = camera
    view_.background = background
    return view_

def ids(structure, sites):
    return [structure.index(site) for site in sites]

def orthogonalize_hexagonal(structure, atol=0.1):
    structure = structure.copy()
    hexagonal_lattice = structure.lattice
    structure.make_supercell([2, 2, 1])
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


def group(structure, group='all'):
    if group == 'all':
        return structure
    return Structure.from_sites([site for site in structure if any([g in ensure_iterable(site.properties['group']) for g in ensure_iterable(group)])])


def indices(structure, group='all'):
    if group == 'all':
        return np.arange(len(structure))
    return np.array([i for i, site in enumerate(structure.sites) if any([g in ensure_iterable(site.properties['group']) for g in ensure_iterable(group)])])

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


def wrap_positions(structure, center=(0.5, 0.5, 0.5), pbc=None, eps=1e-7):

    atoms = AseAtomsAdaptor.get_atoms(structure) if isinstance(structure, Structure) else structure
    if pbc is None:
        pbc = atoms.pbc
    positions = ase_wrap_positions(atoms.positions, atoms.cell,
                                    pbc, center, eps)
    return positions

def natural_cutoffs(atoms, mult=1, **kwargs):
    """Generate a radial cutoff for every atom based on covalent radii.

    The covalent radii are a reasonable cutoff estimation for bonds in
    many applications such as neighborlists, so function generates an
    atoms length list of radii based on this idea.

    * atoms: An atoms object
    * mult: A multiplier for all cutoffs, useful for coarse grained adjustment
    * kwargs: Symbol of the atom and its corresponding cutoff, used to override the covalent radii
    """
    return [kwargs.get(atom.symbol, covalent_radii[atom.number] * mult)
            for atom in atoms]


def build_neighbor_list(atoms, cutoffs=None, **kwargs):
    """Automatically build and update a NeighborList.

    Parameters:

    atoms : :class:`~ase.Atoms` object
        Atoms to build Neighborlist for.
    cutoffs: list of floats
        Radii for each atom. If not given it will be produced by calling :func:`ase.neighborlist.natural_cutoffs`
    kwargs: arbitrary number of options
        Will be passed to the constructor of :class:`~ase.neighborlist.NeighborList`

    Returns:

    return: :class:`~ase.neighborlist.NeighborList`
        A :class:`~ase.neighborlist.NeighborList` instance (updated).
    """
    if cutoffs is None:
        cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, **kwargs)
    nl.update(atoms)

    return nl
