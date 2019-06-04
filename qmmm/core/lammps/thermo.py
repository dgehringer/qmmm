from qmmm.core.lammps.command import CommandStyle, Command
from qmmm.core.lammps.constraints import PrimitiveConstraint, IterableConstraint
from qmmm.core.lammps.compute import make_compute_for_atom_validator, make_compute_for_atom_help, Compute
from qmmm.core.lammps.fix import Fix

Attributes = {
    'step': 'timestep',
    'elapsed': 'timesteps since start of this run',
    'elaplong': 'timesteps since start of initial run in a series of runs',
    'dt': 'timestep size',
    'time': 'simulation time',
    'cpu': 'elapsed CPU time in seconds since start of this run',
    'tpcpu': 'time per CPU second',
    'spcpu': 'timesteps per CPU second',
    'cpuremain': 'estimated CPU time remaining in run',
    'part': 'which partition (0 to Npartition-1) this is',
    'timeremain': 'remaining time in seconds on timer timeout.',
    'atoms': '# of atoms',
    'temp': 'temperature',
    'press': 'pressure',
    'pe': 'total potential energy',
    'ke': 'kinetic energy',
    'etotal': 'total energy (pe + ke)',
    'enthalpy': 'enthalpy (etotal + press*vol)',
    'evdwl': 'VanderWaal pairwise energy (includes etail)',
    'ecoul': 'Coulombic pairwise energy',
    'epair': 'pairwise energy (evdwl + ecoul + elong)',
    'ebond': 'bond energy',
    'eangle': 'angle energy',
    'edihed': 'dihedral energy',
    'eimp': 'improper energy',
    'emol': 'molecular energy (ebond + eangle + edihed + eimp)',
    'elong': 'long-range kspace energy',
    'etail': 'VanderWaal energy long-range tail correction',
    'vol': 'volume',
    'density': 'mass density of system',
    'lx': 'box length in x',
    'ly': 'box length in y',
    'lz': 'box length in z',
    'xlo': 'box boundaries',
    'ylo': 'box boundaries',
    'zlo': 'box boundaries',
    'xhi': 'box boundaries',
    'yhi': 'box boundaries',
    'zhi': 'box boundaries',
    'xy': 'box tilt for triclinic (non-orthogonal) simulation boxes',
    'yz': 'box tilt for triclinic (non-orthogonal) simulation boxes',
    'xz': 'box tilt for triclinic (non-orthogonal) simulation boxes',
    'xlat': 'lattice spacings as calculated by lattice command',
    'ylat': 'lattice spacings as calculated by lattice command',
    'zlat': 'lattice spacings as calculated by lattice command',
    'bonds': '# of these interactions defined',
    'angles': '# of these interactions defined',
    'dihedrals': '# of these interactions defined',
    'mpropers': '# of these interactions defined',
    'pxx': '6 components of pressure tensor',
    'pyy': '6 components of pressure tensor',
    'pzz': '6 components of pressure tensor',
    'pxy': '6 components of pressure tensor',
    'pxz': '6 components of pressure tensor',
    'pyz': '6 components of pressure tensor',
    'fmax': 'max component of force on any atom in any dimension',
    'fnorm': 'length of force vector for all atoms',
    'nbuild': '# of neighbor list builds',
    'ndanger': '# of dangerous neighbor list builds',
    'cella': 'periodic cell lattice constant a',
    'cellb': 'periodic cell lattice constant b',
    'cellc': 'periodic cell lattice constant c',
    'cellalpha': 'periodic cell angle alpha',
    'cellbeta': 'periodic cell angle alpha',
    'cellgamma': 'periodic cell angle alpha'
}



class One(CommandStyle):

    Style = 'one'

class Multi(CommandStyle):

    Style = 'multi'

class Custom(CommandStyle):

    Style = 'custom'

    Args = [IterableConstraint('fields', element_validator=make_compute_for_atom_validator(Attributes),
                        help=make_compute_for_atom_help)]

    def __init__(self,fields):
        # Preprocess fields
        # If a compute is in the fields take it's variable name instead of the object
        fields = [f.format_variable() if isinstance(f, (Compute, Fix)) else f for f in fields]
        super(Custom, self).__init__(*(fields,))

class ThermoStyle(Command):

    Command = 'thermo_style'

    @classmethod
    def styles(cls):
        return (One, Multi, Custom)

class Thermo(Command):

    Command = 'thermo'

    Args = [
        PrimitiveConstraint('N', int, help='output thermodynamics every N timesteps')
    ]
