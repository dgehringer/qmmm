from qmmm.core.lammps.command import Command, CommandStyle
from qmmm.core.lammps.constraints import PrimitiveConstraint, IterableConstraint, ChoiceConstraint, ReferenceConstraint
from qmmm.core.lammps.region import Region
from qmmm.core.lammps.group import Group
from qmmm.core.lammps.fix import Fix

class Clear(Command):
    Command = 'clear'

    @classmethod
    def styles(cls):
        return ()


class Lj(CommandStyle):
    Style = 'lj'


class Metal(CommandStyle):
    Style = 'metal'


class Real(CommandStyle):
    Style = 'real'


class Si(CommandStyle):
    Style = 'si'


class Cgs(CommandStyle):
    Style = 'cgs'


class AtomStyle(Command):
    Command = 'atom_style'


class Atomic(CommandStyle):
    Style = 'atomic'


class Units(Command):
    Command = 'units'

    @classmethod
    def styles(cls):
        return (Lj, Metal, Real, Si, Cgs)


class Eam(CommandStyle):
    Style = 'Eam'


class EamAlloy(CommandStyle):
    Style = 'eam/alloy'

class EamFs(CommandStyle):
    Style = 'eam/fs'


class Meam(CommandStyle):
    Style = 'meam'


class PairStyle(Command):
    Command = 'pair_style'

    @classmethod
    def styles(cls):
        return (Eam, EamAlloy, Meam, EamFs)


class TimeStep(Command):

    Command = 'timestep'

    Args = [PrimitiveConstraint('dt', (float, int), help='timestep size (time units)')]


class PairCoeff(Command):
    Command = 'pair_coeff'
    Args = [
        PrimitiveConstraint('i', (str, int), help='atom type i ( i <= j )'),
        PrimitiveConstraint('j', (str, int), help='atom type j ( i <= j )'),
        IterableConstraint('coeff', help='coefficients for one or more pairs of atom types')
    ]

    @classmethod
    def styles(cls):
        return ()


class ResetTimeStep(Command):
    Command = 'reset_timestep'

    Args = [
        PrimitiveConstraint('N', int, help='timestep number ')
    ]

    @classmethod
    def styles(cls):
        return ()


class Cg(CommandStyle):
    Style = 'cg'


class Sd(CommandStyle):
    Style = 'sd'


class Hftn(CommandStyle):
    Style = 'Hftn'


class MinStyle(Command):
    Command = 'min_style'

    @classmethod
    def styles(cls):
        return (Cg, Sd, Hftn)


class Minimize(Command):
    Command = 'minimize'

    Args = [
        PrimitiveConstraint('etol', float, help='stopping tolerance for energy (unitless)'),
        PrimitiveConstraint('ftol', float, help='stopping tolerance for force (force units)'),
        PrimitiveConstraint('maxiter', int, help='max iterations of minimizer'),
        PrimitiveConstraint('maxeval', int, help='max number of force/energy evaluations')
    ]

    @classmethod
    def styles(cls):
        return ()

class Boundary(Command):

    Command = 'boundary'

    Args = [
        ChoiceConstraint('x', ['p', 'f', 's', 'fs', 'fm']),
        ChoiceConstraint('y', ['p', 'f', 's', 'fs', 'fm']),
        ChoiceConstraint('z', ['p', 'f', 's', 'fs', 'fm']),
    ]

    @classmethod
    def styles(cls):
        return ()

class Dimension(Command):

    Command = 'dimension'

    Args = [PrimitiveConstraint('dim', int)]

    @classmethod
    def styles(cls):
        return ()

class ReadData(Command):

    Command = 'read_data'

    Args = [PrimitiveConstraint('filename', str, help='name of data file to read in')]

    Keywords = {
        'group': ReferenceConstraint('group_id', Group, help='add atoms in data file to this group')
    }
    @classmethod
    def styles(cls):
        return ()

class WriteData(Command):

    Command = 'write_data'

    Args = [PrimitiveConstraint('filename', str, help='name of data file to read in')]

    @classmethod
    def styles(cls):
        return ()

class CreateBox(Command):

    Command = 'create_box'

    Args = [PrimitiveConstraint('N', int, help='# of atom types to use in this simulation'),
            ReferenceConstraint('region_id', Region, help='ID of region to use as simulation domain')]

    @classmethod
    def styles(cls):
        return ()


class Print(Command):

    Command = 'print'

    Args = [PrimitiveConstraint('string', str, help='text string to print, which may contain variables')]

    Keywords = {
        'file' : PrimitiveConstraint('filename', str, help='filename is specified to which the output will be written'),
        'append' : PrimitiveConstraint('filename', str, help='filename is specified to which the output will be written'),
        'screen' : ChoiceConstraint('value', ['yes', 'no']),
        'universe' : ChoiceConstraint('value', ['yes', 'no']),
    }


class Run(Command):

    Command = 'run'

    Args = [PrimitiveConstraint('N', int, help='# of timesteps')]

    @classmethod
    def styles(cls):
        return ()

class ThermoModify(Command):

    Command = 'thermo_modify'

    Keywords = {
        'format': PrimitiveConstraint('format', str)
    }

    @classmethod
    def styles(cls):
        return ()


class Velocity(Command):

    Command = 'velocity'

    Args = [
        ReferenceConstraint('gid', Group, help='ID of group of atoms whose velocity will be changed')
    ]

    Keywords = {
        'dist': ChoiceConstraint('value', ['uniform', 'gaussian']),
        'loop': ChoiceConstraint('value', ['local', 'geom']),
        'units': ChoiceConstraint('value', ['box', 'lattice']),
        'sum': ChoiceConstraint('value', ['yes', 'no']),
        'rot': ChoiceConstraint('value', ['yes', 'no']),
        'mom': ChoiceConstraint('value', ['yes', 'no']),
        'bias': ChoiceConstraint('value', ['yes', 'no']),

    }

    @classmethod
    def styles(cls):
        return (Create, )

class Create(CommandStyle):

    Style = 'create'

    Args = [
        PrimitiveConstraint('temp', (float, int), help='temperature value (temperature units)'),
        PrimitiveConstraint('seed', (int,), help='random # seed (positive integer)'),
    ]

class Unfix(Command):

    Command = 'unfix'
    Args = [
        ReferenceConstraint('fix-ID', Fix, help='ID of a previously defined fix')
    ]