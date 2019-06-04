
from qmmm.core.lammps.command import CommandStyle, Command
from qmmm.core.lammps.constraints import PrimitiveConstraint, ReferenceConstraint, IterableConstraint
from qmmm.core.lammps.group import Group
from qmmm.core.lammps.fix import Fix
from qmmm.core.lammps.compute import Attributes as AllAttributes, make_compute_for_atom_help, make_compute_for_atom_validator as default_validator, Compute
from abc import ABCMeta
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from os.path import join


Attributes = { attr: AllAttributes[attr] for attr in AllAttributes.keys()}

def make_compute_for_atom_validator(attributes):

    def _validate(el):
        default = default_validator(attributes)
        if isinstance(el, str):
            if el.startswith(Compute.VariablePrefix+ '_'):
                return True
            else:
                return default(el)
        else:
            return default(el)
    return _validate


class Dump(Command):
    Command = 'dump'
    Args = [PrimitiveConstraint('dump_id', str, help='the dump-ID'),
            ReferenceConstraint('group', Group, help='the ID of a group')]

    def __init__(self, *args, **kwargs):
        super(Dump, self).__init__(*args, **kwargs)
        # Initialization went well add the compute to the group
        for group in Group.resolve(self.group):
            group.dumps[self.identifier] = self
        self._data = None

    @property
    def identifier(self):
        return self.dump_id

    @property
    def data(self):
        return self._data

    @property
    def file(self):
        return self.style.file

    @file.setter
    def file(self, value):
        self.style.file = value

    @property
    def N(self):
        return self.style.N

    def parse_data(self, prefix=None):
        if prefix:
            path = join(prefix, self.file)
        else:
            path = self.file
        data = {lammps_dump.timestep: lammps_dump.data for lammps_dump in parse_lammps_dumps(path)}
        self._data = data


class DumpCommandStyle(CommandStyle, metaclass=ABCMeta):

    Style = None
    Args = [
        PrimitiveConstraint('N', int, help='dump every this many timesteps'),
        PrimitiveConstraint('file', str, help='name of file to write dump info to')
    ]


class Custom(DumpCommandStyle):

    Style = 'custom'
    Args = DumpCommandStyle.Args + [IterableConstraint('fields', element_validator=make_compute_for_atom_validator(Attributes),
                                              help=make_compute_for_atom_help)]

    def __init__(self, N, fname, fields):
        # Preprocess fields
        # If a compute is in the fields take it's variable name instead of the object
        fields = [f.format_variable() if isinstance(f, (Compute, Fix)) else f for f in fields ]
        super(Custom, self).__init__(*(N, fname, fields))