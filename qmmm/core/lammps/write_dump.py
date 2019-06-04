
from .command import CommandStyle, Command
from .group import Group
from qmmm.core.lammps.constraints import ReferenceConstraint, PrimitiveConstraint, IterableConstraint
from .dump import make_compute_for_atom_validator, make_compute_for_atom_help, Attributes
from .fix import Fix
from .compute import Compute
from abc import ABCMeta
from os.path import join
from pymatgen.io.lammps.outputs import parse_lammps_dumps


class WriteDump(Command):

    Command = 'write_dump'
    Args = [ReferenceConstraint('group', Group, help='the ID of a group')]

    def __init__(self, *args, **kwargs):
        super(WriteDump, self).__init__(*args, **kwargs)
        # Initialization went well add the compute to the group
        for group in Group.resolve(self.group):
            group.write_dumps[self.identifier] = self
        self._data = None

    @property
    def file(self):
        return self.style.file

    @file.setter
    def file(self, value):
        self.style.file = value

    def parse_data(self, prefix=None):
        if prefix:
            path = join(prefix, self.file)
        else:
            path = self.file
        dump_obj = list(parse_lammps_dumps(path))
        assert len(dump_obj) == 1
        dump_data = dump_obj[0].data
        self._data = dump_data

    @property
    def data(self):
        return self._data

class DumpCommandStyle(CommandStyle, metaclass=ABCMeta):

    Style = None
    Args = [
        PrimitiveConstraint('file', str, help='name of file to write dump info to')
    ]


class Custom(DumpCommandStyle):

    Style = 'custom'
    Args = DumpCommandStyle.Args + [IterableConstraint('fields', element_validator=make_compute_for_atom_validator(Attributes),
                                              help=make_compute_for_atom_help)]


    def __init__(self, fname, fields):
        # Preprocess fields
        # If a compute is in the fields take it's variable name instead of the object
        fields = [f.format_variable() if isinstance(f, (Compute, Fix)) else f for f in fields ]
        super(Custom, self).__init__(*(fname, fields))