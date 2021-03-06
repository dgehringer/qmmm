
from qmmm.core.lammps.command import Command, CommandStyle
from qmmm.core.lammps.constraints import PrimitiveConstraint, ReferenceConstraint, IterableConstraint
from qmmm.core.lammps.region import Region as RegionCommand
from qmmm.core.utils import once, consecutive_groups


class Group(Command):

    Command = 'group'
    Args = [PrimitiveConstraint('group_id', str, help='the group-ID')]
    
    def __init__(self, *args, **kwargs):
        super(Group, self).__init__(*args, **kwargs)
        self._computes = {}
        self._dumps = {}
        self._fixes = {}
        self._write_dumps = {}

    @property
    def write_dumps(self):
        return self._write_dumps

    @property
    def computes(self):
        return self._computes

    @property
    def dumps(self):
        return self._dumps

    @property
    def fixes(self):
        return self._fixes

    @property
    def identifier(self):
        return self._group_id

    @once
    def delete(self):
        return Group(self.identifier, Delete)

    def clear(self):
        return Group(self.identifier, Clear)

    @classmethod
    def from_region(cls, region, id=None):
        if not id:
            gid = 'group_{}'.format(region.identifier if isinstance(region, RegionCommand) else region)
        else:
            gid = id
        return Group(gid, Region, region)

class Clear(CommandStyle):
    Style = 'clear'

class Delete(CommandStyle):
    Style = 'delete'

class Region(CommandStyle):

    Style = 'region'
    Args = [ReferenceConstraint('region', RegionCommand, help='the ID of a region')]


    @property
    def region(self):
        return RegionCommand.resolve(self._region)

class Id(CommandStyle):

    Style = 'id'
    Args = [IterableConstraint('id', help='list of one or more atom types, atom IDs, or molecule IDs any entry in list can be a sequence formatted as A:B or A:B:C where  A = starting index, B = ending index, C = increment between indices, 1 if not specified')]

    @classmethod
    def from_id_list(cls, id_list):
        id_list = list(sorted(id_list))
        crumbs = []
        for group in consecutive_groups(id_list):
            group_ = list(group)
            if len(group_) == 1:
                crumbs.append('{}'.format(group_[0]))
            else:
                # It is a consecutive region
                crumbs.append('{}:{}'.format(group_[0], group_[-1]))
        return cls(crumbs)

All = Group('all')

