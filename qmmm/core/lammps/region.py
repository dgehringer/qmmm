from abc import ABCMeta
from qmmm.core.utils import is_iterable
from qmmm.core.lammps.command import Command, CommandStyle
from qmmm.core.lammps.constraints import PrimitiveConstraint, ChoiceConstraint, IterableConstraint

class Region(Command):
    Command = 'region'
    Args = [PrimitiveConstraint('region_id', str, help='the region-ID')]
    Keywords = {
        'side': ChoiceConstraint('value', ['in', 'out'], help=['the region is inside the specified geometry',
                                                               'the region is outside the specified geometry']),
        'units': ChoiceConstraint('value', ['lattice', 'box'], help=['the geometry is defined in lattice units',
                                                                     'the geometry is defined in box units']),
        'move': [
            PrimitiveConstraint('v_x', (float, int), help='equal-style variable x displacement of region over time'),
            PrimitiveConstraint('v_y', (float, int), help='equal-style variable y displacement of region over time'),
            PrimitiveConstraint('v_z', (float, int), help='equal-style variable z displacement of region over time')],
        'rotate': [
            PrimitiveConstraint('v_theta', (float, int), help='equal-style variable for rotaton of region over time (in radians)'),
            PrimitiveConstraint('Px', (float, int), help='origin for axis of rotation (distance units)'),
            PrimitiveConstraint('Py', (float, int), help='origin for axis of rotation (distance units)'),
            PrimitiveConstraint('Pz', (float, int), help='origin for axis of rotation (distance units)'),
            PrimitiveConstraint('Rx', (float, int), help='axis of rotation vector'),
            PrimitiveConstraint('Ry', (float, int), help='axis of rotation vector'),
            PrimitiveConstraint('Rz', (float, int), help='axis of rotation vector'),
        ],
        'open': PrimitiveConstraint('value', int, help='from 1-6 corresponding to face index')
    }

    @property
    def identifier(self):
        return self._region_id


class Cylinder(CommandStyle):

    Style = 'cylinder'
    Args = [ChoiceConstraint('dim', ['x', 'y', 'z'], help='axis of cylinder'),
            PrimitiveConstraint('c1', (float, int), help='coords of cylinder axis in other 2 dimensions (distance units)'),
            PrimitiveConstraint('c2', (float, int), help='coords of cylinder axis in other 2 dimensions (distance units)'),
            PrimitiveConstraint('radius', (float, int), 'cylinder radius (distance units)'),
            PrimitiveConstraint('lo', (float, int), help='bounds of cylinder in dim (distance units)'),
            PrimitiveConstraint('hi', (float, int), help='bounds of cylinder in dim (distance units)')]

class Block(CommandStyle):

    Style = 'block'
    Args =  [PrimitiveConstraint('xlo', (float, int), help='bounds of block in all dimensions (distance units)'),
        PrimitiveConstraint('xhi', (float, int), help='bounds of block in all dimensions (distance units)'),
        PrimitiveConstraint('ylo', (float, int), help='bounds of block in all dimensions (distance units)'),
        PrimitiveConstraint('yhi', (float, int), help='bounds of block in all dimensions (distance units)'),
        PrimitiveConstraint('zlo', (float, int), help='bounds of block in all dimensions (distance units)'),
        PrimitiveConstraint('zhi', (float, int), help='bounds of block in all dimensions (distance units)')
    ]

class OperationsRegions(CommandStyle, metaclass=ABCMeta):

    Style = None
    Args = [IterableConstraint('regions', element_validator=lambda el: Region.exists(el), help='region IDs')]

    def __init__(self, *args):
        if not is_iterable(args[0]):
            raise TypeError('Argument "regions" must be iterable')
        args =([a.region_id if isinstance(a, Region) else a for a in args[0]], )
        super(OperationsRegions, self).__init__(*args)

    def format_arguments(self):
        return '{} {}'.format(len(self._regions), super(OperationsRegions, self).format_arguments())


    @classmethod
    def from_string(cls, string, preprocess=None, return_remaining=False, make_obj=True):
        # Define custom argument preprocessor
        def _preprocess_args(args):
            # Omit the first argument
            crumbs = [c for c in args[0].split(' ') if c != '']
            return crumbs[1:]
        return super().from_string(string, preprocess=_preprocess_args if not preprocess else preprocess,
                                   return_remaining=return_remaining)

    @property
    def regions(self):
        return [Region.resolve(rid) for rid in self._regions]

    def add_region(self, region):
        if isinstance(region, Region):
            region = region.region_id
        elif isinstance(region, str):
            pass
        else:
            raise TypeError('region argument must be a Region object')
        for r in self.regions:
            if r.region_id == region:
                raise KeyError('Region {} is already in the {}'.format(r.region_id, type(self).Style))
        it_type = type(self._regions)
        self._regions = it_type(list(self._regions) + [region])

class Union(OperationsRegions):

    Style = 'union'

class Intersect(OperationsRegions):

    Style = 'intersect'

