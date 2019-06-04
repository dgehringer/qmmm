from qmmm.core.lammps.command import Command, CommandStyle
from qmmm.core.lammps.constraints import PrimitiveConstraint, IterableConstraint, ReferenceConstraint, ChoiceConstraint
from qmmm.core.lammps.group import Group
from qmmm.core.lammps.fix import Fix
from qmmm.core.lammps.region import Region
from qmmm.core.utils import indent
from numpy import array


Attributes = {
    'id': 'atom ID',
    'mol': 'molecule ID',
    'proc': 'ID of processor that owns atom',
    'type': 'atom type',
    'mass': 'atom mass',
    'element': 'element',
    'x': 'unscaled atom coordinates',
    'y': 'unscaled atom coordinates',
    'z': 'unscaled atom coordinates',
    'xs': 'scaled atom coordinates',
    'ys': 'scaled atom coordinates',
    'zs': 'scaled atom coordinates',
    'xu': 'unwrapped atom coordinates',
    'yu': 'unwrapped atom coordinates',
    'zu': 'unwrapped atom coordinates',
    'ix': 'box image that the atom is in',
    'iy': 'box image that the atom is in',
    'iz': 'box image that the atom is in',
    'vx': 'atom velocities',
    'vy': 'atom velocities',
    'vz': 'atom velocities',
    'fx': 'forces on atoms',
    'fy': 'forces on atoms',
    'fz': 'forces on atoms',
    'q': 'atom charge',
    'mux': 'orientation of dipole moment of atom',
    'muy': 'orientation of dipole moment of atom',
    'muz': 'orientation of dipole moment of atom',
    'mu': 'magnitude of dipole moment of atom',
    'sp': 'atomic magnetic spin moment',
    'spx': 'direction of the atomic magnetic spin',
    'spy': 'direction of the atomic magnetic spin',
    'spz': 'direction of the atomic magnetic spin',
    'fmx': 'magnetic force',
    'fmy': 'magnetic force',
    'fmz': 'magnetic force',
    'radius': 'radius',
    'diameter': 'diameter of spherical particle',
    'omegax': 'angular velocity of spherical particle',
    'omegay': 'angular velocity of spherical particle',
    'omegaz': 'angular velocity of spherical particle',
    'angmomx': 'angular momentum of aspherical particle',
    'angmomy': 'angular momentum of aspherical particle',
    'angmomz': 'angular momentum of aspherical particle',
    'shapex': 'diameters of aspherical particle',
    'shapey': 'diameters of aspherical particle',
    'shapez': 'diameters of aspherical particle',
    'quatw': 'quaternion components for aspherical or body particles',
    'quati': 'quaternion components for aspherical or body particles',
    'quatj': 'quaternion components for aspherical or body particles',
    'quatk': 'quaternion components for aspherical or body particles',
    'tqx': 'torque on finite-size particles',
    'tqy': 'torque on finite-size particles',
    'tqz': 'torque on finite-size particles',
    'end1x': 'end points of line segment',
    'end1y': 'end points of line segment',
    'end1z': 'end points of line segment',
    'end2x': 'end points of line segment',
    'end2y': 'end points of line segment',
    'end2z': 'end points of line segment',
    'corner1x': 'corner points of triangle',
    'corner1y': 'corner points of triangle',
    'corner1z': 'corner points of triangle',
    'corner2x': 'corner points of triangle',
    'corner2y': 'corner points of triangle',
    'corner2z': 'corner points of triangle',
    'corner3x': 'corner points of triangle',
    'corner3y': 'corner points of triangle',
    'corner3z': 'corner points of triangle',
    'nbonds': 'number of bonds assigned to an atom'
}

def make_compute_for_atom_help():
    first_line = ' '.join(Attributes) + '\n'
    attr_lines = indent(''.join(['{} = {}\n'.format(attr, Attributes[attr]) for attr in Attributes.keys()]))
    return first_line + attr_lines

def make_compute_for_atom_validator(attributes):

    def _validate(el):
        default = lambda el: el in attributes
        if isinstance(el, str):
            if el.startswith(Compute.VariablePrefix+ '_'):
                return True
            else:
                return default(el)
        else:
            return default(el)
    return _validate


class Compute(Command):
    Command = 'compute'
    Args = [PrimitiveConstraint('compute_id', str, help='the compute-ID'),
            ReferenceConstraint('group', Group, help='the ID of a group')]
    VariablePrefix = 'c'

    def __init__(self, *args, **kwargs):
        super(Compute, self).__init__(*args, **kwargs)
        # Initialization went well add the compute to the group
        for group in Group.resolve(self.group):
            group.computes[self.identifier] = self

    @property
    def identifier(self):
        return self.compute_id

    def __getitem__(self, item):
        # Pass __getitem__ to style
        return self.style[(self, item)]

    def format_variable(self):
        return '{}_{}'.format(self.VariablePrefix, self.identifier)

class PerAtom(CommandStyle):

    Style = 'property/atom'
    Args = [IterableConstraint('fields', element_validator=make_compute_for_atom_validator(Attributes),
                                              help=make_compute_for_atom_help)]


    def format_variable(self, compute, index=None):
        base = '{}_{}'.format(compute.VariablePrefix, compute.identifier)
        if index is not None:
            base += '[{}]'.format(index)
        return base

    def __getitem__(self, item):
        compute, item = item
        indices = None
        field_list = list(self.fields)
        if isinstance(item, str):
            if item not in self.fields:
                raise KeyError(item)
            indices = field_list.index(item)
        if isinstance(item, int):
            indices = item
        if indices is not None:
            if indices >= len(self.fields) or indices < 0:
                raise IndexError(indices)
            return [self.format_variable(compute, index=indices)]
        else:
            # It's something else pass it to numpy
            wrapper = array(self.fields)
            field_names = wrapper[item]
            indices = [field_list.index(fname) for fname in field_names]
            return list(self.format_variable(compute, index=i) for i in indices)

class PotentialEnergyPerAtom(CommandStyle):

    Style = 'pe/atom'

class Reduce(CommandStyle):

    Style = 'reduce'

    Args = [ ChoiceConstraint('mode', ['sum', 'min', 'max', 'ave', 'sumsq', 'avesq']),
        IterableConstraint('fields', element_validator=make_compute_for_atom_validator(Attributes),
                        help=make_compute_for_atom_help)]

    def __init__(self, mode, fields, *args):
        # Preprocess fields
        # If a compute is in the fields take it's variable name instead of the object
        fields = [f.format_variable() if isinstance(f, (Compute, Fix)) else f for f in fields]
        super(Reduce, self).__init__(*(*args, mode, fields))

class ReduceRegion(Reduce):

    Style = 'reduce/region'
    Args = [ReferenceConstraint('region_id', Region, help='ID of region to use for choosing atoms')] + Reduce.Args

    def __init__(self, region_id, fields):
        fields = [f.format_variable() if isinstance(f, (Compute, Fix)) else f for f in fields]
        super(ReduceRegion).__init__(*(fields, region_id))
