from .command import Command, KeywordCommandStyle, process_args
from qmmm.core.lammps.constraints import PrimitiveConstraint, ReferenceConstraint, ChoiceConstraint, IterableConstraint
from .group import Group
from ..utils import is_iterable


class Fix(Command):
    Command = 'fix'
    VariablePrefix = 'f'
    Args = [PrimitiveConstraint('fix_id', str, help='the dump-ID'),
            ReferenceConstraint('group', Group, help='the ID of a group')]

    @property
    def identifier(self):
        return self._fix_id

    def __init__(self, *args, **kwargs):
        super(Fix, self).__init__(*args, **kwargs)
        # Initialization went well add the compute to the group
        for group in Group.resolve(self.group):
            group.fixes[self.identifier] = self

    def format_variable(self):
        return '{}_{}'.format(self.VariablePrefix, self.identifier)


class BoxRelax(KeywordCommandStyle):
    Style = 'box/relax'

    Keywords = {
        'iso': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'aniso': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'tri': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'x': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'y': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'z': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'xy': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'yz': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'xz': PrimitiveConstraint('ptarget', (float, int), help='desired pressure (pressure units)'),
        'couple': ChoiceConstraint('value', ['xyz', 'xy', 'yz', 'xz'], help='coupling'),
        'nreset': PrimitiveConstraint('value', int, help='reset reference cell every this many minimizer iterations'),
        'vmax': PrimitiveConstraint('fraction', float, help='max allowed volume change in one iteration'),
        'dilate': ChoiceConstraint('value', ['all', 'partial'], help='coupling'),
        'fixedpoint': [
            PrimitiveConstraint('x', (float, int),
                                help='perform relaxation dilation/contraction around this point (distance units)'),
            PrimitiveConstraint('y', (float, int),
                                help='perform relaxation dilation/contraction around this point (distance units)'),
            PrimitiveConstraint('z', (float, int),
                                help='perform relaxation dilation/contraction around this point (distance units)')
        ]
    }


class Npt(KeywordCommandStyle):
    Style = 'npt'

    Keywords = {
        'temp': [
            PrimitiveConstraint('Tstart', (float, int), help='external temperature at start of run'),
            PrimitiveConstraint('Tstop', (float, int), help='external temperature at end of run'),
            PrimitiveConstraint('Tdamp', (float, int), help='temperature damping parameter (time units)')
        ],
        'iso': [
            PrimitiveConstraint('Pstart', (float, int),
                                help='scalar external pressure at start of run (pressure units)'),
            PrimitiveConstraint('Pstop', (float, int), help='scalar external pressure at end of run (pressure units)'),
            PrimitiveConstraint('Pdamp', (float, int), help='pressure damping parameter (time units)')
        ],
        'mtk': ChoiceConstraint('value', ['yes', 'no'], help='add in MTK adjustment term or not'),
        'drag': PrimitiveConstraint('Df', (float, int),
                                    help='drag factor added to barostat/thermostat (0.0 = no drag)'),
        'tloop': PrimitiveConstraint('M', (int,), help='number of sub-cycles to perform on thermostat'),
        'ploop': PrimitiveConstraint('M', (int,), help='number of sub-cycles to perform on barostat thermostat'),
        'nreset': PrimitiveConstraint('value', (int,), help='reset reference cell every this many timesteps'),
        'scalexy': ChoiceConstraint('value', ['yes', 'no'], help='scale xy with ly'),
        'scaleyz': ChoiceConstraint('value', ['yes', 'no'], help='scale yz with lz'),
        'scalexz': ChoiceConstraint('value', ['yes', 'no'], help='scale xz with lz'),
        'flip': ChoiceConstraint('value', ['yes', 'no'],
                                 help='allow or disallow box flips when it becomes highly skewed'),
        'fixedpoint': [
            PrimitiveConstraint('x', (float, int),
                                help='perform barostat dilation/contraction around this point (distance units)'),
            PrimitiveConstraint('y', (float, int),
                                help='perform barostat dilation/contraction around this point (distance units)'),
            PrimitiveConstraint('z', (float, int),
                                help='perform barostat dilation/contraction around this point (distance units)'),
        ],
        'dilate': ReferenceConstraint('gid', Group),
        'tchain': PrimitiveConstraint('N', (int,), help='length of thermostat chain (1 = single thermostat)'),
        'pchain': PrimitiveConstraint('N', (int,), help='length of thermostat chain on barostat (0 = no thermostat)'),
        'x': [
            PrimitiveConstraint('pstart', (float, int), help='external stress tensor component at start of run (pressure units)'),
            PrimitiveConstraint('pend', (float, int), help='external stress tensor component at end of run (pressure units)'),
            PrimitiveConstraint('pdamp', (float, int), help='stress damping parameter (time units)')
        ],
        'y': [
            PrimitiveConstraint('pstart', (float, int),
                                help='external stress tensor component at start of run (pressure units)'),
            PrimitiveConstraint('pend', (float, int),
                                help='external stress tensor component at end of run (pressure units)'),
            PrimitiveConstraint('pdamp', (float, int), help='stress damping parameter (time units)')
        ],
        'z': [
            PrimitiveConstraint('pstart', (float, int),
                                help='external stress tensor component at start of run (pressure units)'),
            PrimitiveConstraint('pend', (float, int),
                                help='external stress tensor component at end of run (pressure units)'),
            PrimitiveConstraint('pdamp', (float, int), help='stress damping parameter (time units)')
        ],
        'xy': [
            PrimitiveConstraint('pstart', (float, int),
                                help='external stress tensor component at start of run (pressure units)'),
            PrimitiveConstraint('pend', (float, int),
                                help='external stress tensor component at end of run (pressure units)'),
            PrimitiveConstraint('pdamp', (float, int), help='stress damping parameter (time units)')
        ],
        'yz': [
            PrimitiveConstraint('pstart', (float, int),
                                help='external stress tensor component at start of run (pressure units)'),
            PrimitiveConstraint('pend', (float, int),
                                help='external stress tensor component at end of run (pressure units)'),
            PrimitiveConstraint('pdamp', (float, int), help='stress damping parameter (time units)')
        ],
        'xz': [
            PrimitiveConstraint('pstart', (float, int),
                                help='external stress tensor component at start of run (pressure units)'),
            PrimitiveConstraint('pend', (float, int),
                                help='external stress tensor component at end of run (pressure units)'),
            PrimitiveConstraint('pdamp', (float, int), help='stress damping parameter (time units)')
        ]
    }


class Deform(KeywordCommandStyle):
    Style = 'deform'

    Final = 'final'
    Delta = 'delta'
    Scale = 'scale'
    Vel = 'vel'
    Erate = 'erate'
    Trate = 'trate'
    Volume = 'volume'
    Wiggle = 'wiggle'

    Args = [PrimitiveConstraint('N', (int,), help='perform box deformation every this many timesteps')]

    Keywords = {
        'remap': ChoiceConstraint('value', ['x', 'v', None], help='remap coords or velocities'),
        'flip': ChoiceConstraint('value', ['yes', 'no'],
                                 help='allow or disallow box flips when it becomes highly skewed'),
        'units': ChoiceConstraint('value', ['lattice', 'box'],
                                  help='allow or disallow box flips when it becomes highly skewed')
    }

    __Sub_Style_Mapping = {
        ('x', 'y', 'z'): {
            'final': [PrimitiveConstraint('lo', (float, int), help='box boundaries at end of run (distance units)'),
                      PrimitiveConstraint('hi', (float, int), help='box boundaries at end of run (distance units)')],
            'delta': [PrimitiveConstraint('dlo', (float, int),
                                          help='change in box boundaries at end of run (distance units)'),
                      PrimitiveConstraint('dhi', (float, int),
                                          help='change in box boundaries at end of run (distance units)')],
            'scale': [PrimitiveConstraint('factor', (float, int),
                                          help='multiplicative factor for change in box length at end of run')],
            'vel': [PrimitiveConstraint('v', (float, int),
                                        help='change box length at this velocity (distance/time units), effectively an engineering strain rate')],
            'erate': [PrimitiveConstraint('R', (float, int), help='engineering strain rate (1/time units)')],
            'trate': [PrimitiveConstraint('R', (float, int), help='true strain rate (1/time units)')],
            'volume': [],
            'wiggle': [PrimitiveConstraint('A', (float, int), help='amplitude of oscillation (distance units)'),
                       PrimitiveConstraint('Tp', (float, int), help='period of the oscillation (distance units)')]
        },
        ('xy', 'xz', 'yz'): {
            'final': [PrimitiveConstraint('tilt', (float, int), help='tilt factor at end of run (distance units)')],
            'delta': [PrimitiveConstraint('dtilt', (float, int), help='change in tilt factor at end of run (distance units)')],
            'vel': [PrimitiveConstraint('v', (float, int),
                                        help='change box length at this velocity (distance/time units), effectively an engineering shear strain rate')],
            'erate': [PrimitiveConstraint('R', (float, int), help='engineering strain rate (1/time units)')],
            'trate': [PrimitiveConstraint('R', (float, int), help='true strain rate (1/time units)')],
            'wiggle': [PrimitiveConstraint('A', (float, int), help='amplitude of oscillation (distance units)'),
                       PrimitiveConstraint('Tp', (float, int), help='period of the oscillation (distance units)')]

        }

    }

    def __init__(self, N, *args, **kwargs):
        self._parameters = []
        # Process arguments
        self._process_arguments(*args)
        # Call CommandStyle constructor
        KeywordCommandStyle.__init__(self,*(N,), **kwargs)

    def _process_arguments(self, *args):
        # Make a copy of the arguments
        def is_parameter(arg):
            if isinstance(arg, str):
                arg = arg.lower()
                options = None
                result = False
                for k in Deform.__Sub_Style_Mapping.keys():
                    if arg in k:
                        # Get the options for the parameters
                        options = Deform.__Sub_Style_Mapping[k]
                        result = True
                return result, options
            else:
                return False, None

        args = list(args).copy()

        while len(args) > 0:
            # Expect a parameter
            curr_arg = args.pop(0)
            is_p, options = is_parameter(curr_arg)
            if is_p:
                parameter = curr_arg
                # Not expect and option
                option_arg = args.pop(0)
                if option_arg in options:
                    constraints = options[option_arg]
                    option = option_arg
                else:
                    raise ValueError('Parameter "{}" has the following options [{}]'.format(parameter, ', '.join(list(options.keys()))))
                option_args = []
                for c in constraints:
                    value = args.pop(0)
                    if not c.validate(value):
                        raise ValueError('Cannot accept value "{}" for constraint {}'.format(value, c.name))
                    option_args.append(value)
                option_args = option_args if len(option_args) != 1 else option_args[0]
                self._parameters.append((parameter, option, option_args))
            else:
                raise ValueError('Expected a parameter [{}]'.format(', '.join([str(k) for k in Deform.__Sub_Style_Mapping.keys()])))

    @classmethod
    def from_string(cls, string, preprocess=None, return_remaining=False, make_obj=True):
        # Get parameters out
        #Get parameters first
        crumbs = [c for c in string.split(' ') if c.lstrip().rstrip() != '']
        style = crumbs.pop(0)
        curr_arg = style

        if cls._has_arguments():
            # If there is an iterable argument consume everything
            has_iterable_argument = any([isinstance(argument, IterableConstraint) for argument in cls.Args])
            if has_iterable_argument:
                args = crumbs.copy()
                crumbs = []
            else:
                # Consume just one thing for each argument
                args = [crumbs.pop(0) for _ in cls.Args]
        else:
            args = []

        style_args = process_args(cls, args, preprocess=preprocess) if cls._has_arguments() else ()

        condition = lambda string_: string_ in cls.Keywords if cls._has_keywords() else True
        parameters = []
        while len(crumbs) > 0 or condition(curr_arg):
            # Expect parameter
            curr_arg = crumbs.pop(0)
            option_found = False
            for k in Deform.__Sub_Style_Mapping.keys():
                if curr_arg in k:
                    options = Deform.__Sub_Style_Mapping[k]
                    option_found = True
                    break
            if not option_found:
                # Noting was found , continue
                # Check if it already was an keyword, then break imideately
                if curr_arg in cls.Keywords:
                    # Put it back to the crumbs
                    crumbs.insert(0, curr_arg)
                    break
                else:
                    raise ValueError('Expected a parameter [{}]'.format(
                        ', '.join([str(k) for k in Deform.__Sub_Style_Mapping.keys()])))
            # We are sure that it is an parameter
            parameter = curr_arg
            curr_arg = crumbs.pop(0)
            if curr_arg not in options:
                raise ValueError(
                    'Parameter "{}" has the following options [{}]'.format(parameter, ', '.join(list(options.keys()))))
            option = curr_arg
            constraints = options[curr_arg]
            option_args = [c.parse_string(crumbs.pop(0)) for c in constraints]
            option_args = option_args if len(option_args) != 1 else option_args[0]
            parameters.append((parameter, option, option_args))

        if cls._has_keywords():
            # consume until no keywords are available
            # Now after the style I expect a keyword
            keywords = {}
            while len(crumbs) > 0:
                kword = crumbs.pop(0)
                if kword in cls.Keywords:
                    constraints = cls.Keywords[kword]
                    if not is_iterable(constraints):
                        constraints = [constraints]
                    # Consume all arguments
                    try:
                        kvalues = [c.parse_string(crumbs.pop(0)) for c in constraints]
                    except IndexError:
                        raise
                    keywords[kword] = kvalues if len(constraints) > 1 else kvalues[0]
        else:
            keywords = {}

        if style != cls.Style:
            raise ValueError('Invalid style {} for CommandStyle Fix'.format(style))
        #Make param args
        total_args = list(style_args)
        parameter_args = []
        for p, o, args_ in parameters:
            if not is_iterable(args_):
                a = [args_]
            else:
                a = args_
            parameter_args += [p, o]
            parameter_args += a
        total_args += parameter_args
        if make_obj:
            obj = cls(*total_args, **keywords)
        else:
            obj = (total_args, keywords)
        return (obj, ' '.join(crumbs)) if return_remaining else obj
        # The remaining crumbs contain only keywords

    def format(self):
        # Format parameters
        param_crumbs = []
        for p, o, args_ in self._parameters:
            param_crumbs.append(p)
            param_crumbs.append(o)
            constraint = self.__Sub_Style_Mapping[[k for k in self.__Sub_Style_Mapping.keys() if p in k][0]][o]

            if is_iterable(args_):
                for a, c in zip(args_, constraint):
                    param_crumbs.append(c.format_argument(a))
            else:
                param_crumbs.append(constraint[0].format_argument(args_))

        arg_str = ' ' + self.format_arguments() if self._has_arguments() else ''
        kwd_str = ' ' + self.format_keywords() if self._has_keywords() else ''

        par_str = ' ' + ' '.join(param_crumbs) if len(self._parameters) >  0 else ''

        return self.Style + arg_str + par_str + kwd_str