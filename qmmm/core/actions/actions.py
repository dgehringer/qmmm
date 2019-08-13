from qmmm.core.lammps import *
from qmmm.core.utils import is_iterable, ensure_iterable, run_once, is_primitive
from qmmm.core.actions.utils import LoggerMixin, IODictionary, Crumb, CombinedInput, InputDictionary
from qmmm.core.calculation import LAMMPSCalculation, Status, VASPCalculation
from numpy import sum as asum, array, sqrt as asqrt, newaxis, inf as ainf, zeros, amax, abs as aabs
from numpy.linalg import norm
from enum import Enum
from abc import ABCMeta, abstractmethod
from pymatgen import Structure
from logging import ERROR, INFO, DEBUG

class ActionState(Enum):
    Default = 0
    Yes = 1
    No = 2


class Action(LoggerMixin, metaclass=ABCMeta):

    def __init__(self, name):
        self.input = InputDictionary()
        self.output = IODictionary()
        self._log_level = ERROR
        self._state = ActionState.Default
        self._valid_states = (ActionState.Default,)
        self._name = name
        self._history = 1


    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def state(self):
        return self._state

    @property
    def valid_states(self):
        return self._valid_states

    @property
    def name(self):
        return self._name

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        self._log_level = value

    def log(self, message, level=INFO, exc_info=None):
        self.logger.log(level, message, exc_info=exc_info)
        if self.log_level <= level:
            print('{}[{}]:{}'.format(self.fullname(),self._name, message))


class CheckStep(Action):

    def __init__(self, name):
        super(CheckStep, self).__init__(name)
        self._step = 0
        self._valid_states = (ActionState.Yes, ActionState.No)
        self.input.default.max_step = 100

    def apply(self, *args, max_step=None):
        self._step += 1
        self.log('{}/{}'.format(self._step, max_step), level=DEBUG)
        if self._step < max_step:
            self._state = ActionState.No
        else:
            self._state = ActionState.Yes

        return {
            'step': self._step
        }


class CheckForce(Action):
    """
    Checks if the largest per-atom force across all atoms and jobs job is below a limit.

    Attributes:
        job (AtomisticGenericJob/list): The job (or list of jobs) in which to check forces.
        f_tol (float): The force tolerance below which to set `vertex_state` to `True`. (Default is infinity.)
        max_force (float): The largest force encountered in the last check.
    """

    def __init__(self, name):
        super(CheckForce, self).__init__(name)
        self._max_force = ainf
        self._valid_states = (ActionState.Yes, ActionState.No)
        self._max_action = None
        self.input.default.ftol = 1.0
        self.input.default.fdiff = False
        self._state = ActionState.No

    def apply(self, forces, ftol, fdiff=None,**kwargs):
        # Find the single biggest force
        old_max_force = self._max_force
        if old_max_force == ainf:
            # Avoid inf-inf == nan
            import sys
            old_max_force = sys.float_info.max
        self._max_force = amax(norm(forces, axis=1))
        self.log('\tMax force{}:{}'.format(' diff' if fdiff else '', self._max_force), level=DEBUG )
        if not fdiff:
            if self._max_force > ftol:
                self._state = ActionState.No
            else:
                self._state = ActionState.Yes
                self.log('YAAAA we reached force convergence: {} > {}'.format(ftol, self._max_force), level=DEBUG)
        else:
            if aabs(self._max_force - old_max_force) > ftol:
                self._state = ActionState.No
            else:
                self._state = ActionState.Yes
                self.log('YAAAA we reached force convergence (diff): {} > {}'.format(ftol, aabs(self._max_force - old_max_force)), level=DEBUG)


class CalculationAction(Action, metaclass=ABCMeta):

    def __init__(self, name):
        super(CalculationAction, self).__init__(name)
        self.setup = InputDictionary()
        self._calculation = None
        self._step = None
        self._structure = None
        self._calculation_history = []
        self._output_key_mapping = {
            'forces': lambda: array(self._structure.site_properties['forces']),
            'velocities': lambda: self._structure.site_properties['velocities'],
            'positions': lambda: self._structure.cart_coords,
            'scaled': lambda: self._structure.frac_coords,
            'id': lambda: array(self._structure.site_properties['id']),
            'masses': lambda: array([site.specie.atomic_mass for site in self._structure]),
            'structure': lambda: self._structure,
            'potential_energy': lambda: self._calculation.potential_energy
        }

    @abstractmethod
    def get_calculation_suffix(self):
        raise NotImplementedError

    @abstractmethod
    def _get_run_arguments(self, **kwargs):
        raise NotImplementedError

    def _make_calculation(self, structure, calculation=None, prefix=None, step=None, working_directory=None, **kwargs):
        if prefix is not None:
            calculation_name = prefix
        else:
            calculation_name = ''
        if prefix is not None:
            calculation_name += '-{}'.format(self.get_calculation_suffix())
        else:
            calculation_name += self.get_calculation_suffix()
        if step is not None:
            calculation_name += '-{}'.format(step)

        input_parameters = {
            'structure': structure,
            'name': calculation_name,
            'working_directory': working_directory
        }

        return calculation(**input_parameters)

    def apply(self, structure, output_keys=None, groups=None, **kwargs):
        calc = self._make_calculation(structure, **self.setup, **kwargs)
        try:
            calc = run_once(calc, **self._get_run_arguments(**kwargs))
        except Exception as e:
            raise e
        if calc.status != Status.Ready:
            raise RuntimeError('Failed to execute calculation: {}'.format(calc.status))
        self._calculation = calc
        self._calculation_history.append(self._calculation.id)
        if groups is not None:
            sites = [site for site in calc.final_structure.sites
             if any([g in ensure_iterable(groups)
                     for g in ensure_iterable(site.properties['group'])
                     ])
             ]
            self._structure = Structure.from_sites(sites)
        else:
            self._structure = calc.final_structure

        output_keys = self._output_key_mapping.keys() if output_keys is None else output_keys
        return {output_key: self._output_key_mapping[output_key]() for output_key in output_keys}


class MMCalculation(CalculationAction, metaclass=ABCMeta):

    def __init__(self, name):
        super(MMCalculation, self).__init__(name)

    def _get_run_arguments(self, **kwargs):
        # Make a copy because we want to modify it
        args = self.input.args.copy()
        args['run'] = self.make_commands(**kwargs)
        args['potential'] = self.input.potential
        return args

    @abstractmethod
    def make_commands(self, **kwargs):
        raise NotImplementedError


class QMCalculation(CalculationAction, metaclass=ABCMeta):

    def __init__(self, name):
        super(QMCalculation, self).__init__(name)

    def _get_run_arguments(self, **kwargs):
        # Override run arguments if there are any
        return self.input.args


class GradientDescent(Action):
    """
    Simple gradient descent update for positions in `flex_output` and structure.

    Input attributes:
        gamma0 (float): Initial step size as a multiple of the force. (Default is 0.1.)
        fix_com (bool): Whether the center of mass motion should be subtracted off of the position update. (Default is
            True)
        use_adagrad (bool): Whether to have the step size decay according to adagrad. (Default is True)

    TODO:
        Fix adagrad bug when it's operating on a list -- each job needs its own accumumated force (or is parallel fnc?)
    """

    def __init__(self, name):
        super(GradientDescent, self).__init__(name)
        self.input.default.gamma0 = 0.1
        self.input.default.fix_com = False
        self.input.default.use_adagrad = True
        self._accumulated_force = 0

    def apply(self, positions, forces, gamma0, use_adagrad, fix_com, mask=None, masses=None):
        if mask is not None:
            masked = True
            mask = array(mask)
            # Preserve data
            unmasked_positions = positions.copy()

            # Mask input data
            positions = positions[mask]
            forces = forces[mask]
            if fix_com:
                masses = masses[mask]
        else:
            masked = False
        if masses is not None:
            masses = array(masses)

        if use_adagrad:
            self._accumulated_force += asqrt(asum(forces * forces))
            gamma0 /= self._accumulated_force

        pos_change = gamma0 * forces

        if fix_com:
            masses = masses[:, newaxis]
            total_mass = asum(masses)
            com_change = asum(pos_change * masses, axis=0) / total_mass
            pos_change -= com_change
        # TODO: fix angular momentum

        new_pos = positions + pos_change
        displacements = pos_change
        if masked:
            unmasked_positions[mask] = new_pos
            new_pos = unmasked_positions
            disp = zeros(unmasked_positions.shape)
            disp[mask] = displacements
        else:
            disp = displacements

        return {
            'positions': new_pos,
            'displacements': disp
        }



class ApplyPositions(Action):

    def __init__(self, name):
        super(ApplyPositions, self).__init__(name)
        self.input.default.copy = False
        self.input.default.groups = None
        self.input.default.displacements = False

    def apply(self, positions, structure, copy=None, groups=None, displacements=None, **kwargs):
        if isinstance(structure, Structure):
            return {
                'structure' : self._apply(positions, structure, copy=copy, groups=groups, displacements=displacements, **kwargs)
            }
        elif is_iterable(structure):
            return {
                'structure': [self._apply(positions, s, copy=copy, groups=groups, displacements=displacements, **kwargs) for s in structure]
            }
        else:
            raise TypeError

    def _apply(self, positions, structure, copy=None, groups=None, displacements=None, **kwargs):
        assert isinstance(structure, Structure)
        if copy:
            structure = structure.copy()
        old_structure = structure.copy()

        # Find sites which match the
        if groups is not None:
            sites = [ site for site in structure.sites
                      if any([g in ensure_iterable(groups)
                              for g in ensure_iterable(site.properties['group'])
                              ])
                      ]
        else:
            sites = structure.sites
        if len(sites) != len(positions):
            raise ValueError('List of sites "{}" of positions do not match "{}"'.format(len(sites), len(positions)))

        if displacements:
            pass
            #print('\tMax displacements:',amax(norm(positions, axis=1)))
        for site, position in zip(sites, positions):
            if displacements:
                site.coords += position
            else:
                site.coords = position
        #print('DISPLACEMENTS', self.name, structure.formula, norm(structure.cart_coords-old_structure.cart_coords, axis=1))
        return structure
