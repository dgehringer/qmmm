from qmmm.templates.abstract import AbstractCalculationSet
from qmmm.core.utils import is_iterable, is_primitive
from itertools import product
from abc import ABCMeta


class FindLatticeParameters(AbstractCalculationSet, metaclass=ABCMeta):
    """
    A class representing a set of calculation to find optimal cell parameters.

    """

    def __init__(self, name, structure, steps, delete=False, prefix=None):
        super(FindLatticeParameters, self).__init__(name, delete=delete, prefix=prefix)
        self._structure = structure
        self._steps = steps
        self._parameter_list = None

    @property
    def structure(self):
        return self._structure

    def get_parameter_configurations(self, **kwargs):
        self._parameter_list = list(self._steps.keys())
        steps = [self._steps[p] for p in self._parameter_list]
        for lattice_params in product(*steps):
            yield dict(zip(self._parameter_list, lattice_params))


class ConvergenceTest(AbstractCalculationSet, metaclass=ABCMeta):
    """
    A class representing a set of calculations to find optimal plane wave cutoff and k-mesh sampling
    """

    def __init__(self, name, structure, cutoffs, kpoints, delete=False, prefix=False):
        super(ConvergenceTest, self).__init__(name, delete=delete, prefix=prefix)
        self._structure = structure
        self._parameter_list = ['cutoff', 'kpoints']
        self._cutoffs = cutoffs
        self._kpoints = kpoints

    @property
    def cutoffs(self):
        return self._cutoffs

    @property
    def kpoints(self):
        return self._kpoints

    @property
    def structure(self):
        return self._structure

    def get_calculation_name(self, cutoff, kpoints, **kwargs):
        """
        Provides a name for the calculation based on the parameter values
        :param cutoff: (int) cutoff energy in eV
        :param kpoints: (int, tuple(int)) integer for automatic k-mesh or manual mesh
        :param kwargs: Additionally unused parameters
        :return: (str) the calculation name
        """
        if is_primitive(kpoints):
            kpt_str = '{}'.format(int(kpoints))
        elif is_iterable(kpoints):
            if not len(kpoints) == 3:
                raise ValueError('Kpoints must be of length 3')
            kpt_str = 'x'.join([str(int(k)) for k in kpoints])
        else:
            raise ValueError('Cannot handle kpoint argument: "{}"'.format(kpoints))
        return '{}_cutoff_{}_kpoints_{}'.format(self.name, int(cutoff), kpt_str)

    def get_structure(self, **kwargs):
        """
        Returns the structure (it is always the same structure)
        :param kwargs: the configuration
        :return: (pymatgen.Structure): the structure
        """
        return self._structure

    def get_parameter_configurations(self, **kwargs):
        """
        Yields all combinations of `cutoffs` and `kpoints` as a dictionary:
        Output format is:
        ```
            {
                'cutoff': 400,
                'kpoints': (3,3,3)
            }
        ```

        :param kwargs: passed from run(**kwargs)
        :return: (dict) All combinations of `cutoffs` and `kpoints`
        """
        for conf in product(self._cutoffs, self._kpoints):
            yield dict(zip(self._parameter_list, conf))