from abc import ABCMeta
from qmmm.templates.misc import ConvergenceTest
from qmmm.core.calculation import VASPCalculation
from qmmm.core.utils import is_primitive, is_iterable
from pymatgen.io.vasp import Incar, Kpoints

TEMPLATE_INCAR = Incar({
    'NSW': 0, # No iconic steps
    'IBRION': -1,
    'LREAL': 'Auto',
    'LWAVE': False,
    'LCHARG': False
})


class VASPConvergenceTest(ConvergenceTest, metaclass=ABCMeta):

    DEFAULT_INCAR = TEMPLATE_INCAR

    def __init__(self, name, structure, cutoffs, kpoints, incar=DEFAULT_INCAR, delete=False, prefix=False):
        super(VASPConvergenceTest, self).__init__(name, structure, cutoffs, kpoints, delete=delete, prefix=prefix)
        self._incar = incar

    @property
    def incar(self):
        return self._incar

    @incar.setter
    def incar(self, value):
        self._incar = Incar(value)

    def get_calculation_type(self, **kwargs):
        return VASPCalculation

    def get_calculation_config(self, config, cutoff, kpoints):
        local_config = config.copy()
        local_incar = self._incar.copy()
        local_incar['ENCUT'] = cutoff
        if is_primitive(kpoints):
            local_kpoints = Kpoints.automatic(int(kpoints))
        elif is_iterable(kpoints):
            local_kpoints = Kpoints.automatic_density(self.structure, kpoints, force_gamma=True)
        local_config['incar'] = local_incar
        local_config['kpoints'] = local_kpoints
        return local_config


