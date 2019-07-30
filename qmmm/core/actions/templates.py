from abc import ABCMeta
from qmmm.core.lammps import *
from qmmm.core.actions.actions import QMCalculation, MMCalculation
from qmmm.core.calculation import VASPCalculation, LAMMPSCalculation
from qmmm.core.utils import ensure_iterable
from pymatgen.io.vasp import Incar, Kpoints


class GenericVASPCalculation(QMCalculation, metaclass=ABCMeta):

    def __init__(self, name):
        super(GenericVASPCalculation, self).__init__(name)

    def _get_run_arguments(self, incar=None, kpoints=None, potcar=None, **kwargs):
        run_args = self.input.args.copy()
        if incar is not None:
            run_args['incar'] = incar
        if kpoints is not None:
            run_args['kpoints'] = kpoints
        if potcar is not None:
            run_args['potcar'] = potcar
        return run_args


class VASPStatic(GenericVASPCalculation):

    def __init__(self, name):
        super(VASPStatic, self).__init__(name)
        self.setup.initial.calculation = VASPCalculation
        self.input.initial.kpoints = Kpoints.gamma_automatic((1, 1, 1))
        self.input.initial.incar = Incar({
                'ALGO': 'Fast',
                'LREAL': 'Auto',
                'LCHARG': 'False',
                'IBRION': -1
            })

    def get_calculation_suffix(self):
        return 'static'


class LAMMPSRelaxation(MMCalculation):

    def __init__(self, name):
        super(LAMMPSRelaxation, self).__init__(name)
        self.setup.initial.calculation = LAMMPSCalculation
        self.input.initial.etol = 1e-8
        self.input.initial.ftol = 1e-8
        self.input.initial.maxiter = 1000000
        self.input.initial.maxeval = 1000000
        self.input.initial.vmax = 0.01
        self.input.initial.fix_group = None
        self.input.initial.relax_box = True

    def make_commands(self, etol=None, ftol=None, maxiter=None, maxeval=None, vmax=None, fix_group=None, relax_box=None, **kwargs):
        command_list = [
            ResetTimeStep(0),
            MinStyle(MinStyle.Cg),
            Minimize(etol, ftol, maxiter, maxeval)
        ]
        if relax_box is not None:
            command_list.insert(1, Fix('relaxation', 'all', Fix.BoxRelax, aniso=0.0, vmax=vmax))
        if fix_group is not None:
            for group in ensure_iterable(fix_group):
                fix_name = 'fix_{}'.format(group)
                command_list.insert(0, Fix(fix_name, group, Fix.SetForce, 0.0, 0.0, 0.0))
                command_list.append((Unfix(fix_name)))
            command_list.append(Run(0))
        return command_list

    def get_calculation_suffix(self):
        return 'relaxation'


class LAMMPSStatic(MMCalculation):

    def __init__(self, name):
        super(LAMMPSStatic, self).__init__(name)
        self.setup.initial.calculation = LAMMPSCalculation

    def make_commands(self, **kwargs):
        return [
                ResetTimeStep(0),
                Run(0)
            ]

    def get_calculation_suffix(self):
        return 'static'