from qmmm.core.actions.actions import CheckStep, ActionState, CheckForce, GradientDescent, ApplyPositions, Action
from qmmm.core.actions.templates import LAMMPSRelaxation, LAMMPSStatic, VASPStatic
from qmmm.core.actions.workflow import Workflow
from qmmm.core.actions.utils import Pointer as pointer, CombinedInput
from numpy import inf, ones


class Partitioning(Action):

    def __init__(self, name):
        super(Partitioning, self).__init__(name)
        self.input.initial.copy = False

    def apply(self, structure, qm_region, copy=None, **kwargs):
        if copy:
            structure = structure.copy()
        return {
            'structure': qm_region(structure)
        }


class QMMMCalculation(Workflow):

    def __init__(self, name):
        self._name = name
        super(QMMMCalculation, self).__init__(name)


    def define_workflow(self):
        self.mm_relaxation_total = LAMMPSRelaxation('mm_relaxation_initial')
        self.check_steps = CheckStep('check_steps')
        self.check_forces = CheckForce('check_forces')
        self.mm_relaxation_two = LAMMPSRelaxation('mm_relaxation_two')
        self.partitioning = Partitioning('partitioning')
        self.mm_static = LAMMPSStatic('mm_static')
        self.qm_static = LAMMPSStatic('qm_static')
        self.qm_gradient_descent = GradientDescent('qm_gradient_descent')
        self.qm_apply_positions = ApplyPositions('qm_apply_positions')


        self.edge(self.mm_relaxation_total, self.check_steps)
        self.edge(self.check_steps, self.check_forces, vertex_state=ActionState.No)
        self.edge(self.check_forces, self.mm_relaxation_two, vertex_state=ActionState.No)
        self.edge(self.mm_relaxation_two, self.partitioning)
        self.edge(self.partitioning, self.mm_static)
        self.edge(self.mm_static, self.qm_static)
        self.edge(self.qm_static, self.qm_gradient_descent)
        self.edge(self.qm_gradient_descent, self.qm_apply_positions)
        self.edge(self.qm_apply_positions, self.check_steps)


    def define_dataflow(self):
        # Define setup
        self.mm_relaxation_total.input.potential = pointer(self.input).potential
        self.mm_relaxation_total.input.args = pointer(self.input).lammps
        self.mm_relaxation_total.setup.prefix = self._name

        self.mm_relaxation_two.input.potential = pointer(self.input).potential
        self.mm_relaxation_two.input.args = pointer(self.input).lammps
        self.mm_relaxation_two.setup.prefix = self._name

        self.mm_static.setup.potential = pointer(self.input).potential
        self.mm_static.input.args = pointer(self.input).lammps
        self.mm_static.setup.prefix = self._name+'-mm'

        self.qm_static.input.args = pointer(self.input).vasp
        self.qm_static.input.incar = pointer(self.input).incar
        self.qm_static.input.kpoints = pointer(self.input).kpoints
        self.qm_static.setup.prefix = self._name+'-qm'

        self.mm_relaxation_total.input.structure = pointer(self.input).structure

        self.mm_relaxation_two.input.initial.structure = pointer(self.mm_relaxation_total).output.structure[-1]
        self.mm_relaxation_two.input.structure = pointer(self.qm_apply_positions).output.structure[-1]
        self.mm_relaxation_two.input.step = pointer(self.check_steps).output.step[-1]
        self.mm_relaxation_two.input.fix_group = ['core']
        self.mm_relaxation_two.input.output_keys = ['structure', 'forces']

        self.partitioning.input.copy = True
        self.partitioning.input.qm_region = pointer(self.input).qm_region
        self.partitioning.input.structure = pointer(self.mm_relaxation_two).output.structure[-1]
        #self.partitioning.input.structure = self.mm_relaxation_two.output.structure[-1]

        self.mm_static.input.structure = pointer(self.partitioning.output).structure[-1]
        self.mm_static.input.step = pointer(self.check_steps).output.step[-1]
        self.mm_static.input.output_keys = ['structure']

        self.qm_static.input.structure = pointer(self.partitioning.output).structure[-1]
        self.qm_static.input.step = pointer(self.check_steps).output.step[-1]
        self.qm_static.input.output_keys = ['structure', 'forces']
        self.qm_static.input.groups = ['core']

        self.qm_gradient_descent.input.forces = pointer(self.qm_static).output.forces[-1]
        self.qm_gradient_descent.input.initial.positions = pointer(self.qm_static.output).structure[-1].cart_coords
        self.qm_gradient_descent.input.positions = pointer(self.qm_gradient_descent).output.positions[-1]
        self.qm_gradient_descent.input.gamma0 = 0.15
        self.qm_gradient_descent.input.use_adagrad = False

        self.qm_apply_positions.input.displacements = True
        self.qm_apply_positions.input.copy = True
        self.qm_apply_positions.input.structure = pointer(self.mm_relaxation_two).output.structure[-1]
        self.qm_apply_positions.input.positions = pointer(self.qm_gradient_descent).output.displacements[-1]
        self.qm_apply_positions.input.groups = ['core']

        self.check_forces.input.initial.forces = inf * ones((3,3))
        self.check_forces.input.forces = pointer(self.qm_static).output.forces[-1]

        self.check_steps.input.max_step = 10

        self.active_vertex = self.mm_relaxation_total

    def make_qm_structure(self):
        pass





