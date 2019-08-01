from qmmm.core.actions.actions import CheckStep, ActionState, CheckForce, GradientDescent, ApplyPositions
from qmmm.core.actions.templates import LAMMPSRelaxation, LAMMPSStatic, VASPStatic
from qmmm.core.actions.workflow import Workflow
from qmmm.core.utils import flatten, is_iterable
from qmmm.core.actions.utils import Pointer as pointer
from numpy import inf, ones, setdiff1d, array, concatenate, unique, amax, amin, vstack, mean, arange, dot, prod, abs as np_abs, any as np_any, identity, ptp
from numpy.linalg import inv, norm
from qmmm.structure_utils import wrap_positions, build_neighbor_list, indices
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NewPrimitiveNeighborList
from pymatgen import Structure, Lattice


class QMMMCalculation(Workflow):

    def __init__(self, name):
        self._name = name
        super(QMMMCalculation, self).__init__(name)
        self.input.default.qm_region = lambda structure: structure
        self.input.default.gamma0 = 0.1
        self.input.default.use_adagrad = False
        self.input.default.max_steps = 15
        self.input.default.ftol = 0.1
        self.input.default.fdiff = False
        self.finished += self._compute_qmmm_energy

    def define_workflow(self):
        self.mm_relaxation_total = LAMMPSRelaxation('mm_relaxation_initial')
        self.check_steps = CheckStep('check_steps')
        self.check_forces = CheckForce('check_forces')
        self.mm_relaxation_two = LAMMPSRelaxation('mm_relaxation_two')
        self.mm_static = LAMMPSStatic('mm_static')
        self.qm_static = LAMMPSStatic('qm_static')
        self.qm_gradient_descent = GradientDescent('qm_gradient_descent')
        self.qm_apply_positions = ApplyPositions('qm_apply_positions')

        self.edge(self.check_steps, self.check_forces, vertex_state=ActionState.No)
        self.edge(self.check_forces, self.mm_relaxation_two, vertex_state=ActionState.No)
        self.edge(self.mm_relaxation_two, self.qm_static)
        self.edge(self.qm_static, self.qm_gradient_descent)
        self.edge(self.qm_gradient_descent, self.qm_apply_positions)
        self.edge(self.qm_apply_positions, self.check_steps)
        self.edge(self.check_steps, self.mm_static, vertex_state=ActionState.Yes)
        self.edge(self.check_forces, self.mm_static, vertex_state=ActionState.Yes)


    def define_dataflow(self):
        # Define setup
        self.mm_relaxation_total.input.potential = pointer(self.input).potential
        self.mm_relaxation_total.input.args = pointer(self.input).lammps
        self.mm_relaxation_total.setup.prefix = self._name

        self.mm_relaxation_two.input.potential = pointer(self.input).potential
        self.mm_relaxation_two.input.args = pointer(self.input).lammps
        self.mm_relaxation_two.setup.prefix = self._name + '-mm-two'

        self.mm_static.input.potential = pointer(self.input).potential
        self.mm_static.input.args = pointer(self.input).lammps
        self.mm_static.setup.prefix = self._name+'-mm-one'

        self.qm_static.input.args = pointer(self.input).lammps
        self.qm_static.input.potential = pointer(self.input).potential
        #self.qm_static.input.incar = pointer(self.input).incar
        #self.qm_static.input.kpoints = pointer(self.input).kpoints
        self.qm_static.setup.prefix = self._name+'-qm-one'

        self.mm_relaxation_total.input.structure = pointer(self.input).structure
        self.mm_relaxation_total.input.output_key = ['structure']

        self.mm_relaxation_two.input.default.structure = pointer(self).input.mm_relaxed_structure
        self.mm_relaxation_two.input.structure = pointer(self.qm_apply_positions).output.structure[-1][0]
        self.mm_relaxation_two.input.step = pointer(self.check_steps).output.step[-1]
        self.mm_relaxation_two.input.fix_group = ['core', 'seed']
        self.mm_relaxation_two.input.output_keys = ['structure', 'forces', 'potential_energy']
        self.mm_relaxation_two.input.relax_box = False


        #self.partitioning.input.structure = self.mm_relaxation_two.output.structure[-1]

        #self.mm_static.input.structure = pointer(self.qm_static).output.structure[-1]
        self.mm_static.input.structure = pointer(self.qm_apply_positions).output.structure[-1][3]
        self.mm_static.input.step = pointer(self.check_steps).output.step[-1]
        self.mm_static.input.output_keys = ['structure', 'potential_energy']

        self.qm_static.input.structure = pointer(self.qm_apply_positions).output.structure[-1][1]
        self.qm_static.input.default.structure = pointer(self.input).qm_structure
        self.qm_static.input.step = pointer(self.check_steps).output.step[-1]
        self.qm_static.input.output_keys = ['structure', 'forces', 'potential_energy']
        #self.qm_static.input.groups = ['seed', 'core']

        self.qm_gradient_descent.input.forces = pointer(self.qm_static).output.forces[-1]
        self.qm_gradient_descent.input.default.positions = pointer(self.input).qm_structure.cart_coords
        self.qm_gradient_descent.input.positions = pointer(self.qm_gradient_descent).output.positions[-1]
        self.qm_gradient_descent.input.gamma0 = 0.1
        self.qm_gradient_descent.input.use_adagrad = False

        self.qm_apply_positions.input.displacements = True
        self.qm_apply_positions.input.copy = True
        self.qm_apply_positions.input.structure = [pointer(self.mm_relaxation_two).output.structure[-1],
                                                   pointer(self.qm_static).output.structure[-1],
                                                   pointer(self.input).qm_structure,
                                                   pointer(self.input).qm_structure_mm]
        self.qm_apply_positions.input.positions = pointer(self.qm_gradient_descent).output.displacements[-1][pointer(self)._qm_core_only]
        self.qm_apply_positions.input.groups = ['core', 'seed']

        self.check_forces.input.default.forces = inf * ones((3,3))
        self.check_forces.input.forces = pointer(self.qm_static).output.forces[-1][pointer(self)._qm_core_only]
        self.check_forces.input.fdiff = pointer(self.input).fdiff
        self.check_forces.input.ftol = pointer(self.input).ftol

        self.check_steps.input.max_step = 15

        self.active_vertex = self.check_steps

        self.output.energy_mm = pointer(self.mm_relaxation_two).output.potential_energy[-1]
        self.output.energy_qm = pointer(self.qm_static).output.potential_energy[-1]
        self.output.energy_mm_one = pointer(self.mm_static).output.potential_energy[-1]
        self.output.qm_structure = pointer(self.qm_apply_positions).output.structure[-1][2]


    def _qm_core_only(self):
        return indices(self.qm_static.output.structure[-1], group=['core', 'seed'])


    def run(self):
        self._execute_vertex(self.mm_relaxation_total, [])
        self.input.mm_relaxed_structure = self.mm_relaxation_total.output.structure[-1]
        self.make_qm_structure()
        # Remove sites change species or whatever
        self.input.qm_structure_mm = self.input.qm_structure.copy()
        self.input.qm_structure = self.input.qm_region(self.input.qm_structure)
        super(QMMMCalculation, self).run()

    def _compute_qmmm_energy(self):
        self.output.energy_qmmm = self.output.energy_mm + self.output.energy_qm - self.output.energy_mm_one

    def make_qm_structure(self):
        # Take already the relaxed structure
        superstructure = AseAtomsAdaptor.get_atoms(self.input.mm_relaxed_structure)
        id_list = array(self.input.mm_relaxed_structure.site_properties['id'])

        if 'domain_ids' not in self.input:
            if self.input.seed_ids is None:
                raise ValueError('Either the domain ids must be provided explicitly, or seed ids must be given.')
            seed_ids = array(self.input.seed_ids, dtype=int)
            shells = self._build_shells(superstructure,
                                        self.input.core_shells + self.input.buffer_shells,
                                        self.input.seed_ids, cutoff=self.input.neighbor_cutoff if 'neighbor_cutoff' in self.input else None, bothways=True, self_interaction=True)
            core_ids = concatenate(shells[:self.input.core_shells])
            buffer_ids = concatenate(shells[self.input.core_shells:])
            region_I_ids = concatenate((seed_ids, core_ids, buffer_ids))

            bb = self._get_bounding_box(superstructure[region_I_ids])
            extra_box = 0.5 * array(self.input.filler_width)
            bb[:, 0] -= extra_box
            bb[:, 1] += extra_box

            # Store it because get bounding box return a tight box and is different

            filler_ids = self._get_ids_within_box(superstructure, bb)
            filler_ids = setdiff1d(filler_ids, region_I_ids)

            bb = self._get_bounding_box(superstructure[concatenate((region_I_ids, filler_ids))])

            self.input.domain_ids = {'seed': seed_ids, 'core': core_ids, 'buffer': buffer_ids, 'filler': filler_ids}
        elif 'seed_ids' not in self.input:
            raise ValueError('Only *one* of `seed_ids` and `domain_ids` may be provided.')
        # Use domains provided
        else:
            seed_ids = self.input.domain_ids['seed']
            core_ids = self.input.domain_ids['core']
            buffer_ids = self.input.domain_ids['buffer']
            filler_ids = self.input.domain_ids['filler']
            region_I_ids = concatenate((seed_ids, core_ids, buffer_ids))
            bb = self._get_bounding_box(superstructure[concatenate((region_I_ids, filler_ids))])
        # Extract the relevant atoms

        # Build the domain ids in the qm structure
        qm_structure = None
        domain_ids_qm = {}
        offset = 0
        clipped_ids = []
        for key, ids in self.input.domain_ids.items():
            if qm_structure is None:
                qm_structure = superstructure[ids]
            else:
                qm_structure += superstructure[ids]
            clipped_ids.extend(id_list[ids].tolist())
            id_length = len(ids)
            domain_ids_qm[key] = arange(id_length) + offset
            offset += id_length

        self.input.domain_ids_qm = domain_ids_qm
        # And put everything in a box near (0,0,0)
        extra_vacuum = 0.5 * self.input.vacuum_width
        bb[:, 0] -= extra_vacuum
        bb[:, 1] += extra_vacuum

        # If the bounding box is larger than the MM superstructure
        bs = np_abs(bb[:, 1] - bb[:, 0])
        supercell_lengths = [norm(row) for row in superstructure.cell]
        shrinkage = array([(box_size - cell_size) / 2.0 if box_size > cell_size else 0.0 for box_size, cell_size in
                              zip(bs, supercell_lengths)])
        if np_any(shrinkage > 0):
            self.logger.info('The QM box is larger than the MM Box therefore I\'ll shrink it')
            bb[:, 0] += shrinkage
            bb[:, 1] -= shrinkage
        elif any([0.9 < box_size / cell_size < 1.0 for box_size, cell_size in zip(bs, supercell_lengths)]):
            # Check if the box is just slightly smaller than the superstructure cell
            self.logger.warn(
                'Your cell is nearly as large as your supercell. Probably you want to expand it a little bit')

        qm_structure.cell = identity(3) * ptp(bb, axis=1)

        box_center = tuple(dot(inv(superstructure.cell), mean(bb, axis=1)))
        qm_structure.wrap(box_center)
        # Wrap it to the unit cell
        qm_structure.positions = dot(qm_structure.get_scaled_positions(), qm_structure.cell)

        # Add group property to the super_structure structure
        mm_groups = array([None] * len(self.input.mm_relaxed_structure))
        qm_groups = array([None] * len(qm_structure))

        for group_name, group_ids in self.input.domain_ids.items():
            mm_groups[group_ids] = group_name
        for group_name, group_ids in self.input.domain_ids_qm.items():
            qm_groups[group_ids] = group_name
        self.input.mm_relaxed_structure.add_site_property('group', mm_groups.tolist())

        qm_structure = Structure(Lattice(qm_structure.cell), qm_structure.get_chemical_symbols() ,qm_structure.positions,
                                 coords_are_cartesian=True,
                                 site_properties={'group': qm_groups.tolist(), 'id': clipped_ids})
        #from main import vesta, make_vis
        #vesta([self.input.mm_relaxed_structure, make_vis(self.input.mm_relaxed_structure), qm_structure, make_vis(qm_structure)])
        self.input.qm_structure = qm_structure

    @staticmethod
    def _build_shells(structure, n_shells, seed_ids, cutoff=None, **kwargs):
        indices = [seed_ids]
        current_shell_ids = seed_ids
        if cutoff is not None:
            cutoff = [cutoff] * len(structure) if not is_iterable(cutoff) else cutoff

        nl = build_neighbor_list(structure,
                                 cutoffs=cutoff,
                                 primitive=NewPrimitiveNeighborList, **kwargs)
        for _ in range(n_shells):
            neighbors = flatten([nl.get_neighbors(sid)[0] for sid in current_shell_ids])
            new_ids = unique(neighbors)
            # Make it exclusive
            for shell_ids in indices:
                new_ids = setdiff1d(new_ids, shell_ids)
            indices.append(new_ids)
            current_shell_ids = new_ids
        # Pop seed ids
        indices.pop(0)
        return indices


    @staticmethod
    def _get_bounding_box(structure):
        """
        Finds the smallest rectangular prism which encloses all atoms in the structure after accounting for periodic
        boundary conditions.

        So what's the problem?

        |      ooooo  |, easy, wrap by CoM
        |ooo        oo|, easy, wrap by CoM
        |o    ooo    o|,


        Args:
            structure (Atoms): The structure to bound.

        Returns:
            numpy.ndarray: A 3x2 array of the x-min through z-max values for the bounding rectangular prism.
        """
        wrapped_structure = structure.copy()
        # Take the frist positions and wrap the atoms around there to determine the size of the bounding box
        wrap_center = tuple(dot(inv(structure.cell), structure.positions[0, :]))
        wrapped_positions = wrap_positions(structure, center=wrap_center)

        bounding_box = vstack([
            amin(wrapped_positions, axis=0),
            amax(wrapped_positions, axis=0)
        ]).T
        return bounding_box

    def show_boxes(self):

        self._plot_boxes([self.input.structure.cell, self.input.qm_structure.cell], colors=['r', 'b'],
                         titles=['MM Superstructure', 'QM Structure'])

    @staticmethod
    def _get_ids_within_box(structure, box):
        """
        Finds all the atoms in a structure who have a periodic image inside the bounding box.

        Args:
            structure (Atoms): The structure to search.
            box (np.ndarray): A 3x2 array of the x-min through z-max values for the bounding rectangular prism.

        Returns:
            np.ndarray: The integer ids of the atoms inside the box.
        """
        box_center = mean(box, axis=1)
        box_center_direct = dot(inv(structure.cell), box_center)
        # Wrap atoms so that they are the closest image to the box center
        wrapped_structure = structure.copy()
        wrapped_structure.wrap(tuple(box_center_direct))
        pos = wrapped_structure.positions
        # Keep only atoms inside the box limits
        masks = []
        for d in arange(len(box)):
            masks.append(pos[:, d] > box[d, 0])
            masks.append(pos[:, d] < box[d, 1])
        total_mask = prod(masks, axis=0).astype(bool)
        return arange(len(structure), dtype=int)[total_mask]















