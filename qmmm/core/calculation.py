import numpy as np
import pickle
import os
import tarfile
from abc import ABCMeta, abstractmethod
from qmmm.core.event import Event
from qmmm.core.utils import is_iterable, create_directory, LoggerMixin, recursive_as_dict, process_decoded, \
    StructureWrapper, predicate_generator
from qmmm.core.configuration import get_setting
from qmmm.core.vasp.archive import construct_potcar
from qmmm.core.vasp.vasprun import Vasprun
from os.path import join, exists, isfile
from qmmm.core.lammps import Command, Clear, Units, Boundary, AtomStyle, Dimension, ReadData, \
    CommandStyle, WriteDump, WriteData, ThermoStyle, Thermo, ThermoModify, Group, Dump, PairCoeff
from qmmm.core.lammps.group import All
from qmmm.core.runner import VASPRunner, LAMMPSRunner
from qmmm.core.mongo import get_mongo_client, make_database_config, insert, query
from qmmm.core.lammps.potential import available_potentials, LAMMPSPotential
from enum import Enum
from math import isclose
from pymatgen.io.lammps.inputs import LammpsData
from pymatgen import Structure
from pymatgen.io.vasp import Potcar, Incar, Kpoints, Poscar, Oszicar, Outcar
from tempfile import TemporaryDirectory
from qmmm.core.configuration import Configuration
from monty.json import MSONable
from uuid import uuid4
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log
from pymatgen.core.periodic_table import Element


def element_test(s):
    return isinstance(Element(s), Element)

class Status(Enum):
    NotInitialized = 0
    Initialized = 1
    Prepared = 2
    Execution = 3
    ExecutionSuccessful = 5
    ExecutionFailed = 4
    Processing = 6
    ProcessingSuccessful = 7
    ProcessingFailed = 8
    Ready = 10



class Calculation(MSONable, LoggerMixin, metaclass=ABCMeta):

    def __init__(self, structure, name=None, working_directory=None):
        self._structure = structure.copy()
        self._name = name
        self._working_directory = working_directory
        self.status_changed = Event()
        self._status = Status.NotInitialized
        self._id = str(uuid4())
        self._run_args = None
        self._run_kwargs = None

    def get_path(self, *args):
        return join(self._working_directory, *args)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        self.status_changed.fire(self)

    @property
    def ready(self):
        return self._check_ready()

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def structure(self):
        return self._structure

    @property
    def working_directory(self):
        return self._working_directory

    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def write_input_files(self, *args, **kwargs):
        raise NotImplementedError

    def prepare(self, *args, **kwargs):
        self.initialize(*args, **kwargs)
        self.status = Status.Initialized
        self.write_input_files(*args, **kwargs)
        self.status = Status.Prepared

    @abstractmethod
    def success(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        if self._run_args is None:
            self._run_args = args
        if self._run_kwargs is None:
            self._run_kwargs = kwargs

        args = self._run_args
        kwargs = self._run_kwargs
        # If this method is called just to check the status, status will be Status.Executing
        if not self.status == Status.Execution:
            self.prepare(*args, **kwargs)
            self.status = Status.Prepared
        # Exectute run, submit or check
        result = self.execute(*args, **kwargs)
        if result is None:
            self.status = Status.Execution
            return
        elif not result:
            # Something happened
            self.status = Status.ExecutionFailed
            # Early exit
            return
        else:
            self.status = Status.ExecutionSuccessful
            # Continue
        if self.success(*args, **kwargs):
            # Resume with
            self.status = Status.ExecutionSuccessful
            result = self.process(*args, **kwargs)
            if result:
                self.status = Status.ProcessingSuccessful
                self.status = Status.Ready
            else:
                self.status = Status.ProcessingFailed
        else:
            self.status = Status.ExecutionFailed

    def _check_ready(self, raise_error=False):
        result = self.status == Status.Ready
        if not result:
            if raise_error:
                raise RuntimeError('Calculation is not ready')
        return result

    def as_dict(self):
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        d['id'] = self._id
        d['status'] = self.status.value
        d['structure'] = StructureWrapper.structure_to_dict(self.structure)
        d['name'] = self.name
        d['working_directory'] = self.working_directory
        d['run_args'] = recursive_as_dict(self._run_args)
        d['run_kwargs'] = recursive_as_dict(self._run_kwargs)
        return d

    @staticmethod
    def _set_internal(obj, decoded):
        obj._status = Status(decoded['status'])
        obj._id = decoded['id']
        obj._run_args = decoded['run_args']
        obj._run_kwargs = decoded['run_kwargs']

    @classmethod
    def from_dict(cls, d):
        # Fast deserialize structure
        if 'structure' in d:
            d['structure'] = StructureWrapper.from_dict(d['structure'])
        decoded = {k: process_decoded(v) for k, v in d.items()
                   if not k.startswith("@")}
        obj = cls(decoded['structure'], name=decoded['name'], working_directory=decoded['working_directory'])
        Calculation._set_internal(obj, decoded)
        return obj

    def save(self, prefix=None, db=None):

        if not prefix:
            prefix = os.getcwd()
        tar_file_name = '{}.tar.gz'.format(self._id)
        pickle_file_name = '{}.pickle'.format(self._id)
        tar_file_name, pickle_file_name = join(prefix, tar_file_name), join(prefix, pickle_file_name)
        if self.status.value > Status.Execution.value:
            # There is already some data
            with tarfile.open(tar_file_name, 'w:gz') as archive:
                archive.add(self.working_directory, recursive=True)

        if db:
            db = make_database_config(db)
            _, _, collection = get_mongo_client(**db)
            d = self.as_dict()
            insert(collection, d)
        else:
            with open(pickle_file_name, 'wb') as pickle_file:
                pickle.dump(self.as_dict(), pickle_file)

    @classmethod
    def load(cls, identifier, prefix=None, extract=False, db=None):
        tar_file_name = '{}.tar.gz'.format(identifier)
        pickle_file_name = '{}.pickle'.format(identifier)
        if not prefix:
            prefix = os.getcwd()
        tar_file_name = join(prefix, tar_file_name)
        pickle_file_name = join(prefix, pickle_file_name)
        if db:
            db = make_database_config(db)
            _, _, collection = get_mongo_client(**db)
            d = query(collection, identifier)
            calc = cls.from_dict(d)
        else:
            with open(pickle_file_name, 'rb') as pickle_file:
                calc = cls.from_dict(pickle.load(pickle_file))

        alter_wd = False
        if calc.status.value > Status.Execution.value:
            with tarfile.open(tar_file_name, 'r:gz') as archive:
                # Extract it to this path
                with TemporaryDirectory() as temp_dir:
                    if extract:
                        if isinstance(extract, str):
                            extract_prefix = extract
                            alter_wd = True
                            old_wd = calc.working_directory
                            calc._working_directory = join(extract_prefix, old_wd)
                        elif isinstance(extract, bool):
                            extract_prefix = os.getcwd()
                        else:
                            raise TypeError('extract keyword must be either a prefix or a bool')
                    else:
                        extract_prefix = temp_dir
                        alter_wd = True
                        old_wd = calc.working_directory
                        calc._working_directory = join(extract_prefix, old_wd)

                    archive.extractall(extract_prefix)

                    result = calc.process(*calc._run_args, **calc._run_kwargs)
                    if not result:
                        from logging import getLogger
                        getLogger('{}.{}'.format(cls.__module__, cls.__name__)).warning('Failed to parse outputs in directory: {}'.format(calc.working_directory))
                    if alter_wd:
                        calc._working_directory = old_wd

        return calc


DEFAULT_THERMO_ARGS = ('step', 'pe', 'lx', 'ly', 'lz', 'pxx', 'pyy', 'pzz', 'pxy', 'pyz', 'pxz', 'vol')


class LAMMPSCalculation(Calculation):

    def __init__(self, structure, name, working_directory=None, thermo_args=DEFAULT_THERMO_ARGS):
        if not working_directory:
            working_directory = '{}.lammps'.format(name)
        super(LAMMPSCalculation, self).__init__(structure, name=name, working_directory=working_directory)
        self._sequence = []
        self._thermo_args = thermo_args
        self._thermo_column_mapping = {k: i for i, k in enumerate(self._thermo_args)}
        self._dump_folder = 'dumps'
        self._dump_system_final = '{}.system.dump'.format(name)
        self._dump_system_group_ids = '{name}.{group_id}.system.dump'
        self._structure_folder = 'structures'
        self._structure_file_input = '{}.atoms'.format(name)
        self._structure_file_final = '{}.final.atoms'.format(name)
        self._input_file = '{}.in'.format(name)
        self._log_file = '{}.log'.format(name)
        self._num_species = len(set([site.species_string for site in structure]))
        self._runner = LAMMPSRunner()
        self._runner_success = None
        self._final_forces = None
        self._final_structure = None
        self._thermo_data = None
        self._thermo_style = None
        self._potential = None
        self._groups = {'all': [All]}
        self._groups_ids = {}
        self._group_write_dumps = {}
        self._dumps = {}
        self._runner_bound = False
        self._id_mapping = {}
        self._species_order = None
        # Order structure as it done by LammpsData
        # But therefore it is ensure that our id matches that assigned by LammpsData and the ordering is ensured
        ordered_structure = self._structure.get_sorted_structure()
        self._structure = ordered_structure.copy()

        if 'id' not in self._structure.site_properties:
            self._structure.add_site_property('id', [i + 1 for i, _ in enumerate(self._structure)])
            self._id_mapping = {i + 1: i+1 for i, _ in enumerate(self._structure)}
        else:
            self._id_mapping = {i + 1: v for i, v in enumerate(self._structure.site_properties['id'])}

        self._groups_ids['all'] = self._structure.site_properties['id']
        reverse_id_mapping = {v : k for k, v in self._id_mapping.items()}
        if 'group' in self._structure.site_properties:
            groups = {}
            for site in self._structure:
                site_groups = site.properties['group']
                # Ensure to use local Ids
                site_id = reverse_id_mapping[site.properties['id']]
                if site_groups is not None:
                    # The group is not none
                    if isinstance(site_groups, str):
                        site_groups = [site_groups]
                        # The site is only in one group and it is given explicitly
                    for group_name in site_groups:
                        if group_name not in groups:
                            groups[group_name] = [site_id]
                        else:
                            groups[group_name].append(site_id)

            # Add them to the local groups
            for group_name, ids in groups.items():
                if group_name not in self._groups:
                    self._groups[group_name] = [Group(group_name, Group.Id.from_id_list(ids))]
                else:
                    self._groups[group_name].append(Group(group_name, Group.Id.from_id_list(ids)))
                self._groups_ids[group_name] = ids


    @property
    def available_potentials(self):
        species = [s.species_string for s in self._structure]
        return available_potentials(species)

    @property
    def input_file(self):
        return self._input_file

    @property
    def log_file(self):
        return self._log_file

    def define_initialization(self):
        return [
            Clear(),
            Units(Units.Metal),
            Dimension(3),
            Boundary('p', 'p', 'p'),
            AtomStyle(AtomStyle.Atomic)
        ]

    def define_atoms(self):
        l = self.structure.lattice
        if not all([isclose(angle, 90, abs_tol=1e-9) for angle in (l.alpha, l.beta, l.gamma)]):
            pass
            #raise ValueError('Canno\'t handle simulation boxes which are not orthorhombic')

        return [
            ReadData(join(self._structure_folder, self._structure_file_input), group='all')
        ]

    def define_potential(self, potential=None):
        local_sequence = []
        potentials = [potential] if not is_iterable(potential) else potential
        assert len(potentials) > 0
        result = []
        for pot in potentials:
            if isinstance(pot, str):
                pot_ = LAMMPSPotential.load(pot)
                if pot_ is None:
                    raise ValueError('Potential "{}" does not exist'.format(pot))
                else:
                    result.append(pot_)
            elif isinstance(pot, LAMMPSPotential):
                result.append(pot)

        self._potential = result

        # Now add the commands to the sequence
        for pot in result:
            local_sequence.extend(pot.commands)

        return local_sequence

    def define_run(self, run=None):
        return run or []

    def define_epilog(self):
        result = [
            WriteDump('all',
                      WriteDump.Custom,
                      join(self._dump_folder, self._dump_system_final),
                      ['id', 'type', 'fx', 'fy', 'fz']),
            WriteData(join(self._structure_folder, self._structure_file_final))]
        for group_id in self._groups.keys():
            w = WriteDump(group_id,
                          WriteDump.Custom,
                          join(self._dump_folder,
                               self._dump_system_group_ids.format(name=self.name, group_id=group_id)),
                          ['id'])
            self._group_write_dumps[group_id] = w
            result.insert(0, w)
        return result

    def define_output(self, output=None):
        thermo_style = ThermoStyle(ThermoStyle.Custom, list(self._thermo_args))
        self._thermo_style = thermo_style.style
        return output or [
            Thermo(1),
            thermo_style,
            ThermoModify(format='float %20.15g')
        ]

    def define_settings(self, settings=None):
        """
        Computes and fixes go here
        :return: A list of commands
        """
        return settings or []

    def initialize(self, *args, **kwargs):
        """
        Assembles the command sequence
        :param args:
        :param kwargs:
        :return:
        """
        # Assemble command seqence
        if 'potential' not in kwargs:
            raise ValueError('potential argument is missing')
        self._sequence.extend(self.define_initialization())
        self._sequence.extend(self.define_atoms())
        # Define already defined groups, should be done in the setting
        for group_name, commands in self._groups.items():
            # Skip it if it is the 'all' group
            if group_name == 'all':
                continue
            self._sequence.extend(commands)

        self._potential = kwargs['potential']
        self._sequence.extend(self.define_potential(self._potential))
        if 'run' in kwargs:
            run = kwargs['run']
        else:
            run = None
        if 'settings' in kwargs:
            settings = kwargs['settings']
        else:
            settings = None
        if 'output' in kwargs:
            output = kwargs['output']
        else:
            output = None
        # Check if some groups were defined here
        # Get structure and system defined groups
        self._sequence.extend(self.define_settings(settings=settings))
        self._sequence.extend(self.define_output(output=output))
        self._sequence.extend(self.define_run(run=run))
        for command in self._sequence:
            if isinstance(command, Group):
                # This groups command was newly define
                if command.identifier not in self._groups:
                    self._groups[command.identifier] = [command]
                else:
                    # Check if that command is already there
                    existing_group_commands = self._groups[command.identifier]
                    command_found = False
                    for existing_group_command in existing_group_commands:
                        if existing_group_command == command:
                            command_found = True
                            break
                    if not command_found:
                        # This command was not found although the group is already registered
                        self._groups[command.identifier].append(command)
        self._sequence.extend(self.define_epilog())
        # self._sequence.append(Print('__end_of_auto_invoked_calculation__'))

        self._dumps = {}
        for group_id, group_commands in self._groups.items():
            for group in group_commands:
                for dump_id, dump in group.dumps.items():
                    if dump_id not in self._dumps:
                        self._dumps[dump_id] = dump
                        # Fix paths of the dumps to store them in the dump directory
                        if not dump.file.startswith(self._dump_folder):
                            pass
                            # dump.file = join(self._dump_folder, dump.file)

    def __getitem__(self, item):
        self._check_ready(raise_error=True)
        if isinstance(item, str):
            # Its just one string
            item = (slice(None, None, None), item)
        elif is_iterable(item):
            # Check if it is a list of strings
            if all([isinstance(v, str) for v in item]):
                item = (slice(None, None, None), list(item))
            elif all(isinstance(v, slice) or is_iterable(v) for v in item):
                item = item
        # Check if multiple thermo outputs were generated
        data = [np.array(thermo_data.loc[item]) for thermo_data in self._thermo_data]
        if len(data) == 1:
            data = data[0]
        return data

    def write_input_files(self, *args, **kwargs):
        create_directory(self.working_directory)
        create_directory(self.get_path(self._structure_folder))
        create_directory(self.get_path(self._dump_folder))

        lammps_data = LammpsData.from_structure(self._structure, atom_style='atomic')
        lammps_data.write_file(self.get_path(self._structure_folder, self._structure_file_input))

        # Check species_order
        species = self._structure.symbol_set
        masses = lammps_data.masses['mass'].astype(float).tolist()
        species_order = []
        self._species_order = species_order
        for mass in masses:
            match = False
            for specie in species:
                if isclose(Element(specie).atomic_mass.real, mass):
                    species_order.append(specie)
                    match = True
                    #Break the inner loop
                    break
            if not match:
                raise RuntimeError('Could not manage to map atomic species')
            # We do not want to get here, lets raise an Exception

        # Write control file
        with open(self.get_path(self._input_file), 'w') as input_file:
            for command in self._sequence:
                if isinstance(command, PairCoeff):
                    old_coeff = command.coeff
                    # Get all parts which do not contain species
                    #
                    is_element = predicate_generator(element_test)
                    mask = [is_element(c) for c in command.coeff]
                    elements = list(filter(is_element, command.coeff))
                    species_order_ = species_order.copy()
                    # Ensure that the atoms species in the pair_coeff command match those in the structure file .atoms
                    if len(elements) > len(species_order):
                        self.logger.info('Adapting potential command. The potential supports {} but is/are {} is needed'.format(elements, species_order))
                        while len(mask) > len(species_order)+1:
                            mask.pop(mask.index(True))

                        old_cmd = str(command).rstrip()
                        command.coeff = [c if not m else species_order_.pop(0) for c, m in zip(command.coeff, mask)]
                        self.logger.info('Adapted "pair_coeff" command from "{}" to "{}"'.format(old_cmd, str(command).rstrip()))
                    elif len(elements) == len(species_order):
                        if not elements == species_order:
                            old_cmd = str(command).rstrip()
                            command.coeff = [c if not m else species_order_.pop(0) for c, m in zip(command.coeff, mask)]
                            self.logger.info('Adapted "pair_coeff" command from "{}" to "{}"'.format(old_cmd, str(command).rstrip()))
                        else:
                            self.logger.info('Potential is OK!')
                    else:
                        raise RuntimeError('An error ocurred while adapting the potential command')
                    input_file.write(str(command))
                    command.coeff = old_coeff
                    continue # Skip write for this command
                input_file.write(str(command))

        # Write potential files
        for potential in self._potential:
            with open(self.get_path(potential.file_name), 'wb') as potential_file:
                potential_file.write(potential.data)

    def success(self, *args, **kwargs):
        success = True
        paths = [(self._dump_folder, self._dump_system_final),
                 (self._structure_folder, self._structure_file_final),
                 (self._log_file,)]
        paths.extend([(w.file,) for w in self._group_write_dumps.values()])
        if not all([exists(self.get_path(*p)) for p in paths]):
            success = False
        return success and self._runner_success

    def process(self, *args, **kwargs):
        try:
            # Very important sort everything by id
            # Parse final structure
            self._final_structure = LammpsData.from_file(
                self.get_path(self._structure_folder, self._structure_file_final), atom_style='atomic').structure
            # Parse force dump, velocities are in site_properties of final_structure
            dump_obj = list(parse_lammps_dumps(self.get_path(self._dump_folder, self._dump_system_final)))
            assert len(dump_obj) == 1
            self._final_forces = dump_obj[0].data
            for group_id, write_dump in self._group_write_dumps.items():
                write_dump.parse_data(prefix=self.get_path(''))
                # Store group id's so that each we know which atoms are in which group
                dump_id_list = np.array(write_dump.data.loc[:, 'id']).tolist()
                assert len(self._groups_ids[group_id]) == len(dump_id_list)
                self._groups_ids[group_id] = dump_id_list
            # Write forces to site properties
            # Append forces to
            site_properties = self._final_structure.site_properties
            # Ensure to parse forces in correct order
            site_properties['forces'] = np.array(self._final_forces.loc[:, ('fx', 'fy', 'fz')])
            # Assign real previous ids
            site_properties['id'] = np.array([self._id_mapping[lammps_id] for lammps_id in self._final_forces.loc[:, 'id']])

            property_mapping = {site.properties['id']: {k : v for k, v in site.properties.items() if k not in ['id', 'forces']}  for site in self._structure}

            for key, property in self._structure.site_properties.items():
                if key not in site_properties:
                    site_properties[key] = [property_mapping[id_][key] for id_ in site_properties['id']]

            # Assign each site the id from the all group dump

            self._final_structure = self._final_structure.copy(site_properties=site_properties)
            # Sort by id
            self._final_structure = self._final_structure.get_sorted_structure(key=lambda s: s.properties['id'])
            self._thermo_data = parse_lammps_log(self.get_path(self._log_file))

            # Reassign column headers of data frame
            for thermo_data in self._thermo_data:
                thermo_data.columns = list(self._thermo_args)

            # Take care of all other dumps

            for dump_id, dump in self._dumps.items():
                dump.parse_data(prefix=self.get_path(''))

        except Exception as e:
            self.logger.exception('An error occurred while processing the data', exc_info=e)
            return False
        return True

    def execute(self, *args, **kwargs):
        if 'preamble' in kwargs:
            preamble = kwargs['preamble']
        else:
            preamble = ()

        if 'remote' in kwargs:
            remote = kwargs['remote']
        else:
            remote = None

        if 'manager' in kwargs:
            manager = kwargs['manager']
        else:
            manager = None

        if 'command' in kwargs:
            command = kwargs['command']
        else:
            command = get_setting('lammps_command')

        if not self._runner_bound:
            self._runner.bind(self, remote=remote, manager=manager)
            self._runner_bound = True
        try:
            status = self._runner.run(command, preamble=preamble)
        except RuntimeError as e:
            self.logger.exception('An error occurred while executing LAMMPS!', exc_info=e)
            self._runner_success = False
        else:
            self._runner_success = status if status is None else status == 0
        return self._runner_success

    def __del__(self):
        if self._runner_bound:
            self._runner.unbind()

    @property
    def final_structure(self):
        self._check_ready(raise_error=True)
        return self._final_structure

    @property
    def forces(self):
        self._check_ready(raise_error=True)
        return np.array(self._final_forces.loc[:, ('fx', 'fy', 'fz')])

    @property
    def potential_energy(self):
        self._check_ready(raise_error=True)
        if 'pe' not in self._thermo_args:
            raise ValueError('potential energy was not displayed in thermo_style')
        pe = self['pe'][-1]
        if is_iterable(pe) and len(pe) == 1:
            return pe[0]
        else:
            return pe

    def get_structure(self, group='all'):
        self._check_ready(raise_error=True)
        if isinstance(group, Group):
            group = group.identifier
        elif isinstance(group, str):
            group = group
        else:
            raise TypeError('Group argument must be a group object or group_id string')

        group_ids = self._groups_ids[group]
        sites = []
        # Filter structure
        reverse_mapping = {v: k for k, v in self._id_mapping.items()}
        for site in self.final_structure.sites:
            if 'id' not in site.properties:
                raise KeyError('id not in Structure.site_properties')
            local_id = reverse_mapping[site.properties['id']]
            if local_id in group_ids:
                sites.append(site)

        return Structure.from_sites(sites).get_sorted_structure()

    @property
    def groups(self):
        return self._groups

    def get_dumps(self, group='all'):
        if isinstance(group, Group):
            group = group.identifier
        elif isinstance(group, str):
            group = group
        else:
            raise TypeError('Group argument must be a group object or group_id string')
        return self._groups[group][0].dumps

    def get_dump(self, dump, group='all'):
        if isinstance(dump, Dump):
            dump = dump.identifier
        elif isinstance(dump, str):
            dump = dump
        else:
            raise TypeError('Dump argument must be a Dump object or dump_id string')
        return self.get_dumps(group=group)[dump]

    def as_dict(self):
        d = super(LAMMPSCalculation, self).as_dict()
        d['runner'] = self._runner.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        # TODO: This is just a workaround for a bug which existed before
        if 'runner' not in d:
            d['runner'] = LAMMPSRunner().as_dict()
        obj._runner = LAMMPSRunner.from_dict(d['runner'])
        return obj

    def reset(self, *args, **kwargs):
        self._status = Status.NotInitialized
        if self._runner_bound:
            self._runner.unbind()
            self._runner_bound = False
        self._runner = LAMMPSRunner()


class VASPCalculation(Calculation):
    XC_FUNCS = ['lda', 'pbe']

    def __init__(self, structure, name, working_directory=None):
        if not working_directory:
            working_directory = '{}.vasp'.format(name)
        super(VASPCalculation, self).__init__(structure, name=name, working_directory=working_directory)
        self._runner = VASPRunner()


        # Group structure sites by species to have nice poscar file
        # But rememeber the id if it's in site_properties
        ordered_structure = self._structure.get_sorted_structure()
        self._structure = ordered_structure
        if 'id' not in self._structure.site_properties:
            self._structure.add_site_property('id', [i + 1 for i, _ in enumerate(self._structure)])

        self._index_id_map = {i: v for i, v in enumerate(self._structure.site_properties['id'])}

        self._poscar = Poscar(self.structure)
        self._configuration = Configuration()
        self._runner_bound = False
        self._xc_func = None
        self._log_file = '{}.log'.format(name)
        self._vasprun = None
        self._oszicar = None
        self._outcar = None
        self._final_structure = None
        self._final_energy = None

    def as_dict(self):
        d = super(VASPCalculation, self).as_dict()
        d['xc_func'] = self._xc_func
        d['runner'] = self._runner.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        obj._runner = VASPRunner.from_dict(d['runner'])
        obj._xc_func = d['xc_func']
        return obj

    @property
    def poscar(self):
        return self._poscar

    @property
    def potcar(self):
        return self._potcar

    @property
    def incar(self):
        return self._incar

    @property
    def kpoints(self):
        return self._kpoints

    def write_input_files(self, *args, **kwargs):
        create_directory(self.working_directory)

        self._incar.write_file(self.get_path('INCAR'))
        self._kpoints.write_file(self.get_path('KPOINTS'))
        self._poscar.write_file(self.get_path('POSCAR'))
        self._potcar.write_file(self.get_path('POTCAR'))

    def initialize(self, *args, **kwargs):
        if 'incar' in kwargs:
            incar = kwargs['incar']
        else:
            raise ValueError('incar argument is missing')

        if isinstance(incar, Incar):
            self._incar = incar
        elif isinstance(incar, str):
            self._incar = Incar.from_file(incar)
        elif isinstance(incar, dict):
            self._incar = Incar(params=incar)
        else:
            raise TypeError('potcar argument must be of type pymatgen.io.vasp.Incar, dict or str')

        if 'kpoints' in kwargs:
            kpoints = kwargs['kpoints']
        else:
            raise ValueError('kpoints argument is missing')

        if isinstance(kpoints, Kpoints):
            self._kpoints = kpoints
        elif isinstance(kpoints, str):
            self._kpoints = Kpoints.from_file(kpoints)
        else:
            raise TypeError('potcar argument must be of type pymatgen.io.vasp.Kpoints or str')

        if 'xc_func' in kwargs:
            xc_func = kwargs['xc_func']
        else:
            xc_func = 'gga'
        self._xc_func = xc_func

        if 'potcar' in kwargs:
            potcar = kwargs['potcar']
        else:
            potcar = None

        if potcar is not None:
            if isinstance(potcar, Potcar):
                self._potcar = potcar
            elif isinstance(potcar, str):
                self._potcar = Potcar.from_file(potcar)
            else:
                raise TypeError('potcar argument must be of type pymatgen.io.vasp.Potcar or str')
        else:
            potcar = construct_potcar(self._poscar,)

        self._potcar = potcar

    def reset(self, *args, **kwargs):
        self._status = Status.NotInitialized
        if self._runner_bound:
            self._runner.unbind()
            self._runner_bound = False
        self._runner = VASPRunner()

    def process(self, *args, **kwargs):
        # Try to parse all output files
        try:
            self._oszicar = Oszicar(self.get_path('OSZICAR'))
            self._vasprun = Vasprun(self.get_path('vasprun.xml'))
            #self._final_structure = Poscar.from_file(self.get_path('CONTCAR')).structure
            self._final_structure = self._vasprun.final_structure
            # Add forces to it
            site_properties = self._structure.site_properties
            site_properties['forces'] = self._vasprun.forces
            for property_name, values in site_properties.items():
                self._final_structure.add_site_property(property_name, values)
            self._final_energy = float(self._oszicar.final_energy)
        except Exception as e:
            self.logger.exception('Failed to parse VASP output for calculation {}:{}!'.format(self.name, self.id), exc_info=e)
            return False
        else:
            return True

    @property
    def potential_energy(self):
        self._check_ready(raise_error=True)
        return self._final_energy

    def execute(self, *args, **kwargs):
        if 'preamble' in kwargs:
            preamble = kwargs['preamble']
        else:
            preamble = ()

        if 'remote' in kwargs:
            remote = kwargs['remote']
        else:
            remote = None

        if 'manager' in kwargs:
            manager = kwargs['manager']
        else:
            manager = None

        if 'command' in kwargs:
            command = kwargs['command']
        else:
            command = get_setting('vasp_command')

        if not self._runner_bound:
            self._runner.bind(self, remote=remote, manager=manager)
            self._runner_bound = True
        try:
            status = self._runner.run(command, preamble=preamble)
        except RuntimeError as e:
            self.logger.exception('An error occurred while executing VASP!', exc_info=e)
            self._runner_success = False
        else:
            self._runner_success = status if status is None else status == 0
        return self._runner_success

    def success(self, *args, **kwargs):
        # Check only if the most important files are here
        paths = [
            ('OSZICAR',),
            ('OUTCAR',),
            ('vasprun.xml', ),
            ('CONTCAR', )
        ]
        success =  all([exists(self.get_path(*p)) for p in paths])
        return success and self._runner_success


    @property
    def log_file(self):
        return self._log_file

    def __del__(self):
        if self._runner_bound:
            self._runner.unbind()

    @property
    def final_structure(self):
        self._check_ready(raise_error=True)
        return self._final_structure


    @property
    def oszicar(self):
        self._check_ready(raise_error=True)
        return self._oszicar

    @property
    def vasprun(self):
        self._check_ready(raise_error=True)
        return self._vasprun