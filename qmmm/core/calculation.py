from abc import ABCMeta, abstractmethod
from qmmm.core.event import Event
from qmmm.core.utils import is_iterable, create_directory, remove_white, LoggerMixin, recursive_as_dict, \
    process_decoded, StructureWrapper
from qmmm.core.configuration import get_setting
from qmmm.core.vasp.archive import TarPotentialArchive, DirectoryPotentialArchive, PotentialException
from qmmm.core.vasp import potcar_from_string
from qmmm.core.vasp.vasprun import Vasprun
from os.path import join, exists, isfile
from qmmm.core.lammps import Command, Clear, Units, Boundary, AtomStyle, Dimension, ReadData, PairStyle, PairCoeff, CommandStyle, \
    WriteDump, WriteData, ThermoStyle, Thermo, ThermoModify, Group, Dump
from qmmm.core.lammps.group import All
from qmmm.core.runner import VASPRunner, LAMMPSRunner
from qmmm.core.mongo import get_mongo_client, make_database_config, insert, query
from enum import Enum
from math import isclose
from pymatgen.io.lammps.inputs import LammpsData
from pymatgen.core.periodic_table import Element
from io import StringIO
from pymatgen import Structure
from pymatgen.io.vasp import Potcar, Incar, Kpoints, Poscar, Oszicar, Outcar
from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from qmmm.core.configuration import Configuration
import numpy as np
import pickle
import os
import tarfile
from monty.json import MSONable
from uuid import uuid4

from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log


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
                        raise RuntimeError('Failed to parse outputs in directory: {}'.format(calc.working_directory))
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
        self._region_box = None
        self._num_species = len(set([site.species_string for site in structure]))
        self._runner = LAMMPSRunner()
        self._runner_success = None
        self._final_forces = None
        self._final_structure = None
        self._thermo_data = None
        self._thermo_style = None
        self._groups = {'all': [All]}
        self._groups_ids = {}
        self._group_write_dumps = {}
        self._dumps = {}
        self._runner_bound = False

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
            raise ValueError('Canno\'t handle simulation boxes which are not orthorhombic')

        return [
            ReadData(join(self._structure_folder, self._structure_file_input), group='all')
        ]

    def define_potential(self, pair_style, pair_coeff):
        local_sequence = []
        if isinstance(pair_style, CommandStyle):
            local_sequence.append(PairStyle(pair_style))
        elif isinstance(pair_style, Command):
            local_sequence.append(pair_style)
        elif isinstance(pair_style, type):
            if issubclass(pair_style, CommandStyle):
                local_sequence.append(PairStyle(pair_style()))
            else:
                raise ValueError
        else:
            raise ValueError
        if not is_iterable(pair_coeff):
            pair_coeff = [pair_coeff]
        for coeff in pair_coeff:
            if isinstance(coeff, PairCoeff):
                local_sequence.append(coeff)
            elif is_iterable(coeff):
                i, j, args = coeff
                local_sequence.append(PairCoeff(i, j, args))
            else:
                raise ValueError

        return local_sequence

    def define_run(self, commands=None):
        return commands or []

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

    def define_preamble(self):

        thermo_style = ThermoStyle(ThermoStyle.Custom, list(self._thermo_args))
        self._thermo_style = thermo_style.style
        return [
            Thermo(1),
            thermo_style,
            ThermoModify(format='float %20.15g'),
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
        if 'pair_style' not in kwargs:
            raise ValueError('pair_style argument is missing')
        if 'pair_coeff' not in kwargs:
            raise ValueError('pair_coeff argument is missing')
        self._sequence.extend(self.define_initialization())
        self._sequence.extend(self.define_atoms())

        potential_args = {k: kwargs[k] for k in ['pair_style', 'pair_coeff']}
        self._sequence.extend(self.define_potential(**potential_args))
        if 'commands' in kwargs:
            commands = kwargs['commands']
        else:
            commands = None
        if 'settings' in kwargs:
            settings = kwargs['settings']
        else:
            settings = None

            # Extract groups

        self._sequence.extend(self.define_settings(settings=settings))
        self._sequence.extend(self.define_preamble())
        self._sequence.extend(self.define_run(commands=commands))
        for command in self._sequence:
            if isinstance(command, Group):
                # Check if group has style
                if command.identifier not in self._groups:
                    self._groups[command.identifier] = [command]
                else:
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
        return np.array(self._thermo_data.loc[item])

    def write_input_files(self, *args, **kwargs):
        create_directory(self.working_directory)
        create_directory(self.get_path(self._structure_folder))
        create_directory(self.get_path(self._dump_folder))

        lammps_data = LammpsData.from_structure(self._structure, atom_style='atomic')
        lammps_data.write_file(self.get_path(self._structure_folder, self._structure_file_input))

        with open(self.get_path(self._input_file), 'w') as input_file:
            for command in self._sequence:
                input_file.write(str(command))

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
                self._groups_ids[group_id] = np.array(write_dump.data.loc[:, 'id']).tolist()
            # Write forces to site properties
            # Append forces to
            site_properties = self._final_structure.site_properties
            site_properties['forces'] = np.array(self._final_forces.loc[:, ('fx', 'fy', 'fz')])
            site_properties['id'] = np.array(self._final_forces.loc[:, 'id'])
            # Assign each site the id from the all group dump

            self._final_structure = self._final_structure.copy(site_properties=site_properties)
            self._thermo_data = parse_lammps_log(self.get_path(self._log_file))[0]

            # Reassign column headers of data frame
            self._thermo_data.columns = list(self._thermo_args)

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
        return self['pe'][-1]

    def get_structure(self, group='all'):
        self._check_ready(raise_error=True)
        if isinstance(group, Group):
            group = group.identifier
        elif isinstance(group, str):
            group = group
        else:
            raise TypeError('Group argument must be a group object or group_id string')

        site_properties = {k: [] for k in self.final_structure.site_properties.keys()}

        group_ids = self._groups_ids[group]
        species = []
        frac_coords = []
        # Filter structure
        for site in self.final_structure.sites:
            if 'id' not in site.properties:
                raise KeyError('id not in Structure.site_properties')
            if site.properties['id'] in group_ids:
                species.append(site.species_string)
                frac_coords.append(site.frac_coords)
            for k, v in site.properties.items():
                site_properties[k].append(v)

        return Structure(self.final_structure.lattice, species, frac_coords, site_properties=site_properties)

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


class VASPCalculation(Calculation):
    XC_FUNCS = ['lda', 'pbe']

    def __init__(self, structure, name, incar, kpoints, potcar=None, working_directory=None, xc_func='pbe'):
        if not working_directory:
            working_directory = '{}.vasp'.format(name)
        super(VASPCalculation, self).__init__(structure, name=name, working_directory=working_directory)
        self._runner = VASPRunner()
        if potcar is not None:
            if isinstance(potcar, Potcar):
                self._potcar = potcar
            elif isinstance(potcar, str):
                self._potcar = Potcar.from_file(potcar)
            else:
                raise TypeError('potcar argument must be of type pymatgen.io.vasp.Potcar or str')
        self._potcar = potcar
        if isinstance(kpoints, Kpoints):
            self._kpoints = kpoints
        elif isinstance(kpoints, str):
            self._kpoints = Kpoints.from_file(kpoints)
        else:
            raise TypeError('potcar argument must be of type pymatgen.io.vasp.Kpoints or str')
        if isinstance(incar, Incar):
            self._incar = incar
        elif isinstance(incar, str):
            self._incar = Incar.from_file(incar)
        elif isinstance(incar, dict):
            self._incar = Incar(params=incar)
        else:
            raise TypeError('potcar argument must be of type pymatgen.io.vasp.Incar, dict or str')

        # Group structure sites by species to have nice poscar file
        # But rememeber the id if it's in site_properties
        if 'id' not in self._structure.site_properties:
            self._structure.add_site_property('id', [i + 1 for i, _ in enumerate(self._structure)])

        ordered_structure = Structure.from_sites([s for s in self._structure.group_by_types()])
        self._structure = ordered_structure.copy()
        self._index_id_map = {i: v for i, v in enumerate(self._structure.site_properties['id'])}

        self._poscar = Poscar(self.structure)
        self._poscar_species = self._extract_species()
        self._configuration = Configuration()
        self._runner_bound = False
        if xc_func.lower() == 'gga':
            xc_func = 'pbe'
        self._xc_func = xc_func.lower()

        potential_archive_setting_name = '{}_potential_archive'.format(self._xc_func)
        potential_archive_path = get_setting(potential_archive_setting_name)
        if potential_archive_path is None:
            raise PotentialException('No potential archive set for XC functional "{}". Either set {} environment '
                                     'variable or specify {} in the "general" section '
                                     'in the settings file'.format(self._xc_func,
                                                                   potential_archive_setting_name.upper(),
                                                                   potential_archive_setting_name))
        self._potential_archive = TarPotentialArchive(potential_archive_path) \
            if isfile(potential_archive_path) else DirectoryPotentialArchive(potential_archive_path)
        # Construct POTCAR if it was not yet created
        if self._potcar is None:
            self._construct_potcar()

        self._log_file = '{}.log'.format(name)
        self._vasprun = None
        self._oszicar = None
        self._outcar = None
        self._final_structure = None
        self._final_energy = None

    def _construct_potcar(self):
        archive = self._potential_archive
        functional = self._xc_func
        with NamedTemporaryFile() as final:
            for element in self._poscar_species:
                try:
                    default_potential = self._configuration[join('potentials', element.lower())]
                except PotentialException:
                    self.logger.warning('No default POTCAR found for "{}" for element "{}"'.format(functional, element))
                    default_potential = element
                copyfileobj(archive.potcar(default_potential), final)
            final.seek(0)
            self._potcar = potcar_from_string(final.read().decode('utf-8'))
            if not self._poscar_species == [p.element for p in self._potcar]:
                raise PotentialException('Something went wrong while constructing the POTCAR file')

    def _extract_species(self):
        with StringIO(self._poscar.get_string()) as poscar:
            # Skip the first 5 lines
            for _ in range(5):
                poscar.readline()
            element_line = [remove_white(crumb) for crumb in poscar.readline().split(' ') if
                            remove_white(crumb) != '']
            try:
                for element in element_line:
                    Element(element)
            except ValueError:
                return []
            else:
                return element_line

    def as_dict(self):
        d = super(VASPCalculation, self).as_dict()
        d['incar'] = self.incar.as_dict()
        d['kpoints'] = self.kpoints.as_dict()
        d['xc_func'] = self._xc_func
        d['runner'] = self._runner.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        if 'structure' in d:
            d['structure'] = StructureWrapper.from_dict(d['structure'])
        decoded = {k: process_decoded(v) for k, v in d.items()
                   if not k.startswith("@")}
        obj = cls(decoded['structure'], decoded['name'], decoded['incar'], decoded['kpoints'],
                  working_directory=decoded['working_directory'], xc_func=decoded['xc_func'])
        Calculation._set_internal(obj, decoded)
        obj._runner = decoded['runner']
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
        pass

    def process(self, *args, **kwargs):
        # Try to parse all output files
        try:
            self._oszicar = Oszicar(self.get_path('OSZICAR'))
            self._vasprun = Vasprun(self.get_path('vasprun.xml'))

            final_structure = self._vasprun.final_structure
            # Add forces to it
            site_properties = self._structure.site_properties
            site_properties['forces'] = self._vasprun.forces
            self._final_energy = float(self._oszicar.final_energy)
        except:
            return False
        else:
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
