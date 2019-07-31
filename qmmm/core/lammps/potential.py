from os.path import join, exists
from os import walk
from qmmm.core.utils import get_configuration_directory, LAMMPS_DIRECTORY, RESOURCE_DIRECTORY, LoggerMixin, HashableSet, flatten, process_decoded
import pickle

LAMMPS_POTENTIAL_DIRECTORY = join(get_configuration_directory(), RESOURCE_DIRECTORY, LAMMPS_DIRECTORY)


class LAMMPSPotential(LoggerMixin):

    def __init__(self, name, type, species, file_name, commands, data):
        self._name = name
        self._type = type
        self._species = species
        self._file_name = file_name
        self._commands = commands
        self._data = data

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def species(self):
        return self._species

    @property
    def file_name(self):
        return self._file_name

    @property
    def commands(self):
        return self._commands

    @property
    def data(self):
        return self._data

    @staticmethod
    def load(name):
        # Search potential info
        all_potentials = flatten(list(POTENTIAL_INFO.values()))
        for potential in all_potentials:
            if name == potential.name:
                return potential

    def __repr__(self):
        return 'LAMMPSPotential(name={name}, file_name={file_name}, species={species}, type={type})'.format(
            name=self.name,
            file_name=self.file_name,
            species=self.species,
            type=self.type)

    @classmethod
    def from_dict(cls, dic):
        d= {k: process_decoded(v) for k, v in dic.items()}
        return cls(**d)


def _get_info_data():
    result = {}
    for root, _, files in walk(LAMMPS_POTENTIAL_DIRECTORY):
        potential_files = [f for f in files if f.endswith('.potential')]
        for potential_file in potential_files:
            with open(join(root, potential_file), 'rb') as potential_file_handle:
                data = pickle.load(potential_file_handle)
                species = HashableSet(data['species'])
                potential = LAMMPSPotential.from_dict(data)
                if species not in result:
                    result[species] = [potential]
                else:
                    result[species].append(potential)
    return result


POTENTIAL_INFO = _get_info_data()


def available_potentials(species):
    return POTENTIAL_INFO[HashableSet(species)]