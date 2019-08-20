import abc
import re
import tarfile
from os.path import exists, isfile, isdir, join
import json
from shutil import copyfileobj
from io import StringIO
from qmmm.core.vasp import potcar_from_string
from tempfile import NamedTemporaryFile
from os import listdir
from os.path import isdir, isfile
from qmmm.core.utils import get_configuration_directory, VASP_DIRECTORY, RESOURCE_DIRECTORY, LoggerMixin, remove_white
import logging


POTENTIAL_ARCHIVES = {}
DEFAULT_CONFIG = 'defaults.json'
DEFAULT_POTENTIALS = {}


class PotentialException(Exception):
    def __init__(self, msg):
        super(PotentialException, self).__init__(msg)


def copy_file(fsrc, fdst, length=16 * 1024):
    """copy data from file-like object fsrc to file-like object fdst"""
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


class PotentialArchive(LoggerMixin):
    __metaclass__ = abc.ABCMeta

    def __init__(self, path, xc_func='gga'):
        self.path = path
        self.xc_func = xc_func

    @staticmethod
    def copy_file(fsrc, fdst, length=16 * 1024):
        """copy data from file-like object fsrc to file-like object fdst"""
        while 1:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)

    @abc.abstractmethod
    def has_potential(self, identifier):
        """
        Returns a boolean wether the potential is available
        """

    @abc.abstractmethod
    def is_valid_potential(self, identifier):
        """ Returns a boolean wether the potential has a POTCAR and a PSCTR file """

    @abc.abstractmethod
    def check_valid_archive(self):
        """Returns a boolean wether the archive is valid (all potentials are valid and all default potentials are available """

    @abc.abstractmethod
    def potentials(self):
        """ Returns a list of all potential names """

    @abc.abstractmethod
    def potcar(self, identifier):
        """ Returns a file stream for the POTCAR file"""

    @abc.abstractmethod
    def psctr(self, identifier):
        """ Returns a file stream for the POTCAR file"""

    @abc.abstractmethod
    def get_potentials_for_element(self, element):
        """docstring"""

    def default_potential(self, element):
        global DEFAULT_POTENTIALS
        if element not in DEFAULT_POTENTIALS[self.xc_func]:
            raise PotentialException('No default potential configured for element "{}" for xc_type="{}"'
                                     .format(element, self.xc_func))
        else:
            return DEFAULT_POTENTIALS[self.xc_func][element]


class DirectoryPotentialArchive(PotentialArchive):
    def __init__(self, path):
        super(DirectoryPotentialArchive, self).__init__(path)
        if not isdir(path):
            raise PotentialException('{1}: {0} is not directory'.format(path, self.__class__.__name__))
        self._potential_directories = list(
                filter(
                        lambda pth: isdir(join(self.path, pth)) and not pth.startswith('.'), listdir(self.path)
                )
        )
        #if not self.is_valid_archive():
        #    raise PotentialException('{} is not a valid VASP potential archive'.format(path))

    def has_potential(self, identifier):
        return identifier in self._potential_directories

    def potentials(self):
        return list(self._potential_directories)

    def is_valid_potential(self, identifier):
        if self.has_potential(identifier):
            potcar_path = join(self.path, identifier, 'POTCAR')
            #psctr_path = join(self.path, identifier, 'PSCTR')
            return exists(potcar_path) and isfile(potcar_path)
                   #and exists(psctr_path) \
                   #and isfile(psctr_path)
        else:
            return False

    def check_valid_archive(self):
        for potential in self.potentials():
            if not self.is_valid_potential(potential):
                self.logger.warning('{} potential is corrupted'.format(potential))
                return False

        default_potentials = DEFAULT_POTENTIALS[self.xc_func]

        for element, potential in default_potentials.items():
            if not self.is_valid_potential(potential):
                self.logger.warning('{} default potential is corrupted'.format(potential))
                return False
        return True

    def potcar(self, identifier):
        if self.is_valid_potential(identifier):
            potcar_path = join(self.path, identifier, 'POTCAR')
            return open(potcar_path, mode='rb')
        else:
            return None

    def psctr(self, identifier):
        if self.is_valid_potential(identifier):
            psctr_path = join(self.path, identifier, 'PSCTR')
            return open(psctr_path, mode='rb')
        else:
            return None

    def get_potentials_for_element(self, element):
        return list(filter(lambda pth: pth.startswith(element), self._potential_directories))


class TarPotentialArchive(PotentialArchive):
    def __init__(self, path):
        super(TarPotentialArchive, self).__init__(path)
        if not isfile(path):
            raise PotentialException('{1}: {0} is not a file'.format(path, self.__class__.__name__))
        else:
            if not path.split('.')[-1] in ['bz2', 'tar', 'gz', 'xz']:
                raise PotentialException('{1}: {0} is not a tar archive'.format(path, self.__class__.__name__))

        self._tarfile = tarfile.open(path, 'r:*')
        self._tarinfo = self._tarfile.getmembers()
        self._names = list(map(lambda info: info.name, self._tarinfo))
        self.check_valid_archive()
        #    raise PotentialException('{} is not a valid VASP potential archive'.format(path))

    def check_valid_archive(self):
        for potential in self.potentials():
            if not self.is_valid_potential(potential):
                self.logger.warning('{} default potential is corrupted'.format(potential))
                return False

        default_potentials = DEFAULT_POTENTIALS[self.xc_func]

        for element, potential in default_potentials.items():
            if not self.is_valid_potential(potential):
                self.logger.warning('{} potential is corrupted'.format(potential))
                return False
        return True

    def is_valid_potential(self, identifier):
        if self.has_potential(identifier):
            potcar_path = join(identifier, 'POTCAR')
            psctr_path = join(identifier, 'PSCTR')
            return potcar_path in self._names and psctr_path in self._names
        else:
            return False

    def has_potential(self, identifier):
        return identifier in self._names

    def potentials(self):
        return list(
                map(lambda info: info.name,
                    list(filter(
                            lambda info: info.isdir(), self._tarinfo)
                        )
                    )
        )

    def get_potentials_for_element(self, element):
        return list(filter(lambda pot: pot.startswith(element), self.potentials()))

    def potcar(self, identifier):
        if self.is_valid_potential(identifier):
            potcar_path = join(identifier, 'POTCAR')
            if potcar_path in self._names:
                try:
                    member = list(filter(lambda inf: inf.name == potcar_path, self._tarinfo))[0]
                    file_obj = self._tarfile.extractfile(member)
                except:
                    raise PotentialException('An error occured while extracting {0}.'.format(potcar_path))
                else:
                    return file_obj
            else:
                raise PotentialException('The POTCAR file for the potential {0} was not found.'.format(identifier))
        else:
            return None

    def psctr(self, identifier):
        if self.is_valid_potential(identifier):
            psctr_path = join(identifier, 'PSCTR')
            if psctr_path in self._names:
                try:
                    member = list(filter(lambda inf: inf.name == psctr_path, self._tarinfo))[0]
                    file_obj = self._tarfile.extractfile(member)
                except:
                    raise PotentialException('An error occured while extracting {0}.'.format(psctr_path))
                else:
                    return file_obj
            else:
                raise PotentialException('The PSCTR file for the potential {0} was not found.'.format(identifier))
        else:
            return None


def _make_porential_archives():
    global POTENTIAL_ARCHIVES, DEFAULT_POTENTIALS
    default_potential_config = join(get_configuration_directory(), RESOURCE_DIRECTORY, VASP_DIRECTORY, DEFAULT_CONFIG)
    with open(default_potential_config, 'rb') as default_potential_config_file:
        default_potentials = json.load(default_potential_config_file)
        DEFAULT_POTENTIALS = default_potentials
    functionals = list(default_potentials.keys())
    resources_directory = join(get_configuration_directory(), RESOURCE_DIRECTORY, VASP_DIRECTORY)
    found_directories = [f for f in listdir(resources_directory)
                         if f in functionals and isdir(join(resources_directory, f))]

    # Search for potential archives
    for functional_potential_directory in found_directories:
        # Search at first for .tar.gz files
        archives = [f for f in listdir(join(resources_directory, functional_potential_directory))
                    if f.endswith('.tar.gz')]
        archive_found = False
        for archive in archives:
            # Try to find a right potential archive
            try:
                functional_archive = TarPotentialArchive(
                    join(resources_directory, functional_potential_directory, archive))
            except PotentialException:
                continue
            else:
                # We found a valid potential archive
                POTENTIAL_ARCHIVES[functional_potential_directory] = functional_archive
                logging.getLogger().info('Found valid potential archive "{}"'.format(join(resources_directory,
                                                                                   functional_potential_directory,
                                                                                   archive)))
                archive_found = True
                break
        if not archive_found:
            try:
                functional_archive = DirectoryPotentialArchive(join(resources_directory, functional_potential_directory))
            except PotentialException:
                logging.getLogger().warning('Could not find a potential archive for functional "{}"'.format(functional_potential_directory))
            else:
                POTENTIAL_ARCHIVES[functional_potential_directory] = functional_archive
                logging.getLogger().info('Found valid potential archive "{}"'.format(join(resources_directory,
                                                                                   functional_potential_directory)))


_make_porential_archives()

def _extract_species(poscar):
    from pymatgen.core.periodic_table import Element
    with StringIO(poscar.get_string()) as poscar:
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


def construct_potcar(poscar, xc_func='gga'):
    if xc_func == 'pbe':
        xc_func = 'gga'
    archive = POTENTIAL_ARCHIVES[xc_func]
    functional = xc_func
    poscar_species = _extract_species(poscar)
    with NamedTemporaryFile() as final:
        for element in poscar_species:
            try:
                default_potential = archive.default_potential(element)
            except PotentialException:
                logging.getLogger().warning('No default POTCAR found for "{}" for element "{}"'.format(functional, element))
                default_potential = element
            copyfileobj(archive.potcar(default_potential), final)
        final.seek(0)
        potcar = potcar_from_string(final.read().decode('utf-8'))
        if not poscar_species == [p.element for p in potcar]:
            raise PotentialException('Something went wrong while constructing the POTCAR file')
    return potcar



