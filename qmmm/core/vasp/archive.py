import abc
import re
import tarfile
from tempfile import mkdtemp
from os.path import exists, isfile, isdir, join
from shutil import rmtree
from pymatgen.io.vasp import Oszicar
from os import listdir
from ..configuration import Configuration

settings = Configuration()


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


class PotentialArchive(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, path):
        self.path = path

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
    def is_valid_archive(self):
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
        try:
            return settings.get_option('default', element)
        except:
            raise PotentialException('No default potential found for {0}'.format(element))


class ResultArchive(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, path):
        self.path = path

    @abc.abstractmethod
    def is_valid_archive(self):
        """Returns wether the archive is valid"""

    @abc.abstractmethod
    def kpoint_list(self):
        """Returns a list of available k points"""

    @abc.abstractmethod
    def energy_list(self):
        """Returns a list of energies"""

    @abc.abstractmethod
    def mesh(self):
        """Returns a list of tuples of kpoints and corresponding energies"""

    @abc.abstractmethod
    def data(self):
        """Returns a list of tuples of all data (cutoff, kpoint, total)"""

    @abc.abstractmethod
    def potcar(self):
        """Returns a file object of the POTCAR file"""

    @abc.abstractmethod
    def poscar(self):
        """Returns a file object of the POSCAR file"""

    @abc.abstractmethod
    def psctr(self):
        """Returns a file object of the PSCTR file"""

    @abc.abstractmethod
    def kpoints(self):
        """Returns a file object of the KPOINTS file"""

    @abc.abstractmethod
    def incar(self):
        """Returns a file object of the INCAR file"""


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

    def is_valid_archive(self):
        for potential in self.potentials():
            if not self.is_valid_potential(potential):
                print(potential)
                return False

        default_potentials = [settings.get_option('default', option) for option in settings.get_options('default')]

        for potential in default_potentials:
            if not self.is_valid_potential(potential):
                print(potential)
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

    def is_valid_archive(self):
        for potential in self.potentials():
            if not self.is_valid_potential(potential):
                print(potential)
                return False

        default_potentials = [settings.get_option('default', option) for option in settings.get_options('default')]

        for potential in default_potentials:
            if not self.is_valid_potential(potential):
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


            # class Directory:
            # re.findall(r"[-+]?\d*\.\d+|\d+", s)


class DirectoryResultArchive(ResultArchive):
    def __init__(self, path):
        super(DirectoryResultArchive, self).__init__(path)
        if not isdir(path):
            raise PotentialException('{1}: {0} is not a directory'.format(path, self.__class__.__name__))
        self._directories = list(
                filter(
                        lambda pth: isdir(join(self.path, pth)) and not pth.startswith('.'), listdir(self.path)
                )
        )
        self._kpoints = []
        self._energies = []
        self._data = []
        if self.is_valid_archive():
            self.incar_path = join(self.path, 'master', 'INCAR')
            self.poscar_path = join(self.path, 'master', 'POSCAR')
            self.kpoints_path = join(self.path, 'master', 'KPOINTS')
            self.potcar_path = join(self.path, 'master', 'POTCAR')
            self.psctr_path = join(self.path, 'master', 'PSCTR')
            for directory in self._directories:
                if directory != 'master':
                    energy_, kpoints_ = re.findall(r"[-+]?\d*\.\d+|\d+", directory)
                    energy_, kpoints_ = int(energy_), int(kpoints_)
                    if kpoints_ not in self._kpoints:
                        self._kpoints.append(kpoints_)
                    if energy_ not in self._energies:
                        self._energies.append(energy_)

                    oszicar_path = join(self.path, directory, 'OSZICAR')
                    try:
                        oszicar = Oszicar(oszicar_path)
                    except:
                        raise IOError('Could open OSCICAR file "{0}"'.format(oszicar_path))
                    else:
                        try:
                            self._data.append((kpoints_, energy_, float(oszicar.final_energy)))
                        except:
                            pass
        else:
            raise PotentialException('{1}: {0} is not a valid directory'.format(path, self.__class__.__name__))

    def is_valid_archive(self):
        incar_path, poscar_path, kpoints_path, potcar_path, psctr_path = \
            join(self.path, 'master', 'INCAR'), \
            join(self.path, 'master', 'POSCAR'), \
            join(self.path, 'master', 'KPOINTS'), \
            join(self.path, 'master', 'POTCAR'), \
            join(self.path, 'master', 'PSCTR')

        valid_dirs = True
        for directory in self._directories:
            if directory != 'master':
                integers = re.findall(r"[-+]?\d*\.\d+|\d+", directory)
                if len(integers) != 2:
                    valid_dirs = False
                energy, kpoints = integers
                try:
                    energy = int(energy)
                    kpoints = int(kpoints)
                except:
                    valid_dirs = False
                valid_dirs = exists(join(self.path, directory, 'OSZICAR'))
            if not valid_dirs:
                break

        return exists(incar_path) and exists(poscar_path) and exists(kpoints_path) and exists(potcar_path) and exists(
                psctr_path) and valid_dirs

    def incar(self):
        return open(self.incar_path)

    def poscar(self):
        return open(self.poscar_path)

    def potcar(self):
        return open(self.potcar_path)

    def kpoints(self):
        return open(self.kpoints_path)

    def psctr(self):
        return open(self.psctr_path)

    def kpoint_list(self):
        return self._kpoints

    def energy_list(self):
        return self._energies

    def mesh(self):
        pass

    def data(self):
        return self._data


class TarTesultArchive(ResultArchive):
    def __init__(self, path):
        super(TarTesultArchive, self).__init__(path)
        if not isfile(path):
            raise PotentialException('{1}: {0} is not a file'.format(path, self.__class__.__name__))
        else:
            if not path.split('.')[-1] in ['bz2', 'tar', 'gz', 'xz', 'proj']:
                raise PotentialException('{1}: {0} is not a tar archive'.format(path, self.__class__.__name__))

        self._tarfile = tarfile.open(path, 'r:*')
        self._tarinfo = self._tarfile.getmembers()
        self._names = list(map(lambda info: info.name, self._tarinfo))
        self._kpoints = []
        self._energies = []
        self._data = []
        if self.is_valid_archive():
            self.incar_path = join('master', 'INCAR')
            self.poscar_path = join('master', 'POSCAR')
            self.kpoints_path = join('master', 'KPOINTS')
            self.potcar_path = join('master', 'POTCAR')
            self.psctr_path = join('master', 'PSCTR')

            directories = list(filter(lambda inf: inf.isdir(), self._tarinfo))

            temp_oszicar_directory = mkdtemp()


            for directory in directories:
                if directory.name != 'master':
                    energy_, kpoints_ = re.findall(r"[-+]?\d*\.\d+|\d+", directory.name)
                    energy_, kpoints_ = int(energy_), int(kpoints_)
                    if kpoints_ not in self._kpoints:
                        self._kpoints.append(kpoints_)
                    if energy_ not in self._energies:
                        self._energies.append(energy_)

                    oszicar_archive_path = join(directory.name, 'OSZICAR')
                    oszicar_path = join(temp_oszicar_directory, 'OSZICAR_ENCUT{0}_K{1}'.format(energy_, kpoints_))
                    #print(oszicar_path, oszicar_path in self._names)
                    if oszicar_archive_path in self._names:
                        oszicar_file = open(oszicar_path, "wb")
                        copy_file(self._tarfile.extractfile(oszicar_archive_path), oszicar_file)
                        oszicar_file.close()
                        try:
                            oszicar = Oszicar(oszicar_path)
                        except:
                            raise IOError('Could open OSCICAR file "{0}"'.format(oszicar_path))
                        else:
                            try:
                                self._data.append((kpoints_, energy_, float(oszicar.final_energy)))
                            except:
                                pass

            rmtree(temp_oszicar_directory)

    def is_valid_archive(self):
        incar_path, poscar_path, kpoints_path, potcar_path = \
            join('master', 'INCAR'), \
            join('master', 'POSCAR'), \
            join('master', 'KPOINTS'), \
            join('master', 'POTCAR')

        directories = list(filter(lambda inf: inf.isdir(), self._tarinfo))
        valid_dirs = True
        for directory in directories:
            if directory.name != 'master':
                integers = re.findall(r"[-+]?\d*\.\d+|\d+", directory.name)
                if len(integers) != 2:
                    valid_dirs = False
                energy, kpoints = integers
                try:
                    energy = int(energy)
                    kpoints = int(kpoints)
                except:
                    valid_dirs = False
                valid_dirs = join(directory.name, 'OUTCAR') in self._names
            if not valid_dirs:
                break

        return incar_path in self._names and \
               poscar_path in self._names and \
               potcar_path in self._names and \
               kpoints_path in self._names and valid_dirs

    def incar(self):
        return self._tarfile.extractfile(self.incar_path)

    def poscar(self):
        return self._tarfile.extractfile(self.poscar_path)

    def potcar(self):
        return self._tarfile.extractfile(self.potcar_path)

    def kpoints(self):
        return self._tarfile.extractfile(self.kpoints_path)

    def psctr(self):
        return self._tarfile.extractfile(self.psctr_path)

    def kpoint_list(self):
        return self._kpoints

    def energy_list(self):
        return self._energies

    def mesh(self):
        pass

    def data(self):
        return self._data
