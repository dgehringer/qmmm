from functools import wraps
from os.path import exists, join
from os import mkdir, environ as ENVIRONMENT, getcwd, chdir, getcwd, remove
from shutil import rmtree
from time import sleep
from io import StringIO
from threading import Thread
from monty.json import MontyDecoder
from pymatgen import Structure, Lattice
from itertools import groupby
from operator import itemgetter
from pymatgen import Structure
from uuid import uuid4
import logging
import hashlib
import numpy as np


__ENCODER__ = MontyDecoder()

RESOURCE_DIRECTORY = 'resources'
LAMMPS_DIRECTORY = 'lammps'
VASP_DIRECTORY = 'vasp'


def run_once(calculation, **kwargs):
    id_file_name = '{}.id'.format(calculation.name)
    if not exists(id_file_name):

        # The file does not exist so we definitely have to run the calculation
        with open(id_file_name, 'w') as id_file_handle:
            id_file_handle.write(calculation.id)

            calculation.run(**kwargs)

        calculation.save()
        return calculation
    else:
        cls = calculation.__class__
        # Load the id from the file and load the calculation from the pickle file afterwards
        with open(id_file_name, 'r') as id_file_handle:
            calculation_id = id_file_handle.read()
        calc = cls.load(calculation_id)
        from qmmm.core.calculation import Status
        if calc.status == Status.Execution:
            calc.run()
            calc.save()
        return calc


def all_ready(calcs, run=True, raise_error=False, early_exit=False):
    from qmmm.core.calculation import Status
    result = True
    for i in range(len(calcs)):
        calc = calcs[i]
        if calc.status != Status.Ready:
            # Could be that it is finished already
            # Let's run it again and see if someting changed
            if run:
                ncalc = run_once(calc)
                calcs[i] = ncalc
                calc = ncalc
            else:
                result = False
            # It is still not finished
            if calc.status != Status.Ready:

                if calc.status in [Status.ExecutionFailed, Status.ProcessingFailed]:
                    # Ohhh nooo the calculation crahsed
                    if raise_error:
                        raise RuntimeError('Calculation "{}" crashed! Please have a look at the log files'.format(calc.name))
                # Nevertheless not all are finished
                result = False
                # But we do not want to break the loop because we want to check all of the again
                # But if we are told to do so we exit the loop => undefined behaviour
                if early_exit:
                    return result
    return result


def crashed(calcs):
    from qmmm.core.calculation import Status
    return [
        c for c in calcs if c.status in [Status.ExecutionFailed, Status.ProcessingFailed]
    ]

def restart(calcs, db=False, execute=True):
    from qmmm.core.calculation import Calculation
    logger = logging.getLogger('utils.restart()')
    for calculation in calcs:
        assert isinstance(calculation, Calculation)
        id_file_name = '{}.id'.format(calculation.name)
        pickle_file_name = '{}.pickle'.format(calculation.id)
        archive_file_name = '{}.pickle'.format(calculation.id)
        if exists(id_file_name):
            remove(id_file_name)
            logger.info('Removed id file "{}" for calculation "{}"'.format(id_file_name, calculation.id))
        if exists(archive_file_name):
            remove(archive_file_name)
            logger.info('Removed archive file "{}" for calculation "{}"'.format(archive_file_name, calculation.id))
        if exists(pickle_file_name):
            remove(pickle_file_name)
            logger.info('Removed pickle file "{}" for calculation "{}"'.format(pickle_file_name, calculation.id))
        if db:
            raise NotImplementedError

        logger.info('Resetting calculation "{}:{}"'.format(calculation.name, calculation.id))
        calculation.reset()
        if execute:
            calculation.run()


class working_directory(object):

    def __init__(self, name=None, prefix=None, delete=False):
        self._name = str(uuid4()) if not name else name
        self._delete = delete
        self._curr_dir = getcwd()
        if prefix:
            self._name = join(prefix, self._name)

    def __enter__(self):
        if not exists(self._name):
            mkdir(self._name)
        chdir(self._name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        chdir(self._curr_dir)
        if self._delete:
            rmtree(self._name)


def predicate_generator(test):

    def _predicate_wrapper(element):
        try:
            result = test(element)
        except:
            return False
        else:
            return result
    return _predicate_wrapper


class HashableSet(set):

    def __hash__(self):
        return sum([hash(item) for item in self])


class LoggerMixin(object):
    @property
    def logger(self):
        return logging.getLogger(self.fullname())

    @classmethod
    def fullname(cls):
        name = '.'.join([
            cls.__module__,
            cls.__name__
        ])
        return name

def ensure_iterable(object):
    return [object] if not is_iterable(object) else object


def is_primitive(thing):
    #Also a reference is primitve and can be stored in the attributes
    primitives = (
    int, float, bool, str, complex, np.bool_, np.int_, np.str_, np.float_, np.complex_, np.ndarray, np.int8, np.int16,
    np.int32, np.int64, np.float16, np.float32, np.float64)
    return isinstance(thing, primitives) or thing is None


def get_configuration_directory():
    try:
        config_directory = ENVIRONMENT['QMMM_CONFIG_DIR']
    except KeyError:
        config_directory = getcwd()
    return config_directory


def displacements(struct1, struct2):
    check_id = 'id' in struct1.site_properties and 'id' in struct2.site_properties
    result = {}
    for i, (site1, site2) in enumerate(zip(struct1.sites, struct2.sites)):
        if check_id:
            assert site1.properties['id'] == site2.properties['id']
            id_ = site1.properties['id']
        else:
            id_ = i + 1
        result[id_] = site1.coords - site2.coords
    return result


def flatten(l):
    return [item for sl in l for item in sl]

class StructureWrapper(Structure):

    def __init__(self, *args, **kwargs):
        super(StructureWrapper, self).__init__(*args, **kwargs)

    def as_dict(self, verbosity=1, fmt=None, **kwargs):
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
            'lattice': self.lattice.as_dict(),
            'site_properties': self.site_properties,
            'fcoords': self.frac_coords,
            'species': [s.species_string for s in self.sites]
        }

        return d

    @classmethod
    def from_structure(cls, structure):
        assert isinstance(structure, Structure)
        return StructureWrapper(structure.lattice,
                                [s.species_string for s in structure.sites],
                                structure.frac_coords,
                                site_properties=structure.site_properties)

    @staticmethod
    def structure_to_dict(structure):
        d = {"@module": structure.__class__.__module__,
             "@class": structure.__class__.__name__,
             'lattice': structure.lattice.as_dict(),
             'site_properties': structure.site_properties,
             'fcoords': structure.frac_coords,
             'species': [s.species_string for s in structure.sites]
             }

        return d

    @classmethod
    def from_dict(cls, d, fmt=None):
        lattice = Lattice.from_dict(d['lattice'])
        frac_coords = d['fcoords']
        species = d['species']
        site_properties = d['site_properties']
        return Structure(lattice, species, frac_coords, site_properties=site_properties)

    
def recursive_as_dict(obj):
    if isinstance(obj, type):
        # Could also be a type
        return {
            '@module': obj.__module__,
            '@class': obj.__name__,
            '__meta__' : 'builtin.type'
        }
    elif isinstance(obj, (list, tuple)):
        return [recursive_as_dict(it) for it in obj]
    elif isinstance(obj, dict):
        return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
    elif hasattr(obj, "as_dict"):
        return obj.as_dict()
    return obj

def process_decoded(d):
    if isinstance(d, dict):
        if "@module" in d and "@class" in d:
            modname = d["@module"]
            classname = d["@class"]
        else:
            modname = None
            classname = None
        if modname and '__meta__' in d and d['__meta__'] == 'builtin.type':
            # It is a type object
            mod = __import__(modname, globals(), locals(), [classname], 0)
            if hasattr(mod, classname):
                cls_ = getattr(mod, classname)
                return cls_
        else:
            return __ENCODER__.process_decoded(d)
        return {process_decoded(k): process_decoded(v)
                for k, v in d.items()}
    elif isinstance(d, list):
        return [process_decoded(x) for x in d]

    return d

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class StringStream(StringIO):

    """

    """

    def __init__(self, string=''):
        super(StringStream, self).__init__(initial_value=string)
        self._pos = 0
        self._remaining = 0
        self._length = 0



    def read(self, size=-1):
        while self._remaining < size:
            sleep(0.01)
        result = super(StringStream, self).read()
        # Increase position, from current position seek( ..., 1)
        result_length = len(result)
        # Increase position, and cosume
        self._pos += result_length
        self._remaining = self._length - self._pos
        self.seek(self._pos)
        return result

    def write(self, s):
        write_length = len(s)
        super(StringStream, self).write(s)
        # After write file is at the end
        # Seek from back and make it available
        self._length += write_length
        self._remaining = self._length - self._pos

    def readline(self, size=-1, block=True):
        if self.tell() != self._pos:

            self.seek(self._pos)
        result = super(StringStream, self).readline()
        result_length = len(result)

        self._pos += result_length
        self._remaining = self._length - self._pos
        # Seek new position
        self.seek(self._pos)
        #if block:
        #    while not result:
        #        result = super(StringStream, self).readline()
        #        sleep(0.025)
        return result


def intersect(lst1, lst2):
    return [value for value in lst1 if value in lst2]

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest().upper()


def indent(string):
    return '\n'.join(['\t' + line for line in string.split('\n')])

def remove_white(string):
    whitespace = [' ', '\t', '\n']
    mystr = str(string)
    for removal in whitespace:
        mystr = mystr.replace(removal, '')
    return mystr


def create_directory(directory):
    if not exists(directory):
        mkdir(directory)


def once(f):
    """Runs a function (successfully) only once.
    The running can be reset by setting the `has_run` attribute to False
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result

    wrapper.has_run = False
    return wrapper


def fullname(o):
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__


def is_iterable(o):
    try:
        iter(o)
    except TypeError:
        return False
    else:
        return not isinstance(o, str)


class Tee(object):
    """A special purpose, with limited applicability, tee-like thing.

    A subset of stuff read from, or written to, orig_fd,
    is also written to out_fd.
    It is used by the lammps calculator for creating file-logs of stuff
    read from, or written to, stdin and stdout, respectively.
    """

    def __init__(self, orig_fd, out_fd):
        self._orig_fd = orig_fd
        self._out_fd = out_fd
        self.name = orig_fd.name

    def write(self, data):
        self._orig_fd.write(data)
        self._out_fd.write(data)
        self.flush()

    def read(self, *args, **kwargs):
        data = self._orig_fd.read(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readline(self, *args, **kwargs):
        data = self._orig_fd.readline(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readlines(self, *args, **kwargs):
        data = self._orig_fd.readlines(*args, **kwargs)
        self._out_fd.write(''.join(data))
        return data

    def flush(self):
        self._orig_fd.flush()
        self._out_fd.flush()


class SingletonMetaClass(type):
    def __init__(cls, name, bases, dict):
        super(SingletonMetaClass, cls) \
            .__init__(name, bases, dict)
        original_new = cls.__new__

        def my_new(cls, *args, **kwds):
            if cls.instance == None:
                cls.instance = \
                    original_new(cls, *args, **kwds)
            return cls.instance

        cls.instance = None
        cls.__new__ = staticmethod(my_new)


def consecutive_groups(iterable, ordering=lambda x: x):

    for k, g in groupby(
        enumerate(iterable), key=lambda x: x[0] - ordering(x[1])
    ):
        yield map(itemgetter(1), g)