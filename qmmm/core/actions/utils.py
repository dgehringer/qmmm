from enum import Enum
from qmmm.core.utils import LoggerMixin, ensure_iterable
from types import FunctionType, MethodType
from inspect import getargspec

def requires_arguments(func):
    args, varargs, varkw, defaults = getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    return len(args) > 0


def is_iterable(o):
    """
    Convenience method to test for an iterator

    Args:
        o: the object to test

    Returns:
        bool: wether the input argument is iterable or not
    """
    try:
        iter(o)
    except TypeError:
        return False
    else:
        return not isinstance(o, str)


# convenience function to ensure the passed argument is iterable
ensure_iterable = lambda v: v if is_iterable(v) else [v]


class CombinedInput(list):
    """
    Dummy class to detect if an action gets input from multiple other actions
    """
    pass


class CrumbType(Enum):
    """
    Enum class. Provides the types which Crumbs in an IODictionary are allowed to have
    """

    Root = 0
    Attribute = 1
    Item = 2


class Crumb(LoggerMixin):
    """
    Represents a piece in the path of the IODictionary. The Crumbs are used to resolve a recipe path correctly
    """

    def __init__(self, crumb_type, name):
        """
        Initializer from crumb
        Args:
            crumb_type (CrumbType): the crumb type of the object
            name (str, object): An object if crumb type is CrumbType.Root otherwise a string
        """
        self._crumb_type = crumb_type
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def crumb_type(self):
        return self._crumb_type

    @property
    def object(self):
        if self.crumb_type != CrumbType.Root:
            raise ValueError('Only root crumbs store an object')
        return self._name

    @staticmethod
    def attribute(name):
        """
        Convenience method to produce an attribute crumb

        Args:
            name (str): the name of the attribute

        Returns:
            Crumb: the attribute crumbs

        """
        return Crumb(CrumbType.Attribute, name)

    @staticmethod
    def item(name):
        """
        Convenience method to produce an item crumb

        Args:
            name (str): the name of the item

        Returns:
            Crumb: the item crumb
        """
        return Crumb(CrumbType.Item, name)

    @staticmethod
    def root(obj):
        """
        Convenience method to produce a root crumb

        Args:
            obj: The root object of the path in the IODictionary

        Returns:
            Crumb: A root crumb, meant to be the first item in a IODictionary path
        """
        return Crumb(CrumbType.Root, obj)

    def __repr__(self):
        return '<{}({}, {})'.format(self.__class__.__name__,
                                    self.crumb_type.name,
                                    self.name if isinstance(self.name, str) else self.name.__class__.__name__)


    def __hash__(self):
        crumb_hash = hash(self._crumb_type)
        if self._crumb_type == CrumbType.Root:
            crumb_hash += hash(hex(id(self.object)))
        else:
            try:
                crumb_hash += hash(self._name)
            except Exception as e:
                self.logger.exception('Failed to hash "{}" object'.format(type(self._name).__name__), exc_info=e)
        return crumb_hash

    def __eq__(self, other):
        if not isinstance(other, Crumb):
            return False
        if self.crumb_type == other.crumb_type:
            if self.crumb_type == CrumbType.Root:
                return hex(id(self.object)) == hex(id(other.object))
            else:
                return self.name == other.name
        else:
            return False



class Path(list, LoggerMixin):

    def append(self, item):
        if not isinstance(item, Crumb):
            raise TypeError('A path can only consist of crumbs')
        else:
            super(Path, self).append(item)

    def extend(self, collection):
        if not all([isinstance(item, Crumb) for item in collection]):
            raise TypeError('A path can only consist of crumbs')
        else:
            super(Path, self).extend(collection)

    def index(self, item, **kwargs):
        if not isinstance(item, Crumb):
            raise TypeError('A path can only consist of crumbs')
        else:
            return super(Path, self).index(item, **kwargs)

    def count(self, item):
        if not isinstance(item, Crumb):
            raise TypeError('A path can only consist of crumbs')
        else:
            return super(Path, self).count(item)

    @classmethod
    def join(cls, *p):
        return Path(p)


class Pointer(LoggerMixin):

    def __init__(self, root):
        if root is not None:
            if not isinstance(root, Path):
                if not isinstance(root, Crumb):
                    path = [Crumb.root(root)]
                else:
                    path = [root]
            else:
                path = root.copy()
        else:
            raise ValueError('Root object can never be "None"')
        self.__path = path

    def __getattr__(self, item):
        return Pointer(Path.join(*self.__path, Crumb.attribute(item)))

    def __getitem__(self, item):
        return Pointer(Path.join(*self.__path, Crumb.item(item)))

    @property
    def path(self):
        return self.__path

    def _resolve_path(self):
        """
        This method resolves the object hiding behind a path (A list of Crumbs)

        Args:
            path (list<Crumb>): A list of path crumbs used to resolve the data object
            remaining (int): How many crumbs should be resolved

        Returns:
            The underlying data object, hiding behind path parameter

        """
        # Make a copy to ensure, because by passing by reference
        path = self.path.copy()

        # Have a look at the path and check that it starts with a root crumb
        root = path.pop(0)
        if root.crumb_type != CrumbType.Root:
            raise ValueError('Got invalid path. A valid path starts with a root object')
        # First element is always an object
        result = root.object
        while len(path) > 0:
            # Take one step in the path, pop the next crumb from the list
            crumb = path.pop(0)
            crumb_type = crumb.crumb_type
            crumb_name = crumb.name

            # Resolve it with the correct method - dig deeper
            if crumb_type == CrumbType.Attribute:
                try:
                    result = getattr(result, crumb_name)
                except AttributeError:
                    raise
            elif crumb_type == CrumbType.Item:
                try:
                    result = result.__getitem__(crumb_name)
                except TypeError:
                    raise
                except KeyError:
                    raise
            # Get out of resolve mode
        return result

    def _resolve_function(self, function):
        """
        Convenience function to make IODictionary.resolve more readable. If the value is a function or a CombinedInput
        it calls the resolved functions if the do not require arguments

        Args:
            key (str): the key the value belongs to, just for logging purposes
            value (object, function or CombinedInput): the object to resolve

        Returns:
            (object or CombinedInput): The return value of the functions, if no functions were passed "value" is returned
        """
        result = function
        if isinstance(function, (FunctionType, MethodType)):
            # Make sure that the function has not parameters or all parameters start with
            if not requires_arguments(function):
                try:
                    # Get the return value
                    result = function()
                except Exception as e:
                    self.logger.exception('Failed to execute callable to resolve values for '
                                          'path {}'.format(self), exc_info=e)
                else:
                    self.logger.debug('Successfully resolved callable for path {}'.format(self))
            else:
                self.logger.warning('Found function, but it takes arguments! I \'ll not resolve it.')
        return result

    def __invert__(self):
        return self.resolve()

    def resolve(self):
        return self._resolve_function(self._resolve_path())


class IODictionary(dict, LoggerMixin):

    """
    A dictionary class representing the input parameters of a Command class. The dictionary holds a path which is recipe
    that can be resolved at runtime to obtain underlying values. A dictionary instance can hold multiple instances of
    IODictionary as value items which can be resolved into the real values when desired.
    """
    def __init__(self, **kwargs):
        super(IODictionary, self).__init__(**kwargs)

    def __getattr__(self, item):
        if item == 'initial':
            super(IODictionary, self).__getattribute__(item)
        return self.__getitem__(item)

    def __getitem__(self, item):
        if item in self.keys():
            value = super(IODictionary, self).__getitem__(item)
            if isinstance(value, Pointer):
                # Try to resolve the pointer
                try:
                    resolved = ~value
                except (KeyError, AttributeError, TypeError) as e:
                    #self.logger.exception('Failed to resolve item "{}". I\'ll search in self.initial'.format(item), exc_info=e)
                    return self._search_initial(item)
                else:
                    return resolved

            elif isinstance(value, (list, set, tuple)):
                cls = type(value)
                return cls([element if not isinstance(element, Pointer) else ~element for element in value])
            else:
                return value
        return super(IODictionary, self).__getitem__(item)

    def __setattr__(self, key, value):
        if key == 'initial':
            super(IODictionary, self).__setattr__(key,value)
        super(IODictionary, self).__setitem__(key, value)

    def _search_initial(self, item):
        if 'initial' not in self.keys():
            self.logger.info('No intial dictionary found')
            raise KeyError(item)
        else:
            initial_values = self.initial
            if item not in initial_values:
                raise KeyError
            else:
                return initial_values[item]