from abc import ABCMeta
from monty.json import MSONable
from qmmm.core.utils import fullname, is_iterable, recursive_as_dict, process_decoded, LoggerMixin
from qmmm.core.lammps.constraints import Constraint, IterableConstraint
from uuid import uuid4
import logging
from inspect import getmembers, isclass
import sys


SUBCLASS_REGISTRY = {}


def _make_registry_key(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)


class SubclassRegistry(ABCMeta):

    def __new__(mcls, name, bases, namespace, **kwargs):
        # Import super to create _abc_cache_registry
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        global SUBCLASS_REGISTRY
        registry_key = _make_registry_key(cls)
        if registry_key not in SUBCLASS_REGISTRY:
            SUBCLASS_REGISTRY[registry_key] = cls
        # Register Command styles
        return cls

    def __getattr__(cls, item):
        # Fake inner classes for CommandStyles
        # At first try to get it the conventional way
        try:
            return super(SubclassRegistry, cls).__getattribute__(item)
        except AttributeError:
            # CommandStyle Cannot have a styles function
            if issubclass(cls, Command):
                if hasattr(cls, 'styles'):
                    style_func = super(SubclassRegistry, cls).__getattribute__('styles')
                    for style_cls in style_func():
                        if item == style_cls.__name__:
                            return style_cls
                    # If we get here nothing was found raise again the Attribute error
                    raise
                else:
                    raise
            else:
                raise




def process_args(cls, args, preprocess=None):
    iterable_args = [c for c in cls.Args if isinstance(c, IterableConstraint)]
    if len(iterable_args) > 1:
        raise IOError('Cannot handle more than one IterableConstraint')
    has_iterable_arg = len(iterable_args) > 0

    # Rearrange arguemnts

    if has_iterable_arg:
        iterable_arg = iterable_args[0]
        # Extract a value for each non iterable argument
        non_iterable_args = [c for c in cls.Args if not isinstance(c, IterableConstraint)]
        non_iterable_values = [args.pop(0) for _ in non_iterable_args]
        # The rest of the arguments belongs to the iterable constraint, regoin it
        iterable_value = ' '.join(args)
        # Reconstruct "args's" value
        args = non_iterable_values + [iterable_value]

        # Rearrange arguments
        local_class_args = non_iterable_args + [iterable_arg]
    else:
        local_class_args = cls.Args
    # Else there is nothing to do

    # Allow user to intercept easily
    if preprocess:
        args = preprocess(args)

    class_args = [c.parse_string(v) for c, v in zip(local_class_args, args)]

    return class_args

def _validate_keyword(cls, kword, value):
    constraint = cls.Keywords[kword]
    if isinstance(constraint, Constraint):
        if not constraint.validate(value):
            raise ValueError('Cannot accept value {} for argument "{}"'.format(value, constraint.name))
        return value
    elif hasattr(constraint, '__len__'):
        expected_arguments = len(constraint)
        if not hasattr(value, '__len__'):
            raise ValueError('Keyword {} expects {} arguments'.format(kword, expected_arguments))
        elif len(value) != expected_arguments:
            raise ValueError(
                'Keyword {} expects {} arguments but {} were given'.format(kword, expected_arguments, len(value)))

        ok_vals = []
        for c, v in zip(constraint, value):
            if not c.validate(v):
                raise ValueError(
                    'Cannot accept value {} for argument "{}" of keyword "{}"'.format(v, c.name, kword))
            ok_vals.append(v)
        return tuple(ok_vals)


class CommandStyle(MSONable, LoggerMixin, metaclass=SubclassRegistry):

    Args = None
    Style = None

    def __init__(self, *args):
        cls = type(self)
        if self._has_arguments():
            required_args = len(cls.Args)
            if required_args > len(args):
                raise ValueError(
                    '{}.__init__() is expecting exactly {} arguments but {} were given'.format(fullname(self),
                                                                                               required_args,
                                                                                               len(args)))
            elif required_args < len(args):
                logging.getLogger(fullname(self)).warning(
                    '{}.__init__() is expecting exactly {} arguments but {} were given'.format(fullname(self),
                                                                                               required_args,
                                                                                               len(args)))

            for i, constraint in enumerate(cls.Args):
                value = args[i]
                # Check if the input makes sense
                if isinstance(value, Command):
                    value = value.identifier if constraint.expand else value
                if not constraint.validate(value):
                    raise ValueError(
                        '{}: Cannot accept value {} for argument "{}"'.format(fullname(self), value, constraint.name))
                attribute_name = '_{}'.format(constraint.name)
                # Create an attribute
                setattr(self, attribute_name, value)


    def __getattr__(self, item):
        # Create fallback item was not found look it up in args
        cls = type(self)
        if self._has_arguments():
            if item in [a.name for a in cls.Args]:
                item = '_{}'.format(item)

        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        cls = type(self)
        if self._has_arguments():
            for constraint in cls.Args:
                if key == constraint.name:
                    # Resolve ids automatically
                    if isinstance(value, Command):
                        value = value.identifier if constraint.expand else value
                    if not constraint.validate(value):
                        raise ValueError('Cannot accept value {} for argument "{}"'.format(value, constraint.name))
                    else:
                        key = '_{}'.format(key)
                        break
        # Check if self has this key
        object.__setattr__(self, key, value)


    @classmethod
    def _has_arguments(cls):
        if hasattr(cls, 'Args'):
            if cls.Args is not None and len(cls.Args) > 0:
                return True
            else:
                return False
        else:
            return False

    def as_dict(self):
        cls = type(self)
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        if cls._has_arguments():
            for constraint in cls.Args:
                name = '_{}'.format(constraint.name)
                val = getattr(self, name)
                val = recursive_as_dict(val)
                d[constraint.name] = val
        return d

    @classmethod
    def from_dict(cls, d):
        decoded = {k: process_decoded(v) for k, v in d.items()
                   if not k.startswith("@")}
        if cls._has_arguments():
            args = (decoded[argument.name] for argument in cls.Args)
        else:
            args = ()

        return cls(*args)


    @classmethod
    def from_string(cls, string, preprocess=None, return_remaining=False, make_obj=True):
        assert isinstance(string, str)
        crumbs = string.split(' ')
        crumbs = [c for c in crumbs if c.lstrip().rstrip() != '']
        style = crumbs.pop(0)
        if cls._has_arguments():
            # If there is an iterable argument consume everything
            has_iterable_argument = any([isinstance(argument, IterableConstraint) for argument in cls.Args])
            if has_iterable_argument:
                args = crumbs.copy()
                crumbs = []
            else:
                # Consume just one thing for each argument
                args = [crumbs.pop(0) for _ in cls.Args]
        else:
            args = []
        if style != cls.Style:
            raise ValueError('Invalid style {} for CommandStyle {}'.format(style, _make_registry_key(cls)))

        style_args = process_args(cls, args, preprocess=preprocess) if cls._has_arguments() else ()
        if make_obj:
            obj = cls(*style_args)
        else:
            obj = style_args
        return (obj, ' '.join(crumbs)) if return_remaining else obj



    def format_arguments(self):
        cls = type(self)
        return ' '.join([
            argument.format_argument(
                getattr(self, '_{}'.format(argument.name))
            ) for argument in cls.Args]) if self._has_arguments() else ''

    def format(self):
        return '{} {}'.format(self.Style, self.format_arguments())

    def __str__(self):
        return self.format()


class KeywordCommandStyle(CommandStyle):

    Keywords = None

    def __init__(self,*args, **keywords):
        cls = type(self)
        if self._has_arguments():
            pass
            # Remove this clause
            #raise TypeError('A KeywordCommandStyle must not have any regular arguments')
        self._keywords = None

        if self._has_keywords():
            self._keywords = {}
            for key, value in keywords.items():
                if key not in cls.Keywords:
                    raise KeyError(key)

                self._keywords[key] = _validate_keyword(cls, key, value)
        # Initialize with no arguments
        super(KeywordCommandStyle, self).__init__(*args)

    def __getattr__(self, item):
        # Create fallback item was not found look it up in args
        cls = type(self)
        if self._has_keywords():
            if item in cls.Keywords:
                if item in self._keywords:
                    return self._keywords[item]
                else:
                    raise KeyError('Attribute "{}" was not set to this instance'.format(item))

        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        cls = type(self)
        if self._has_keywords():
            if key in cls.Keywords:
                if isinstance(value, Command):
                    value = value.identifier if cls.Keywords[key].expand else value
                self._keywords[key] = _validate_keyword(cls, key, value)
                return
        # Check if self has this key
        object.__setattr__(self, key, value)

    @classmethod
    def from_string(cls, string, preprocess=None, return_remaining=False, make_obj=True):
        #Create super object

        assert isinstance(string, str)
        args, string = super(KeywordCommandStyle).from_string(string, preprocess=preprocess, return_remaining=True, make_obj=False)
        # Append style at the beginning of the string
        string = '{} {}'.format(cls.Style, string)
        crumbs = string.split(' ')
        crumbs = [c for c in crumbs if c.lstrip().rstrip() != '']
        style = crumbs.pop(0)
        if cls._has_keywords():
            # consume until no keywords are available
            # Now after the style I expect a keyword
            keywords = {}
            while len(crumbs) > 0:
                kword = crumbs.pop(0)
                if kword in cls.Keywords:
                    constraints = cls.Keywords[kword]
                    if not is_iterable(constraints):
                        constraints = [constraints]
                    # Consume all arguments
                    try:
                        kvalues = [c.parse_string(crumbs.pop(0)) for c in constraints]
                    except IndexError:
                        raise
                    keywords[kword] = kvalues if len(constraints) > 1 else kvalues[0]
        else:
            keywords = {}

        if style != cls.Style:
            raise ValueError('Invalid style {} for CommandStyle {}'.format(style, _make_registry_key(cls)))
        if make_obj:
            obj = cls(*args, **keywords)
        else:
            obj = keywords
        return (obj, ' '.join(crumbs)) if return_remaining else obj

    def as_dict(self):
        d = super(KeywordCommandStyle, self).as_dict()

        if self._has_keywords():
            d['keywords'] = recursive_as_dict(self._keywords)

        return d

    @classmethod
    def from_dict(cls, d):
        decoded = {k: process_decoded(v) for k, v in d.items()
                   if not k.startswith("@")}
        if 'keywords' in decoded:
            keywords = decoded['keywords']
        else:
            keywords = {}
        if cls._has_arguments():
            args = (decoded[argument.name] for argument in cls.Args)
        else:
            args = ()
        return cls(*args, **keywords)

    def format_keywords(self):
        cls = type(self)
        if self._has_keywords():
            args = []
            for kword, value in self._keywords.items():
                constraints = cls.Keywords[kword]
                if not is_iterable(value):
                    value = [value]
                    constraints = [constraints]
                args.append('{} {}'.format(kword,
                                           ' '.join([c.format_argument(v) for c,v in zip(constraints, value)])
                                           ))
            return ' '.join(args)
        else:
            return ''

    def format(self):
        arg_str = ' ' + self.format_arguments() if self._has_arguments() else ''
        kw_str = ' ' + self.format_keywords() if self._has_keywords() else ''
        return self.Style + arg_str + kw_str

    def __format__(self, format_spec):
        return self.format()


    @classmethod
    def _has_keywords(cls):
        if hasattr(cls, 'Keywords'):
            if cls.Keywords is not None and len(cls.Keywords) > 0:
                return True
            else:
                return False
        else:
            return False


class Command(MSONable, LoggerMixin, metaclass=SubclassRegistry):
    Command = None
    Args = None
    Keywords = None
    __Instances = {}
    __Sequence = []
    # {command} args style {style_args} {keywords}

    def __init__(self, *args, **kwargs):
        self._id = str(uuid4())
        # self._dumps = {}
        self._keywords = {}
        cls = type(self)
        # Process arguments
        self._style = None
        # Everything ok enough arguments were given
        args = list(args)
        if self._has_arguments():
            args = list(args)


            for i, constraint in enumerate(cls.Args):
                value = args.pop(0)
                if isinstance(value, Command):
                    value = value.identifier if constraint.expand else value
                # Check if the input makes sense
                if not constraint.validate(value):
                    raise ValueError(
                        '{}: Cannot accept value "{}" for argument "{}"'.format(fullname(self), value, constraint.name))
                attribute_name = '_{}'.format(constraint.name)
                # Create an attribute
                setattr(self, attribute_name, value)
            if len(args) == 0:
                self.logger.debug('No command style detected for command "{}"'.format(cls.Command))
                # No style was given
                self._style = None
            else:
                #Now I expect a style arguemnt and consume it
                style = args.pop(0)
                if isinstance(style , CommandStyle):
                    # It is a style object, take it
                    self._style = style
                elif isinstance(style, type):
                    style_cls = style
                    # Its just an type object, instantiate it
                    if issubclass(style_cls, KeywordCommandStyle):
                        style_kwargs = {}
                        # Filter keywords
                        if style_cls._has_keywords():
                            for key, value in kwargs.items():
                                if key in style_cls.Keywords:
                                    # Add it to style_kwargs but remove it from kwargs
                                    style_kwargs[key] = value
                            # Remove it from kwargs
                            for key in style_kwargs.keys():
                                del kwargs[key]
                        self._style = style_cls(*args, **style_kwargs)
                    else:
                        self._style = style_cls(*args)
                else:
                    raise TypeError

        else:
            # First argument is style argument , take it
            if len(args) > 0: # Some args are left
                style_cls = args.pop(0)
                if isinstance(style_cls, type):
                    # Its just an type object, instantiate it
                    if issubclass(style_cls, KeywordCommandStyle):
                        style_kwargs = {}
                        # Filter keywords
                        if style_cls._has_keywords():
                            for key, value in kwargs.items():
                                if key in style_cls.Keywords:
                                    # Add it to style_kwargs but remove it from kwargs
                                    style_kwargs[key] = value
                            # Remove it from kwargs
                            for key in style_kwargs.keys():
                                del kwargs[key]
                        self._style = style_cls(**style_kwargs)
                    else:
                        self._style = style_cls(*args)
                elif isinstance(style_cls, CommandStyle):
                    # it is an constructed instance take it as it is
                    self._style = style_cls
            else:
                self._style = None
            #
        # Process keyword args
        if self._has_keywords():
            for key, value in kwargs.items():
                if key in cls.Keywords:
                    self._keywords[key] = _validate_keyword(cls, key, value)
        if self.identifier not in Command.__Instances:
            cls.__Instances[self.identifier] = [self]
        else:
            cls.__Instances[self.identifier].append(self)
        cls.__Sequence.append(self)


    @classmethod
    def styles(cls):
        current_module = sys.modules[cls.__module__]
        styles = []
        for name, obj in getmembers(current_module, predicate=isclass):
            # Filter for abstract CommandStyles
            if issubclass(obj, CommandStyle):
                if obj.Style is not None:
                    styles.append(obj)
        return tuple(styles)

    @staticmethod
    def exists(id):
        return id in Command.__Instances



    def __getattr__(self, item):
        # Create fallback item was not found look it up in args
        cls = type(self)
        if self._has_keywords():
            if item in cls.Keywords:
                return self._keywords[item]
        if self._has_arguments():
            if item in [a.name for a in cls.Args]:
                item = '_{}'.format(item)

        return object.__getattribute__(self, item)

    @classmethod
    def _has_keywords(cls):
        if hasattr(cls, 'Keywords'):
            if cls.Keywords is not None and len(cls.Keywords) > 0:
                return True
            else:
                return False
        else:
            return False

    @classmethod
    def _has_arguments(cls):
        if hasattr(cls, 'Args'):
            if cls.Args is not None and len(cls.Args) > 0:
                return True
            else:
                return False
        else:
            return False


    def __setattr__(self, key, value):
        cls = type(self)
        if self._has_keywords():
            if key in cls.Keywords:
                self._keywords[key] = _validate_keyword(cls, key, value)
                # Terminate
                return
        if self._has_arguments():
            for constraint in cls.Args:
                if key == constraint.name:
                    # Resolve ids automatically
                    if isinstance(value, Command):
                        value = value.identifier if constraint.expand else value
                    if not constraint.validate(value):
                        raise ValueError('Cannot accept value {} for argument "{}"'.format(value, constraint.name))

                    else:
                        key = '_{}'.format(key)
                        break

        object.__setattr__(self, key, value)

    @property
    def id(self):
        return self._id

    def as_dict(self):
        cls = type(self)
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        d['id'] = self.id
        # d['dumps'] = self._dumps
        if cls._has_keywords():
            d['keywords'] = self._keywords
        if self._style is not None:
            d['style'] = self._style.as_dict()
        if cls._has_arguments():
            for constraint in cls.Args:
                name = '_{}'.format(constraint.name)
                val = getattr(self, name)
                d[constraint.name] = recursive_as_dict(val)
        return d

    @classmethod
    def resolve(cls, id):
        if id not in cls.__Instances:
            raise KeyError(id)
        return cls.__Instances[id]

    @classmethod
    def from_dict(cls, d):
        decoded = {k: process_decoded(v) for k, v in d.items()
                   if not k.startswith("@")}
        if not all([c in decoded for c in ('id',)]):
            raise KeyError('Missing keys')

        id = decoded['id']
        del decoded['id']
        if 'keywords' in decoded:
            keywords = decoded['keywords']
            del decoded['keywords']
        else:
            keywords = {}

        if 'style' in decoded:
            style = decoded['style']
            del decoded['style']
        else:
            style = None
        if cls._has_arguments():
            args = (decoded[argument.name] for argument in cls.Args)
        else:
            args = ()
        if style is not None:
            args = tuple(list(args) + [style])

        object_ = cls(*args, **keywords)
        # Override id
        object_._id = id
        return object_

    def format_arguments(self):
        cls = type(self)
        return ' '.join([
            argument.format_argument(
                getattr(self, '_{}'.format(argument.name))
            ) for argument in cls.Args]) if self._has_arguments() else ''

    def format_keywords(self):
        return ' '.join([self.format_keyword(kword) for kword in self._keywords.keys()]) if self._has_keywords() else ''

    def format_keyword(self, kword):
        cls = type(self)
        constraint = cls.Keywords[kword]
        if is_iterable(constraint):
            return '{} {}'.format(kword, ' '.join(
                [c.format_argument(val) for c, val in zip(constraint, self._keywords[kword])]))
        else:
            return '{} {}'.format(kword, constraint.format_argument(self._keywords[kword]))

    def format(self):
        cls = type(self)
        chunks = [cls.Command]
        if cls._has_arguments():
            chunks.append(self.format_arguments())
        if self._style:
            chunks.append(self._style.format())
        if cls._has_keywords():
            chunks.append(self.format_keywords())

        return ' '.join(chunks) + '\n'

    def __str__(self):
        return self.format()

    def __format__(self, format_spec):
        return self.format()

    @property
    def identifier(self):
        return self._id

    @property
    def style(self):
        return self._style

    @staticmethod
    def sequence(print=True):

        if print:
            for command in Command.__Sequence:
                sys.stdout.write(command)
        return Command.__Sequence

    @staticmethod
    def parse(string):
        for line in string.split('\n'):
            if line.startswith('#'):
                continue
            crumbs = line.split(' ')
            # Filter crumbs for empty strings
            crumbs = [c for c in crumbs if c.lstrip().rstrip() != '']
            # Check if it is an empty line
            if len(crumbs) < 1:
                continue
            command = crumbs[0]
            # Search for all commands
            possible_commands = []
            for registry_key, cls_ in SUBCLASS_REGISTRY.items():
                if issubclass(cls_, Command):
                    if cls_.Command == command:
                        possible_commands.append((registry_key, cls_))
            if len(possible_commands) < 1:
                raise ValueError('No command class defined for "{}"'.format(command))
            elif len(possible_commands) > 1:
                logging.getLogger(Command.__module__ + '.' + Command.Command).warning('More than one command defined for word "{}"'.format(command))

            _, cls_ = possible_commands[0]
            cls_.from_string(line)

    @classmethod
    def from_string(cls, string):
        assert isinstance(string, str)
        args = string.split(' ')
        args = [c for c in args if c.lstrip().rstrip() != '']
        command = args.pop(0)

        if command != cls.Command:
            raise ValueError('Invalid command argument "{}"'.format(command))
        if cls._has_arguments():
            if len(args) < len(cls.Args):
                raise ValueError('Some arguments are missing')
        command_args = []

        allowed_styles = {s.Style: s for s in cls.styles()}

        # Read until a style keyword appear
        if cls._has_arguments():
            while len(args) > 0:
                a = args.pop(0)
                if a not in allowed_styles:
                    # Pop it from the arguments
                    command_args.append(a)
                else:
                    # Push it in again, one was too much
                    args.insert(0, a)
                    break

            command_args = process_args(cls, command_args)
        if len(allowed_styles) == 0:
            # No styles are available
            return cls(*command_args)
        style = args[0] # Do not consume because it's needed for style.from_string
        if style not in allowed_styles:
            raise NotImplementedError('Style type {} not implemented'.format(style))

        # Read until a keyword appears
        allowed_keywords = list(cls.Keywords) if cls._has_keywords() else []
        # Consume
        style_crumbs = []
        while len(args) > 0:
            a = args.pop(0)
            if a not in allowed_keywords:
                style_crumbs.append(a)
            else:
                args.insert(0, a)
                break

        style_type = allowed_styles[style]
        style_object, crumbs_left = style_type.from_string(' '.join(style_crumbs), return_remaining=True)
        if not style_object:
            raise ValueError('Failed parsing command style')

        # Up to now we consumed everything except for the keywords
        # Now one try to pare the keywords
        args  = [c for c in crumbs_left.split(' ') if c.rstrip().lstrip() != '']
        kwords = {}
        if cls._has_keywords():
            while len(args) > 0:
                a = args.pop(0)
                if a in allowed_keywords:
                    kword = a
                    constraints = cls.Keywords[kword]
                    if not is_iterable(constraints):
                        constraints = [constraints]
                    # Consume all arguments
                    try:
                        kvalues = [c.parse_string(args.pop(0)) for c in constraints]
                    except IndexError:
                        raise
                    kwords[kword] = kvalues if len(constraints) > 1 else kvalues[0]


        return cls(*command_args, style_object, **kwords)

    @staticmethod
    def clear():
        Command.__Sequence = []

    @staticmethod
    def sequence_dict():
        return [command.as_dict() for command in Command.__Sequence]

    @staticmethod
    def from_sequence_dict(sequence_dict):
        Command.clear()
        for command_dict in sequence_dict:
            process_decoded(command_dict)





