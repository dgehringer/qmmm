
from qmmm.core.utils import is_iterable, indent

class Constraint:

    def __init__(self, name, help=None, expand=True):
        self._name = name
        self._help = help
        self.expand = expand

    @property
    def name(self):
        return self._name

    @property
    def help(self):
        return self._help

    def parse_string(self, string):
        return string

    def validate(self, val):
        raise NotImplementedError

    def format_argument(self, value):
        return str(value)

    def format_help(self):
        if callable(self._help):
            help = self._help()
        else:
            help = self._help
        return '{} = {}'.format(self.name, help)

class ChoiceConstraint(Constraint):

    def __init__(self, name, choices, help=None):
        super(ChoiceConstraint, self).__init__(name, help=help)
        self._choices = choices

    def validate(self, val):
        return val in self._choices

    @property
    def choices(self):
        return self._choices

    def format_help(self):
        if callable(self.help):
            return '{} = {}'.format(self.name, self.help())
        elif is_iterable(self.help) and not isinstance(self.help, str):
            assert len(self.help) == len(self.choices)
            first_line = '{} = {}\n'.format(self.name, ' or '.join([str(c) for c in self.choices]))
            more = indent(''.join(['{} = {}\n'.format(c, ch) for c, ch in zip(self.choices, self.help)]))
            return first_line + more + '\n'
        else:
            # only one help string is available
            return '{} = {} = {}\n'.format(self.name, ' or '.join([str(c) for c in self.choices]), self.help)


class PrimitiveConstraint(Constraint):

    def __init__(self, name, dtype, help=None):
        super(PrimitiveConstraint, self).__init__(name, help=help)
        self._dtype = dtype

    def validate(self, val):
        return isinstance(val, self._dtype)

    @property
    def dtype(self):
        return self._dtype

    def parse_string(self, string):
        for data_type in self._dtype if is_iterable(self._dtype) else [self.dtype]:
            try:
                result = data_type(string)
            except:
                continue
            else:
                return result
        raise ValueError(string)

class IterableConstraint(Constraint):

    def __init__(self, name, element_validator=None, help=None):
        super(IterableConstraint, self).__init__(name, help=help)
        self._element_validator = element_validator

    def validate(self, val):
        is_iter = is_iterable(val)
        if is_iter:
            if callable(self._element_validator):
                correct_elements = all([self._element_validator(v) for v in val])
            else:
                correct_elements = True
        else:
            correct_elements =  False
        return is_iter and correct_elements

    def format_argument(self, value):
        if not is_iterable(value):
            raise TypeError('{} is not iterable'.format(type(value).__name__))
        return ' '.join([str(v) for v in value])

    def parse_string(self, string):
        return string.split(' ')

class CustomConstraint(Constraint):

    def __init__(self, name, validator=lambda el: True, formatter=lambda el: str(el), help=None):
        super(CustomConstraint, self).__init__(name, help=help)
        self._validator = validator
        self._formatter = formatter

    def validate(self, val):
        return self._validator(val)

    def format_argument(self, value):
        return self._formatter(value)

class ReferenceConstraint(CustomConstraint):

    def __init__(self, name, cls, formatter=lambda el: str(el), help=None):
        super(ReferenceConstraint, self).__init__(name, help=help)
        def _validate(el):
            if isinstance(el, cls):
                # It is already an instance
                return True
            else:
                return cls.exists(el)
        self._validator = _validate
        self._formatter = formatter

    def validate(self, val):
        return self._validator(val)

    def format_argument(self, value):
        return self._formatter(value)
