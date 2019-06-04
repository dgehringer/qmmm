__author__ = 'dominik'

from qmmm.core.utils import get_configuration_directory
from configparser import ConfigParser
from os.path import join
from configparser import NoOptionError

CONFIGURATION_FILE = join(get_configuration_directory(), 'settings')


class SingletonMetaClass(type):
    def __init__(cls, name, bases, dict):
        super(SingletonMetaClass, cls).__init__(name, bases, dict)
        original_new = cls.__new__

        def my_new(cls, *args, **kwds):
            if cls.instance == None:
                cls.instance = \
                    original_new(cls, *args, **kwds)
            return cls.instance
        cls.instance = None
        cls.__new__ = staticmethod(my_new)


def convert_string_to_path(path_string):
    path = path_string.split('/')
    path = list(filter(lambda path_crumb: path_crumb != '', path))
    return path


class Configuration(object):
    __metaclass__ = SingletonMetaClass
    __config = ConfigParser()

    def __init__(self):
        super(Configuration, self).__init__()
        self.load()

    def save(self, file_name=CONFIGURATION_FILE):
        self.__config.write(file_name)

    def load(self, file_name=CONFIGURATION_FILE):
        self.__config.read(file_name)

    def get_option(self, section, option):
        return self.__config.get(section, option)

    def get_options(self, section):
        return self.__config.options(section)

    def __check_path(self, item):
        if isinstance(item, list):
            if len(item) != 2:
                raise ValueError('The identifier for the configuration option must have length 2! e.g '
                                 '["section", "option"]')
            else:
                section = item[0]
                option = item[1]
                return section, option
        elif isinstance(item, str):
            path = convert_string_to_path(item)
            if len(path) != 2:
                raise ValueError('The path for the configuration option must have depth 2! e.g /section/option')
            else:
                section = path[0]
                option = path[1]
                return section, option
        else:
            raise TypeError('The identifier for the configuration option must either be a path of type str or'
                            'a list of length 2')

    def __getitem__(self, item):
        section, option = self.__check_path(item)
        return self.get_option(section, option)

    def __setitem__(self, key, value):
        section, option = self.__check_path(key)
        if not self.__config.has_section(section):
            self.__config.add_section(section)
        self.__config.set(section, option, value)
        self.save()


def get_setting(setting, section='general'):
    from os import environ as ENVIRONMENT
    environment_variable_name = str(setting).upper()
    if environment_variable_name in ENVIRONMENT:
        # An environment variable is defined
        return ENVIRONMENT[environment_variable_name]
    else:
        # Look if a setting is available, but with lower name
        setting_name = environment_variable_name.lower()
        try:
            return Configuration()[join(section, setting_name)]
        except NoOptionError:
            return None
