import json
from qmmm.core.utils import LoggerMixin, flatten, get_configuration_directory
from os.path import join

REMOTE_CONFIG_FILE = join(get_configuration_directory(), 'remote.json')
APPLICATION_CONFIG_FILE = join(get_configuration_directory(), 'application.json')

try:
    with open(REMOTE_CONFIG_FILE) as remote_config:
        REMOTE_CONFIG = json.load(remote_config)
except:
    REMOTE_CONFIG = {}

try:
    with open(APPLICATION_CONFIG_FILE) as application_config:
        APPLICATION_CONFIG = json.load(application_config)
except:
    APPLICATION_CONFIG = {}


class ConfigBuilder(dict, LoggerMixin):

    def __init__(self, **kwargs):
        super(ConfigBuilder).__init__(**kwargs)
        self._resource = 'local'
        self._resource_kwargs = {}
        self._partition = 'default'
        self._partition_kwargs = {}
        self._application = None
        self._manager = False

    @staticmethod
    def applications():
        global APPLICATION_CONFIG
        return list(APPLICATION_CONFIG.keys())

    @staticmethod
    def remotes():
        global REMOTE_CONFIG
        return list(REMOTE_CONFIG.keys())

    def _partition_aliases(self):
        result = flatten([p_conf['id']
                          for p_conf in REMOTE_CONFIG[self._resource]['partitions']
                          if p_conf['name'] == self._partition])
        # local always must have a default
        #if 'default' not in result:
        #    result.append('default')
        return result

    def application(self, application):
        if application is None:
            return self
        global APPLICATION_CONFIG
        if application not in APPLICATION_CONFIG:
            raise KeyError('"{}" is not configured'.format(application))
        app_config = APPLICATION_CONFIG[application]
        if self._resource not in app_config:
            raise KeyError('Resource "{}" is not configured for application "{}"'.format(self._resource, application))
        app_config = app_config[self._resource]
        partition_aliases = self._partition_aliases()
        if self._resource == 'local' and 'default' not in partition_aliases:
            # Local resource should have default partition configured
            partition_aliases.append('default')
        if not any([ p_al in app_config for p_al in partition_aliases]):
            raise KeyError('Partition "{}" is not configured for resource "{}" for application "{}"'.format(self._partition, self._resource, application))
        # Find the right partition alias key
        partition_alias = [pal for pal in partition_aliases if pal in app_config][0]
        app_config = app_config[partition_alias]
        self['preamble'] = app_config['preamble']
        format_kwargs = {'binary': app_config['binary']}
        if self._manager:
            format_kwargs.update(**self['manager'])
        command = app_config['command'].format(**format_kwargs)
        self['command'] = command
        self._application = application
        return self

    def remote(self, resource, **kwargs):
        global REMOTE_CONFIG
        if resource not in REMOTE_CONFIG:
            raise KeyError('"{}" resource is not configured'.format(resource))
        if resource == 'local':
            # Do nothing here but ensure that it is in remote.json config file
            return self
        remote_config = REMOTE_CONFIG[resource]
        self['remote'] = {}
        for k in ['user', 'host', 'identity_file', 'prefix']:
             self['remote'][k] = remote_config[k]

        # Allow the user to override settings
        self['remote'].update(**kwargs)
        self._resource = resource
        self._resource_kwargs = kwargs
        self.application(self._application)
        return self

    def __getattr__(self, item):
        # Fallback method try to get an item by a key
        if item in self:
            return self[item]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def queue(self, partition, **kwargs):
        if self._resource not in REMOTE_CONFIG:
            raise KeyError('"{}" resource is not configured'.format(self._resource))

        remote_config = REMOTE_CONFIG[self._resource]
        if 'manager' not in remote_config:
            raise KeyError('Resource "{}" does not provide a queuing system'.format(self._resource))
        partitions = remote_config['partitions']
        partition_found = False
        job_manager_config = {'name': remote_config['manager']}
        for p in partitions:
            if partition in p['id']:
                partition_config = p
                partition_found = True
                partition = p['name']
                break
        if not partition_found:
            raise KeyError('Resource "{}" has no partition "{}"'.format(self._resource, partition))
        job_manager_config['qos'] = partition_config['qos']
        job_manager_config['partition'] = partition_config['name']
        if 'default' not in partition_config:
            self.logger.warning('Partition "{}" on remote resource "{}" has no default options configured'.format(partition, self._resource))
            partition_defaults = {}
        else:
            partition_defaults = partition_config['default']

        job_manager_config.update(**partition_defaults)
        # Update kwargs allow the user to override settings
        job_manager_config.update(**kwargs)
        self['manager'] = job_manager_config
        self._manager = True
        self._partition = partition
        self._partition_kwargs = kwargs
        # Adapt application setting
        self.application(self._application)
        return self