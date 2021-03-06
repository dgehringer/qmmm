from abc import ABCMeta, abstractmethod
from qmmm.core.utils import LoggerMixin, ensure_iterable
from qmmm.core.actions.utils import IODictionary, InputDictionary
from qmmm.core.actions.actions import Action, ActionState
from qmmm.core.event import Event
from logging import ERROR

class Workflow(LoggerMixin, metaclass=ABCMeta):

    def __init__(self, name):
        self._name = name
        self.input = InputDictionary()
        self.output = IODictionary()
        self.input.default.log_level = ERROR
        self._vertices = {}
        self._edges = {}
        self._active_vertex = None
        self._attribute_vertex_name_mapping = {}
        self.finished = Event()
        self.started = Event()
        self.vertex_processing = Event()
        self.vertex_processed = Event()
        self.define_defaults()
        self.define_workflow()
        self.define_dataflow()

    def __setattr__(self, key, value):
        if isinstance(value, Action):
            if hasattr(self, '_vertices'):
                # Except for (_)active_vertex, skip it
                if 'active_vertex' not in key:
                    self._vertices[key] = value
        object.__setattr__(self, key, value)

    def __getattr__(self, item):
        if item in self._vertices.keys():
            return self._vertices[item]

    def edge(self, from_vertex, to_vertex, vertex_state=ActionState.Default, input_hooks=None):
        self._edges[(from_vertex, vertex_state)] = (to_vertex, ensure_iterable(input_hooks) if input_hooks else [])

    @property
    def name(self):
        return self._name

    @abstractmethod
    def define_defaults(self):
        raise NotImplementedError

    @abstractmethod
    def define_workflow(self):
        raise NotImplementedError

    @abstractmethod
    def define_dataflow(self):
        raise NotImplementedError

    def _execute_vertex(self, vertex, input_hooks):
        input_data = vertex.input
        # Resolve lambdas
        # input should be fully resolved apply input hooks
        for key_to_apply, input_hook in input_hooks:
            if key_to_apply not in vertex.input:
                self.logger.warning('Cannot apply input_hook to key {}'.format(key_to_apply))
                continue
            try:
                input_data[key_to_apply] = input_hook(input_data[key_to_apply])
            except Exception as e:
                self.logger.exception('An error ocurred while executing input_hook "{}"'.format(key_to_apply),
                                      exc_info=e)
                raise e
            else:
                self.logger.info('Input hook successful')
        # Resolve it beforehand to get meaningful errors
        data = {k: v for k, v in input_data.items()}
        output_data = vertex.apply(**data)

        if output_data is not None:
            self.append_output(vertex, output_data)

    def run(self):
        if not self.active_vertex:
            raise ValueError('active_vertex is not set')
        self.started.fire()
        input_hooks = []
        while self.active_vertex:
            # Set the correct log leve
            self.active_vertex.log_level = self.input.log_level
            self.logger.info('{}.active_vertex={}'.format(self.name, self.active_vertex.name))
            self.vertex_processing.fire(self.active_vertex)
            self._execute_vertex(self.active_vertex, input_hooks)
            self.vertex_processed.fire(self.active_vertex)
            # Current vertex, find next one
            next_vertex_key = (self.active_vertex, self.active_vertex.state)
            if next_vertex_key in self._edges:
                next_vertex, hooks = self._edges[next_vertex_key]
            else:
                next_vertex = None
                hooks = []

            input_hooks = hooks
            self.active_vertex = next_vertex
        self.finished.fire()

    def append_output(self, vertex, output_data):
        for key, value in output_data.items():
            if key not in vertex.output:
                vertex.output[key] = [value]
            else:
                history = vertex.output[key]
                # Roll the list if it is necessary
                history.append(value)
                if len(history) > vertex._history:
                    # Remove the head of the queue
                    history.pop(0)
                vertex.output[key] = history

    @property
    def active_vertex(self):
        return self._active_vertex

    @active_vertex.setter
    def active_vertex(self, value):
        self._active_vertex = value
