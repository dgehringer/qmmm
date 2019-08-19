from abc import ABCMeta, abstractmethod
from qmmm.core.utils import LoggerMixin
from qmmm.core.utils import ready, run_once, working_directory


class AbstractCalculationSet(LoggerMixin, metaclass=ABCMeta):

    def __init__(self, name, delete=False, prefix=None):
        self._name = name
        self._delete = delete
        self._calculations = []
        self._ready = []
        self._calculation_mapping = {}
        self._prefix = prefix
        self._working_directory = working_directory(self._name, delete=self._delete, prefix=self._prefix)

    @abstractmethod
    def get_parameter_configurations(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_calculation_name(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_calculation_config(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_calculation_type(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_structure(self, **kwargs):
        raise NotImplementedError

    @property
    def ready(self):
        if self._calculations:
            return len(self._calculations) == len(self._ready)
        else:
            return False

    @property
    def name(self):
        return self._name

    @property
    def calculations(self):
        for calculation in self._calculations:
            yield calculation

    @property
    def active(self):
        return self._working_directory.active

    @property
    def config(self):
        if not self.ready:
            self.logger.warn('Not all calculations have finished yet')
        for calculation in self._calculations:
            yield calculation, self._calculation_mapping[calculation.id]

    def check_ready(self, run=True, raise_error=False):
        if self.active:
            self._ready = ready(self._calculations, run=run, raise_error=raise_error)
        else:
            with self:
                self._ready = ready(self._calculations, run=run, raise_error=raise_error)
        for ready_ in self._ready:
            # Update the calculation list
            self._calculations[self._calculations.index(ready_.id)] = ready_
        return self.ready

    def run(self, **kwargs):
        self._calculations.clear()
        # Avoid double working_directory nesting
        if self.active:
            # The working directory is already active, do nothin
            self._run(**kwargs)
        else:
            # change into the working directory
            with self:
                self._run(**kwargs)
        # Check how many calculations have already finished
        self._ready = ready(self._calculations, run=False)

    def _run(self, **kwargs):
        for config in self.get_parameter_configurations(**kwargs):
            # Get a meaningful name for the calculation
            calculation_name = self.get_calculation_name(**config)
            # Get the structure
            structure = self.get_structure(**config)
            # Get the calculation type
            cls_ = self.get_calculation_type(**config)
            # Execute it
            calculation = cls_(structure, calculation_name)
            combined = config.copy()
            combined.update(kwargs)
            calculation = run_once(calculation, **self.get_calculation_config(**combined))
            self._calculation_mapping[calculation.id] = config
            self._calculations.append(calculation)

    def __enter__(self):
        # We want to have access to all the calculations in the folder
        self._working_directory.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._working_directory.__exit__(exc_type, exc_val, exc_tb)
