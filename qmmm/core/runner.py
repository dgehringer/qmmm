
from .mixins import RunnerMixin, RemoteRunnerMixin
from .configuration import get_setting
from os.path import join, exists
from .utils import StringStream
from monty.json import MSONable

class RemoteRunner(RunnerMixin, RemoteRunnerMixin, MSONable):

    def __init__(self):
        super(RemoteRunner, self).__init__()
        RunnerMixin.__init__(self)
        RemoteRunnerMixin.__init__(self)
        self._remote = False
        self._job_manager = None
        self._has_job_manager = False
        self._submitted = False
        self._job_id = None


    def bind(self, calc, remote=None, manager=None):
        super(RemoteRunner, self).bind(calc)
        if remote:
            self._initialize_remote(remote)
            self._remote = True
        if manager:
            if not 'name' in manager:
                raise KeyError('I do not know which job manager is running')
            from .queues import get_job_manager
            self._job_manager = get_job_manager(manager['name'], self)
            queue_system =  manager['name']
            del manager['name']
            self._job_manager._initialize_manager(manager)
            manager['name'] = queue_system
            self._has_job_manager = True

    def unbind(self):
        if self._remote:
            self._close()
        super(RemoteRunner, self).unbind()

    def calculation_directory(self):
        if self._remote:
            return join(self._remote_working_directory, self._calc.working_directory)
        else:
            return self._calc.working_directory

    def _open(self, fname, mode):
        path = join(self.calculation_directory(), fname)
        if self._remote:
            return self._remote_open(path, mode=mode)
        else:
            return open(path, mode=mode)


    def _get_shell(self):
        if self._remote:
            result = self._remote_get_shell()
        else:
            result = super(RemoteRunner, self)._get_shell()
        if self._has_job_manager:
            if self._job_manager.interactive:
                self._job_manager._make_interactive_shell()
        return result

    def _shell_alive(self):
        if self._remote:
            return self._remote_shell_alive()
        else:
            return super(RemoteRunner, self)._shell_alive()

    def _exists(self, path):
        if self._remote:
            return self._remote_exists(path)
        else:
            return exists(path)

    def _send_command(self, cmd, return_stdout=False, propagate_stdout=True):
        if self._remote:
            return self._remote_send_command(cmd, return_stdout=return_stdout, propagate_stdout=propagate_stdout)
        else:
            return super(RemoteRunner, self)._send_command(cmd, return_stdout=return_stdout, propagate_stdout=propagate_stdout)

    def _chdir(self, directory):
        if self._remote:
            #Change both if remote
            self._remote_chdir(directory)
            super(RemoteRunner, self)._chdir(directory)
        else:
            super(RemoteRunner, self)._chdir(directory)

    def check(self):
        if not self._submitted:
            raise RuntimeError('This job was not submitted yet')
        if not self._job_id:
            raise RuntimeError('I do not have and job_id')
        if not self._job_manager:
            raise RuntimeError('There is no queueing system running')

        from .queues import JobStatus
        status =  self._job_manager._status_job(self._job_id)
        if status == JobStatus.Completed:
            # The job completed but we have to check if everything went correctly
            # Fake shell output
            tmp_shell_out = self._shell_output
            self._shell_output = StringStream()
            log_file_name = '{}.slurm.log'.format(self.calculation.name)
            log_file_path = join(self.calculation_directory(), log_file_name)
            log_thr = self._make_listener_thread(self.calculation.get_path(self.calculation.log_file))
            with self._open(log_file_path, mode='r') as log_file_handle:
                for line in log_file_handle:
                    self._shell_output.write(line)
            exit_code = log_thr.join()
            self._exitcode = exit_code
            return exit_code

        else:
            # Job is in queue

            return None


    def run(self, command, preamble=None):
        # Sync directory with remote server

        if self._remote:
            # Sync first to ensure working directory exists
            # Make sure all files are available on the server
            self._sync_directory(self._calc.working_directory)
            self._remote_chdir(join(self._remote_working_directory, self._calc.working_directory))

        # Check if there's a queuing system
        if not self._has_job_manager:
            # There is no queuing system run it directly
            exitcode = self._run([command], preamble=preamble)
            if self._remote:
                # We run it remotely without a job jobmanager
                # We ran the job interactively it should be finished by now therefore sync
                self._sync_directory(self._calc.working_directory)
                # Remove the working directory
                self._destroy_working_directory(self._calc.working_directory)
        else:
            if self._job_manager.interactive:
                # There is a queing system but we run an interactive job
                exitcode = self._run([command], preamble=preamble)

                # We ran the job interactively it should be finished by now therefore sync
                self._sync_directory(self._calc.working_directory)
                # Remove the working directory
                self._destroy_working_directory(self._calc.working_directory)
            else:
                # There is a queueing system but we just submit the job or check its state
                if not self._submitted:
                    #Submit the job
                    job_id = self._job_manager._submit_job(command, preamble=preamble)
                    self._job_id = job_id
                    self._submitted = True
                    exitcode = None
                else:
                    # Check job_status and check if it has finished
                    exitcode = self.check()
                    if exitcode is not None:
                        self._sync_directory(self._calc.working_directory)
                        # Remove the working directory
                        self._destroy_working_directory(self._calc.working_directory)
        return exitcode

    @property
    def remote(self):
        return self._remote

    @property
    def calculation(self):
        return self._calc

    def as_dict(self):
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        if self._remote:
            # Need it to overwrite
            d['remote_id'] = self._remote_working_directory_name
        d['submitted'] = self._submitted
        if self._job_id is not None:
            d['job_id'] = self._job_id
        return d


    @classmethod
    def from_dict(cls, d):
        obj = cls()
        if 'remote_id' in d:
            obj._remote_working_directory_name = d['remote_id']
        obj._submitted = d['submitted']
        if 'job_id' in d:
            obj._job_id = d['job_id']
        return obj



class VASPRunner(RemoteRunner):


    def run(self, command, preamble=None):
        vasp_command = command
        if vasp_command is None:
            raise RuntimeError('Could not get VASP_COMMAND setting')
        return super(VASPRunner, self).run(vasp_command, preamble=preamble)


class LAMMPSRunner(RemoteRunner):


    def run(self, command, preamble=None):
        lammps_command = command
        # Add some options to execute
        lammps_command = ' '.join([lammps_command, '-in', self.calculation.input_file, '-log', self.calculation.log_file])
        if lammps_command is None:
            raise RuntimeError('Could not get LAMMPS_COMMAND setting')
        return super(LAMMPSRunner, self).run(lammps_command, preamble=preamble)