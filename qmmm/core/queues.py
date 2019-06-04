from qmmm.core.runner import RemoteRunner
from qmmm.core.utils import LoggerMixin
from qmmm.core.configuration import Configuration
from abc import ABCMeta, abstractmethod
from os.path import join
from enum import Enum

CALCULATION_END_MARK = Configuration()['general/end_mark']

class JobStatus(Enum):

    Submitted = 'SUBMITTED'
    Pending = 'PENDING'
    Running = 'RUNNING'
    Completed = 'COMPLETED',
    Completing = 'COMPLETING'
    Crashed = 'CRASHED'


def get_job_manager(manager_name, *args):
    managers = {
        'slurm': SLURMManager
    }
    if manager_name.lower() not in managers:
        raise KeyError('No  implementation available for "{}" job manager!'.format(manager_name))
    else:
        return managers[manager_name](*args)

class AbstractJobManager(LoggerMixin, metaclass=ABCMeta):

    def __init__(self, runner):
        if not isinstance(runner, RemoteRunner):
            raise TypeError('runner_mixin must be a runner core.mixins.RunnerMixin object')
        self._runner = runner


    @abstractmethod
    def _make_interactive_shell(self):
        raise NotImplementedError()

    @abstractmethod
    def _submit_job(self, command, preamble=None):
        raise NotImplementedError()

    @abstractmethod
    def _cancel_job(self, job_id):
        raise NotImplementedError()

    @abstractmethod
    def _status_job(self, job_id):
        raise NotImplementedError()

    @abstractmethod
    def _initialize_manager(self, manager_args):
        raise NotImplementedError()


class SLURMManager(AbstractJobManager):

    def __init__(self, runner):
        super(SLURMManager, self).__init__(runner)
        self._manager_defaults = {
            'tasks_per_node': 1,
            'partition': None,
            'qos': None,
            'exclusive': True,
            'nodes': 1,
            'interactive': True,
            'mem': None
        }
        self._manager_args = None


    def _initialize_manager(self, manager_args):
        self._process_arguments(manager_args)


    def _process_arguments(self, manager_args):
        processed_args = {}
        for k, v in self._manager_defaults.items():
            # If value is None we need something
            if v is None:
                if k in manager_args:
                    processed_args[k] = manager_args[k]
                else:
                    raise ValueError('Need an value for argument "{}"'.format(k))
            else:
                if k in manager_args:
                    processed_args[k] = manager_args[k]
                else:
                    processed_args[k] = v
        self._manager_args = processed_args

    def _status_job(self, job_id):
        # Check if job is in the qeueue
        job_mark = '__job_mark__'
        check_command = 'squeue --user=`whoami` --noheader --format "{} %A"'.format(job_mark)
        exit_status, output = self._runner._send_command(check_command, return_stdout=True, propagate_stdout=False)
        job_list = []
        if exit_status != 0:
            raise RuntimeError('Failed to query "squeue"')
        else:
            # process output
            for line in output:
                if line.rstrip().lstrip().startswith(job_mark):
                    try:
                        mark, jid = line.rstrip().lstrip().split(' ')
                        assert mark == job_mark
                        jid = int(jid)
                    except:
                        continue
                    else:
                        job_list.append(jid)
        if job_id in job_list:
            check_command = 'squeue --job {0} --noheader --format "{1} %A %T"'.format(job_id, job_mark)
            exit_status, output = self._runner._send_command(check_command, return_stdout=True,
                                                             propagate_stdout=False)
            job_states = {}
            if exit_status != 0:
                raise RuntimeError('Failed to query "squeue" status for job id "{}"'.format(job_id))
            else:
                for line in output:
                    if line.rstrip().lstrip().startswith(job_mark):
                        for line in output:
                            if line.rstrip().lstrip().startswith(job_mark):
                                try:
                                    mark, jid, jstatus = line.rstrip().lstrip().split(' ')
                                    assert mark == job_mark
                                    jid = int(jid)
                                    jstatus = JobStatus(jstatus.upper())
                                except:
                                    continue
                                else:
                                    job_states[jid] = jstatus
                if len(job_states) != 1:
                    print(output)
                    raise RuntimeError('Failed to parse output from status request: {}'.format(job_states))
                else:
                    status = job_states[list(job_states.keys())[0]]
            # Look for the current status
        else:
            # If it is not there, it has completed
            # But do not check if the calculation crashed
            status = JobStatus.Completed
        return status

    def _cancel_job(self, job_id):
        command = 'scancel {}'.format(job_id)
        if not self._runner._send_command(command, propagate_stdout=False, return_stdout=False):
            raise RuntimeError('Failed to cancel job "{0}"'.format(job_id))

    @property
    def interactive(self):
        return self._manager_args['interactive']

    def _make_interactive_shell(self):
        # Get shell pipes
        self._shell_input = self._runner._shell_input
        self._shell_output = self._runner._shell_output
        configuration = self._manager_args
        switch_mapping = {
            'nodes': '--nodes',
            'tasks_per_node': '--ntasks-per-node',
            'partition': '--partition',
            'qos': '--qos',
            'exclusive': '--exclusive',
            'mem': '--mem'
        }
        base_command = 'srun'
        srun_args = ['{}{}'.format(v, '={}'.format(
            configuration[k]) if not isinstance(configuration[k], bool) else '')
                     for k, v in switch_mapping.items()]
        command = ' '.join([base_command] + srun_args + ['--pty bash'])
        allocated_mark = '__hpc_resource_acquired__'
        self._shell_input.write(command + '\n')
        self._shell_input.write('echo {} $RANDOM\n'.format(allocated_mark))
        self._shell_input.flush()
        self.logger.info('Wating for resources to be allocated ...')
        # Block execution until HPC resources are acquired
        for line in self._shell_output:
            if allocated_mark in line:
                try:
                    mark, magic = line.split(' ')
                    int(magic)
                except:
                    pass
                else:
                    # Found the line => unblock
                    break

        self.logger.info('Cluster resources acquired!')

    def _submit_job(self, command, preamble=None):
        # Could be remote or not but we don't care

        configuration = self._manager_args
        calc_directory = self._runner.calculation_directory()
        calculation_name = self._runner.calculation.name
        start_script_name = '{}.sh'.format(calculation_name)
        start_script_path = join(calc_directory, start_script_name)
        output_file_name = '{}.slurm.log'.format(calculation_name)
        start_script = []
        start_script.append('#!/bin/bash\n')
        start_script.append('#SBATCH --job-name={}\n'.format(calculation_name))
        start_script.append('#SBATCH --output={}\n'.format(output_file_name))
        switch_mapping = {
            'nodes': '--nodes',
            'tasks_per_node': '--ntasks-per-node',
            'partition': '--partition',
            'qos': '--qos',
            'exclusive': '--exclusive',
            'mem': '--mem'
        }
        for k, v in switch_mapping.items():
            start_script.append('#SBATCH {switch}{value}\n'.format(switch=v,
                                                                  value='={}'.format(configuration[k]) if not isinstance(configuration[k], bool) else ''))

        # Write there a comment
        start_script.append('\n')
        start_script.append('# This script was automatically generated\n')
        start_script.append('\n')
        if not preamble:
            preamble = []
        else:
            preamble = preamble
        for line in preamble:
            line = line if line.endswith('\n') else line+'\n'
            start_script.append(line)
        start_script.append(command if command.endswith('\n') else command+'\n')

        echo_command = 'echo {} $?\n'.format(CALCULATION_END_MARK)

        start_script.append(echo_command)


        with self._runner._open(start_script_path, mode='w') as file_handle:
            for line in start_script:
                file_handle.write(line)
        self.logger.info('Created SLURM script "{}"'.format(start_script_path))

        change_command = 'CURRENT_DIRECTORY=`pwd`; cd {}'.format(calc_directory)
        submit_command = 'sbatch --parsable {}'.format(start_script_name)
        change_back_command = 'cd $CURRENT_DIRECTORY'

        if not self._runner._send_command(change_command, propagate_stdout=False, return_stdout=False):
            raise RuntimeError('Could not switch to calculation directory "{}"'.format(calc_directory))

        exit_status, stdout = self._runner._send_command(submit_command, propagate_stdout=False, return_stdout=True)
        if exit_status != 0:
            # Try to switch back
            if not self._runner._send_command(change_back_command, propagate_stdout=False, return_stdout=False):
                raise RuntimeError('An error ocurred when changing directories')
            raise RuntimeError('Failed to submit job')
        else:
            try:
                last_line = stdout[-1]
                # If no cluster is specified only job id will be returned, therefore append a semicolon to make split work
                if ';' not in last_line:
                    last_line += ';'
                job_id, cluster = last_line.split(';')
                job_id = int(job_id)
                if cluster == '':
                    cluster = None
            except ValueError:
                # Try to switch back
                if not self._runner._send_command(change_back_command, propagate_stdout=False, return_stdout=False):
                    raise RuntimeError('An error ocurred when changing directories')
                raise RuntimeError('Failed to submit job')
            else:
                # Try to switch back
                if not self._runner._send_command(change_back_command, propagate_stdout=False, return_stdout=False):
                    raise RuntimeError('An error ocurred when changing directories')
                # Everything went fine here
                #Check if the job is in the q
                self.logger.info('Successfully submitted job "{0}"'.format(job_id))
                return job_id



