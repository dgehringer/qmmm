from .configuration import Configuration
from paramiko import SSHClient, RSAKey, DSSKey, ECDSAKey, AutoAddPolicy, SSHException
from os.path import exists, join
import re
from .utils import ThreadWithReturnValue, LoggerMixin, remove_white, intersect, StringStream
import os
import tarfile
from uuid import uuid4
from subprocess import PIPE, Popen
from io import TextIOWrapper, StringIO
from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from stat import S_ISDIR
from .configuration import Configuration
from os import mkdir

GLOBAL_CONNECTIONS = {

}
KEY_CLASS_MAPPING = {
    'dss': DSSKey,
    'rsa': RSAKey,
    'dsa': DSSKey,
    'ecdsa': ECDSAKey
}

def close_connection(user, host, logger):
    global GLOBAL_CONNECTIONS
    if (user, host) in GLOBAL_CONNECTIONS:
        ssh_client, sftp_client,  _, _, _ = GLOBAL_CONNECTIONS[(user, host)]
        if sftp_client:
            sftp_client.close()
            logger.info('Closed SFTP connection: "{}@{}!"'.format(user, host))

        if ssh_client:
            ssh_client.close()
            logger.info('Closed SSH connection: "{}@{}!"'.format(user, host))
        del GLOBAL_CONNECTIONS[((user, host))]

def get_connection(user, host, config, logger):
    global GLOBAL_CONNECTIONS
    if (user, host) not in GLOBAL_CONNECTIONS:
        pkey = KEY_CLASS_MAPPING[config['key_type']].from_private_key_file(config['identity_file'])
        host = config['host']
        user = config['user']
        port = config['port']
        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())
        try:
            ssh_client.connect(hostname=host, port=port, username=user, pkey=pkey)
            sftp_client = ssh_client.open_sftp()
            shell_channel = ssh_client.invoke_shell()
            # Prevent it from beeing garbage collected
            shell_channel.keep_this = ssh_client
            shell_stdin = shell_channel.makefile('wb')
            shell_stdout = shell_channel.makefile('r')
        except SSHException:
            logger.exception('Failed to connect to {}@{}'.format(user, host))
            close_connection(user, host, logger)
            raise
        else:
            logger.info('Successfully established connection: "{}@{}"'.format(user, host))
        GLOBAL_CONNECTIONS[(user, host)] = (ssh_client, sftp_client, shell_channel, shell_stdin, shell_stdout)
    return GLOBAL_CONNECTIONS[(user, host)]


class RemoteRunnerMixin(LoggerMixin):

    def __init__(self):
        self._remote_config = None
        self._remote_working_directory_name = Configuration()['general/remote_working_directory_name']
        self._remote_defaults = {
            'host': None,
            'user': None,
            'port': 22,
            'identity_file': None,
            'key_type': 'rsa',
            'prefix': None
        }
        self._ssh_client = None
        self._sftp_client = None
        self._remote_args = None
        self._pkey = None
        self._shell_channel = None
        self._shell_stdin = None
        self._shell_stdout = None
        self._shell_processed_stdout = StringStream()

        self._key_class_mapping = {
            'dss': DSSKey,
            'rsa': RSAKey,
            'dsa': DSSKey,
            'ecdsa': ECDSAKey
        }

    def _initialize_remote(self, remote_kwargs):
        self._process_arguments(remote_kwargs)
        remote = self._remote_args
        self._pkey = self._key_class_mapping[remote['key_type']].from_private_key_file(remote['identity_file'])
        self._host = remote['host']
        self._user = remote['user']
        self._port = remote['port']
        self._remote_prefix = remote['prefix']
        self._remote_working_directory = join(self._remote_prefix, self._remote_working_directory_name)

        ssh_client, sftp_client, shell_channel, shell_stdin, shell_stdout = get_connection(self._user, self._host, remote, self.logger)
        self._ssh_client = ssh_client
        self._sftp_client = sftp_client
        self._shell_channel = shell_channel
        self._shell_stdin = shell_stdin
        self._shell_stdout = shell_stdout

        self._setup_working_directory()

    def _setup_working_directory(self):
        if self._ssh_client and self._sftp_client:
            if not self._remote_exists(self._remote_working_directory):
                self._sftp_client.mkdir(self._remote_working_directory)
                self.logger.info('Created remote directory {}@{}:{}'.format(self._user, self._host, self._remote_working_directory))
            self._sftp_client.chdir(self._remote_working_directory)


    def _remote_is_dir(self, remote_path):
        try:
            return S_ISDIR(self._sftp_client.stat(remote_path).st_mode)
        except IOError:
            return False

    def _remote_rmdir(self, remote_path):
        files = self._sftp_client.listdir(path=remote_path)
        for f in files:
            remote_file_path = join(remote_path, f)
            if self._remote_is_dir(remote_file_path):
                self._remote_rmdir(remote_file_path)
            else:
                self._sftp_client.remove(remote_file_path)
        self._sftp_client.rmdir(remote_path)

    def _destroy_working_directory(self, calculation_directory):
        if self._ssh_client and self._sftp_client:
            if self._remote_exists(self._remote_working_directory):
                remote_calculation_path = join(self._remote_working_directory, calculation_directory)
                if self._remote_exists(remote_calculation_path):
                    self._remote_rmdir(remote_calculation_path)
                    self.logger.info(
                    'Removed remote directory {}@{}:{}'.format(self._user, self._host, remote_calculation_path))
                else:
                    self.logger.warning('Calculation directory {}@{}:{} does not exist!'.format(self._user, self._host, calculation_directory))

    def _remote_exists(self, remote_path):
        try:
            self._sftp_client.stat(remote_path)
        except IOError:
            return False
        else:
            return True

    def _remote_open(self, remote_path, mode):
        return self._sftp_client.open(remote_path, mode=mode)

    def _remote_extract_archive(self, remote_path, remote_dir):
        command = 'tar xvzf {} {}'.format(remote_path, '--directory {}'.format(remote_dir) if remote_dir else '')
        return self._remote_send_command(command)

    def _remote_create_archive(self, remote_path, name, files):
        commands = ['cd {} '.format(remote_path),
                   'tar czvf {} {}'.format(name, ' '.join(files))]
        return [self._remote_send_command(cmd) for cmd in commands]

    def _remote_send_command(self, cmd, return_stdout=False, propagate_stdout=True):
        cmd = cmd.strip('\n')
        finish = '__remote_command_finish_mark__'
        #self._shell_stdin.write(cmd + '\n')
        echo_cmd = 'echo {} $?'.format(finish)
        #self._shell_stdin.write(echo_cmd + '\n')
        self._shell_stdin.write('; '.join([cmd, echo_cmd]) + '\n')
        #shin = self._shell_stdin
        self._shell_stdin.flush()

        shout = []
        sherr = []
        exit_status = 0
        for line in self._shell_stdout:
            if str(line).startswith(cmd) or str(line).startswith(echo_cmd):
                # up for now filled with shell junk from stdin
                shout = []
            elif str(line).startswith(finish):
                # our finish command ends with the exit status
                #try:
                exit_status = int(str(line).rsplit(maxsplit=1)[1])
                #except ValueError:
                #    self.logger.exception('Could not parse exit code for command {}: {}'.format(cmd, line))
                #    exit_status = -1
                #    break
                if exit_status:
                    # stderr is combined with stdout.
                    # thus, swap sherr with shout in a case of failure.
                    sherr = shout
                    shout = []
                break
            else:
                # get rid of 'coloring and formatting' special characters
                processed = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', line).replace('\b', '').replace('\r', '')

                # Skip command line
                if not cmd in processed and not echo_cmd in processed:
                    if propagate_stdout:
                        self._shell_processed_stdout.write(processed)
                        self._shell_processed_stdout.flush()
                    shout.append(processed)

        # first and last lines of shout/sherr contain a prompt
        if shout and echo_cmd in shout[-1]:
            shout.pop()
        if shout and cmd in shout[0]:
            shout.pop(0)
        if sherr and echo_cmd in sherr[-1]:
            sherr.pop()
        if sherr and cmd in sherr[0]:
            sherr.pop(0)

        # Pipe all processed output to

        return exit_status == 0 if not return_stdout else (exit_status, shout)
        #return shin, shout, sherr

    def _remote_shell_alive(self):
        return self._shell_handle is not None and self._shell_input is not None and self._shell_output is not None

    def _remote_get_shell(self):
        if not self._remote_shell_alive():
            import shlex
            self._shell_handle = self._shell_channel
            self._shell_input = self._shell_stdin
            self._shell_output = self._shell_processed_stdout

        return (self._shell_input, self._shell_output)

    def _remote_chdir(self, remote_path):
        return self._remote_send_command('cd {}'.format(remote_path))

    def _sync_directory(self, local_dir):
        if self._ssh_client and self._sftp_client:
            self._sftp_client.chdir(self._remote_working_directory)
            for root, dirs, files in os.walk(local_dir):
                remote_root = join(self._remote_working_directory, root)
                remote_files = []
                local_files = files
                if not self._remote_exists(remote_root):
                    self._sftp_client.mkdir(remote_root)
                else:
                    remote_files = self._sftp_client.listdir(remote_root)

                # Find all common files
                common_files = intersect(remote_files, local_files)
                # Find all files which are not on the server
                local_exclusive = [lf for lf in local_files if lf not in common_files]
                #Find all files which are on the remote server only
                remote_exclusive = [rf for rf in remote_files if rf not in common_files]

                # Transfer local files
                if len(local_exclusive) != 0:

                    with NamedTemporaryFile() as upload:
                        upload_archive_name = str(uuid4())
                        remote_upload_archive_name = join(self._remote_working_directory, upload_archive_name)
                        with tarfile.open(upload.name, 'w:gz') as upload_archive:
                            for lf in local_exclusive:
                                local_file = join(root, lf)
                                #remote_file = join(remote_root, lf)
                                upload_archive.add(local_file, arcname=local_file)
                        self._sftp_client.put(upload.name, remote_upload_archive_name)
                        self.logger.info('Transferred {} -> {}@{}:{}'.format(upload.name, self._user, self._host, remote_upload_archive_name))
                        if self._remote_extract_archive(remote_upload_archive_name, self._remote_working_directory):
                            self._sftp_client.remove(remote_upload_archive_name)
                        else:
                            raise RuntimeError('Could not extract remote archive: "{}"'.format(remote_upload_archive_name))

                # Fetch files from server if there are any
                if len(remote_exclusive) != 0:
                    download_archive_name = str(uuid4())
                    remote_download_archive_name = join(self._remote_working_directory, download_archive_name)
                    if self._remote_create_archive(self._remote_working_directory, download_archive_name, [join(root, rf) for rf in remote_exclusive]):

                        # Open the file remote
                        with NamedTemporaryFile() as download:
                            with self._sftp_client.open(remote_download_archive_name) as remote_archive:
                                # Download the file to the temporary file
                                copyfileobj(remote_archive, download)
                                # Go to the start of the file
                                download.seek(0)
                                with tarfile.open(download.name, 'r:gz') as download_archive:
                                    for member in download_archive.getmembers():
                                        if member.isdir():
                                            if not exists(member.name):
                                                mkdir(member.name)
                                        else:
                                            with open(member.name, 'wb') as destination:
                                                copyfileobj(download_archive.extractfile(member), destination)
                                                self.logger.debug('Extracted from archive {1} file {0}'.format(member.name, download_archive_name))
                            # If we are here everything worked out nicely
                            # Clean up
                            self._sftp_client.remove(remote_download_archive_name)

                    else:
                        raise RuntimeError('Could not create remote archive: "{}"'.format(remote_download_archive_name))

    def _close(self):
        close_connection(self._user, self._host, self.logger)

    def _process_arguments(self, remote_kwargs):
        processed_args = {}
        for k, v in self._remote_defaults.items():
            # If value is None we need something
            if v is None:
                if k in remote_kwargs:
                    processed_args[k] = remote_kwargs[k]
                else:
                    raise ValueError('Need an value for argument "{}"'.format(k))
            else:
                if k in remote_kwargs:
                    processed_args[k] = remote_kwargs[k]
                else:
                    processed_args[k] = v
        self._remote_args = processed_args


CALCULATION_END_MARK = Configuration()['general/end_mark']


class RunnerMixin(LoggerMixin):

    def __init__(self):
        self._calc = None
        self._initialized = False
        self._remote = False
        self._remote_working_directory = str(uuid4())
        self._shell_handle = None
        self._shell_input = None
        self._shell_output = None
        self._exitcode = None

    def bind(self, calc):
        self._calc = calc
        self._initialized = True
        self.logger.debug('Bound calculation "{}" in directory "{}"'.format(self._calc.name, self._calc.working_directory))

    def unbind(self):
        self.logger.debug(
            'Unbound calculation "{}" in directory "{}"'.format(self._calc.name, self._calc.working_directory))
        self._calc = None
        self._initialized = False

    def _get_shell(self):
        if not self._shell_alive():
            import shlex
            self._shell_handle = Popen(shlex.split('/bin/bash'), stdin=PIPE, stdout=PIPE)
            self._shell_stdin = TextIOWrapper(self._shell_handle.stdin, encoding='utf-8')
            self._shell_stdout = TextIOWrapper(self._shell_handle.stdout, encoding='utf-8')
            self._shell_input = self._shell_stdin
            self._shell_output = StringStream()
        return (self._shell_input, self._shell_output)

    def _execute(self, commands):
        #Execute vasp
        command_list = commands +  [
            'echo "{} $?"'.format(CALCULATION_END_MARK)
        ]

        for c in command_list:
            self._send_command(c)
        self._shell_input.flush()

    def _chdir(self, directory):
        os.chdir(directory)

    def _make_listener_thread(self, log_file):
        log_file_fd = open(log_file, 'w')
        thr_read_log = ThreadWithReturnValue(target=self._read_output, args=(log_file_fd,))
        thr_read_log.start()
        return thr_read_log

    def _run(self, commands, preamble=None):
        """Method which explicitely runs VASP."""
        cwd = os.getcwd()
        self._chdir(self._calc.working_directory)
        # Start shell
        self._get_shell()
        thr_read_log = self._make_listener_thread(self._calc.log_file)

        preamble = preamble or []
        self._execute(preamble + commands)
        # Query vasp exit code
        # self._send_command('echo $?')
        # Print calculation end_mark
        exitcode = thr_read_log.join()
        self._chdir(cwd)

        #Check if exitcode is 0
        return exitcode

    def _shell_alive(self):
        return self._shell_handle and not isinstance(
            self._shell_handle.poll(), int)

    def _send_command(self, cmd, return_stdout=False, propagate_stdout=True):
        if not self._shell_alive():
            raise RuntimeError('Shell is not open')

        finish = '__local_command_finish_mark__'
        # self._shell_stdin.write(cmd + '\n')
        echo_cmd = 'echo {} $?\n'.format(finish)
        command = cmd.strip('\n')
        self._shell_input.write('; '.join([cmd, echo_cmd]))
        # Very important - flush() otherwise nothing will happen and the reader thread will get stuck in an infinite
        self._shell_input.flush()

        shout = []
        exit_status = 0
        for line in self._shell_stdout:
            if finish in line:
                try:
                    mark, code = line.lstrip().rstrip().split(' ')
                    exit_status = int(code)
                    assert mark == finish
                except:
                    shout.append(line)
                    if propagate_stdout:
                        self._shell_output.write(line)
                        self._shell_stdout.flush()
                else:
                    break
            else:
                shout.append(line)
                if propagate_stdout:
                    self._shell_output.write(line)
                    self._shell_stdout.flush()

        return exit_status == 0 if not return_stdout else (exit_status, shout)

    def _read_output(self, log_file):
        f = self._shell_output
        line = f.readline()
        log_file.write(line)
        self.logger.debug(line)
        while CALCULATION_END_MARK not in line.strip():
            line = f.readline()
            if line:
                log_file.write(line)
                self.logger.debug(line)
        # If everything worked old_line should contain the exit status
        line, exitcode = line.strip().split(' ')
        log_file.flush()
        try:
            exitcode = int(exitcode)
        except ValueError:
            log_file.close()
            return None

        else:
            log_file.close()
            return exitcode

