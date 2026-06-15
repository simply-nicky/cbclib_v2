"""SLURM job submission and monitoring utilities.

This module provides small wrappers for writing ``sbatch`` scripts, submitting
jobs, polling ``squeue``/``sacct``, streaming output, and waiting for arrays of
jobs to finish.
"""
import asyncio
from contextlib import contextmanager
import os
import re
from time import sleep
from shlex import quote
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper as TemporaryFileWrapper
import subprocess
from typing import (AsyncGenerator, ClassVar, Dict, Iterator, List, NamedTuple, Set,
                    Tuple, overload)
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from .._src.parser import from_container, from_file
from .._src.data_container import Container

class SLURMConfig(NamedTuple):
    """Executable names used by :class:`SLURMJobManager`.

    Attributes:
        sbatch: Command used to submit batch jobs.
        squeue: Command used to query live jobs.
        sacct: Command used to query accounting records.
        scancel: Command used to cancel jobs.
    """
    sbatch  : str = "sbatch"
    squeue  : str = "squeue"
    sacct   : str = "sacct"
    scancel : str = "scancel"

@dataclass
class ScriptSpec(Container):
    """Settings used to construct an ``sbatch`` script header.

    The object stores both ``#SBATCH`` directives and shell setup commands
    that should run before the submitted command.

    Attributes:
        partition: SLURM partition name. If empty, no partition directive is
            written.
        time: Wall-clock time limit passed to ``--time``.
        nodes: Number of nodes requested with ``--nodes``.
        chdir: Working directory for the job. Defaults to the current
            directory.
        mem: Memory request passed to ``--mem``.
        exclusive: If ``True``, request exclusive node access.
        output: SLURM output filename pattern.
        error: SLURM error filename pattern.
        modules: Environment modules loaded before running the command.
        define_macros: Environment variables exported before running the
            command.
        conda_env: Conda environment activated before running the command.
        conda_source: Shell file sourced before ``conda activate``.
    """
    partition       : str = ''
    time            : str = "01:00:00"
    nodes           : int = 1
    chdir           : str = ''
    mem             : str = '0'
    exclusive       : bool = False
    output          : str = 'slurm-%j.out'
    error           : str = 'slurm-%j.out'
    modules         : List[str] = field(default_factory=list)
    define_macros   : Dict[str, str] = field(default_factory=dict)
    conda_env       : str = ''
    conda_source    : str = ''

    shell_pattern   : ClassVar[re.Pattern] = re.compile(
    r"""
    (                               # start of group
        \$\([^)]+\)                 # command or arithmetic substitution: $(...) or $((...))
        |                           # or
        \$\{[^}]+\}                 # variable expansion: ${VAR}
        |                           # or
        \$[A-Za-z_][A-Za-z0-9_]*    # simple variable: $VAR
    )
    """, re.VERBOSE)

    def __post_init__(self):
        if not self.chdir:
            self.chdir = os.getcwd()

    @classmethod
    def is_shell_expression(cls, value: str) -> bool:
        """Return whether a value contains shell expansion syntax.

        Args:
            value: Value to inspect.

        Returns:
            ``True`` if *value* contains shell variable or command
            substitution syntax.
        """
        return bool(cls.shell_pattern.search(value))

    @classmethod
    def read(cls, file: str) -> 'ScriptSpec':
        """Read script parameters from a configuration file.

        Args:
            file: Path to the configuration file.

        Returns:
            Parsed script specification.
        """
        parser = from_file(file, cls, 'parameters')
        return cls.from_dict(**parser.read(file))

    def write(self, file: str):
        """Write script parameters to a configuration file.

        Args:
            file: Output configuration file.
        """
        parser = from_container(file, self, 'parameters')
        parser.write(file, self)

    def add_define(self, key: str, value: str) -> None:
        """Add an exported environment variable.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self.define_macros[key] = value

    def script_header(self) -> List[str]:
        """Return the ``#SBATCH`` header lines.

        Returns:
            Header lines for an ``sbatch`` script.
        """
        header: List[str] = []
        if self.partition:
            header.append(f"#SBATCH --partition={self.partition}\n")
        header.append(f"#SBATCH --time={self.time}\n")
        header.append(f"#SBATCH --nodes={self.nodes}\n")
        header.append(f"#SBATCH --chdir={self.chdir}\n")
        header.append(f"#SBATCH --mem={self.mem}\n")
        if self.exclusive:
            header.append("#SBATCH --exclusive\n")
        header.append("#SBATCH --open-mode=append\n")
        header.append(f"#SBATCH --output={self.output}\n")
        header.append(f"#SBATCH --error={self.error}\n")
        return header

    def script_body(self) -> List[str]:
        """Return shell commands used to prepare the job environment.

        Returns:
            Shell lines that load modules, export variables, and activate a
            Conda environment.
        """
        body = []
        for module in self.modules:
            body.append(f"module load {quote(module)}\n")
        for key, value in self.define_macros.items():
            if self.is_shell_expression(value):
                body.append(f"export {key}={value}\n")
            else:
                body.append(f"export {key}={quote(value)}\n")
        if self.conda_env:
            if self.conda_source:
                body.append(f"source {self.conda_source}\n")
            body.append(f"conda activate {quote(self.conda_env)}\n")
        return body

@dataclass
class SLURMScript:
    """Shell command and metadata for an ``sbatch`` submission.

    Attributes:
        command: Shell command executed by the job.
        job_name: SLURM job name.
        parameters: Header and environment parameters for the generated
            script.
    """
    command         : str
    job_name        : str
    parameters      : ScriptSpec = field(default_factory=ScriptSpec)

    def script_header(self) -> List[str]:
        """Return the complete ``#SBATCH`` header.

        Returns:
            Script header lines, including the job name directive.
        """
        header = self.parameters.script_header()
        header.append(f"#SBATCH --job-name={self.job_name}\n")

        return header

    def script_body(self) -> List[str]:
        """Return the complete script body.

        Returns:
            Environment setup lines followed by the command wrapped in
            ``bash -lc``.
        """
        body = self.parameters.script_body()
        body.append(f"bash -lc {quote(self.command)}\n")
        return body

    @contextmanager
    def write_file(self, directory: str | os.PathLike[str] | None=None
                   ) -> Iterator[TemporaryFileWrapper]:
        """Write the script to a temporary shell file.

        Args:
            directory: Directory in which to create the temporary file.

        Yields:
            Open temporary script file.
        """
        temp_file = NamedTemporaryFile("w", dir=directory, suffix='.sh')
        try:
            temp_file.write("#!/bin/bash\n")
            temp_file.writelines(self.script_header())
            temp_file.writelines(self.script_body())
            temp_file.flush()
            yield temp_file
        finally:
            temp_file.close()

@dataclass
class JobID:
    """SLURM job identifier.

    Attributes:
        id: Base SLURM job id.
        task_id: Array task id. ``None`` denotes a non-array job.
    """
    id      : int
    task_id : int | None = None

    @classmethod
    def from_string(cls, s: str) -> 'JobID':
        """Parse a SLURM job id string.

        Args:
            s: Job id string such as ``"12345"`` or ``"12345_7"``.

        Returns:
            Parsed job identifier.
        """
        if '_' in s:
            jid_str, tid_str = s.split('_', 1)
            return cls(id=int(jid_str), task_id=int(tid_str))
        return cls(id=int(s))

    def __hash__(self) -> int:
        return hash((self.id, self.task_id))

    def __str__(self) -> str:
        if self.task_id is not None:
            return f"{self.id}_{self.task_id}"
        return str(self.id)

class JobOutput(NamedTuple):
    """Output and error files associated with a submitted job.

    Attributes:
        id: Job identifier.
        output: Path to the output file.
        error: Path to the error file.
    """
    id          : JobID
    output      : str
    error       : str

@dataclass
class JobStatus:
    """Status record returned by SLURM.

    Attributes:
        id: Job identifier.
        partition: SLURM partition.
        name: Job name.
        hostname: First node in the reported node list.
        user: Submitting user.
        state: SLURM state string.
        time_used: Elapsed run time.
        nodes: Number of allocated nodes.
        job_id_raw: Unique SLURM job id for this allocation.
    """
    id          : JobID
    partition   : str
    name        : str
    hostname    : str
    user        : str
    state       : str
    time_used   : str
    nodes       : int
    job_id_raw  : int

    def format_filename(self, pattern: str) -> str:
        """Format a SLURM output or error filename pattern.

        Supports common SLURM placeholders: ``%j``, ``%J``, ``%A``, ``%a``,
        ``%N``, ``%s``, ``%u``, ``%x``, and ``%%``. Unknown placeholders are
        left unchanged.

        Args:
            pattern: Filename pattern to format.

        Returns:
            Formatted filename.
        """
        task_id = str(self.id.task_id) if self.id.task_id is not None else str()
        mapping = {'j': str(self.job_id_raw), 'J': str(self.job_id_raw),
                   'A': str(self.id.id), 'a': task_id, 'N': self.hostname,
                   's': str(self.job_id_raw), 'u': self.user, 'x': self.name}

        def repl(m: re.Match[str]) -> str:
            ch = m.group(1)
            if ch == '%':
                return '%'
            return mapping.get(ch, '%' + ch)

        return re.sub(r'%(.)', repl, pattern)

@dataclass
class SLURMJobManager:
    """Submit, query, stream, and wait for SLURM jobs.

    Uses ``sbatch`` for submission, ``squeue`` for live job state, ``sacct``
    for accounting records, and ``scancel`` for cancellation.

    Attributes:
        config: Names or paths of the SLURM command-line tools.
    """
    config      : SLURMConfig = SLURMConfig()
    completed   : ClassVar[str] = "COMPLETED"
    failed      : ClassVar[Set[str]] = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
                                        "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "PREEMPTED"}
    pending     : ClassVar[str] = "PENDING"
    running     : ClassVar[Set[str]] = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}

    @staticmethod
    def _parse_job_ids(lines: List[str]) -> List[JobID]:
        job_ids: List[JobID] = []
        seen: Set[JobID] = set()

        for line in lines:
            value = line.strip()
            if not re.fullmatch(r"\d+(?:_\d+)?", value):
                continue

            job_id = JobID.from_string(value)
            if job_id not in seen:
                job_ids.append(job_id)
                seen.add(job_id)

        return sorted(job_ids, key=lambda jid: (jid.id, -1 if jid.task_id is None else jid.task_id))

    @staticmethod
    def _parse_status_row(values: List[str]) -> JobStatus | None:
        if len(values) < 8:
            raise RuntimeError(f"Unexpected SLURM status output: {values}")

        jid, partition, name, nodelist, user, state, time_used, nodes = values[:8]
        try:
            job_id = JobID.from_string(jid)
            job_id_raw = int(values[8]) if len(values) > 8 and values[8] else job_id.id
        except ValueError:
            return None

        hostname = (nodelist.split(',')[0].strip() if nodelist else '')
        return JobStatus(id=job_id, partition=partition, name=name,
                         hostname=hostname, user=user, state=state, time_used=time_used,
                         nodes=int(nodes), job_id_raw=job_id_raw)

    @staticmethod
    def _split_slurm_rows(output: str) -> List[List[str]]:
        rows: List[List[str]] = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                rows.append(line.split('|'))
        return rows

    @staticmethod
    def _base_job_ids(job_ids: List[JobID]) -> str:
        return ','.join(str(job_id) for job_id in sorted({job.id for job in job_ids}))

    def _squeue_statuses(self, job_ids: List[JobID]) -> Dict[JobID, JobStatus]:
        if not job_ids:
            return {}

        proc = subprocess.run(
            [self.config.squeue, "-j", self._base_job_ids(job_ids), "-r", "-h",
             "-o", "%i|%P|%j|%N|%u|%T|%M|%D|%A"],
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"squeue failed: {proc.stderr.strip()}")

        statuses: Dict[JobID, JobStatus] = {}
        for values in self._split_slurm_rows(proc.stdout):
            status = self._parse_status_row(values)
            if status is not None:
                statuses[status.id] = status
        return statuses

    def _sacct_statuses(self, job_ids: List[JobID]) -> Dict[JobID, JobStatus]:
        if not job_ids:
            return {}

        proc = subprocess.run(
            [self.config.sacct, "-j", self._base_job_ids(job_ids), "-n", "-P", "-X",
             "--format=JobID,Partition,JobName,NodeList,User,State,Elapsed,NNodes,JobIDRaw"],
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"sacct failed: {proc.stderr.strip()}")

        statuses: Dict[JobID, JobStatus] = {}
        for values in self._split_slurm_rows(proc.stdout):
            status = self._parse_status_row(values)
            if status is not None:
                statuses[status.id] = status
        return statuses

    def _get_status_batch(self, job_ids: List[JobID]) -> List[JobStatus | None]:
        statuses: Dict[JobID, JobStatus] = {}

        try:
            statuses.update(self._squeue_statuses(job_ids))
        except (FileNotFoundError, RuntimeError):
            pass

        missing = [job_id for job_id in job_ids if job_id not in statuses]
        if missing:
            try:
                for job_id, status in self._sacct_statuses(missing).items():
                    statuses.setdefault(job_id, status)
            except FileNotFoundError as exc:
                raise RuntimeError("sacct not found") from exc

        return [statuses.get(job_id) for job_id in job_ids]

    def get_job_id(self, job_id: int) -> JobID | List[JobID] | None:
        """Return SLURM job identifiers for a base job id.

        For regular jobs this returns a single :class:`JobID`. For array jobs
        it returns the expanded array task IDs. Accounting data from ``sacct``
        is preferred so completed array tasks are included alongside running
        ones. Returns None when neither ``sacct`` nor ``squeue`` knows about
        the requested id.

        Args:
            job_id: Base SLURM job id.

        Returns:
            Matching job id, expanded array task ids, or ``None`` if the job is
            not found.

        Raises:
            RuntimeError: If a SLURM command fails or no usable query command is
            available.
        """
        jobs: List[JobID] = []
        try:
            proc = subprocess.run(
                [self.config.sacct, "-j", str(job_id), "-n", "-P", "-X",
                 "--format=JobID"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            pass
        else:
            if proc.returncode != 0:
                raise RuntimeError(f"sacct failed: {proc.stderr.strip()}")
            jobs = self._parse_job_ids(proc.stdout.splitlines())

        if not jobs:
            try:
                proc = subprocess.run(
                    [self.config.squeue, "-j", str(job_id), "-r", "-h", "-o", "%i"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("squeue not found") from exc

            if proc.returncode != 0:
                raise RuntimeError(f"squeue failed: {proc.stderr.strip()}")

            jobs = self._parse_job_ids(proc.stdout.splitlines())

        if not jobs:
            return None
        if len(jobs) == 1:
            return jobs[0]
        return jobs

    def cancel(self, job_id: JobID) -> None:
        """Cancel a SLURM job.

        Args:
            job_id: Job to cancel.
        """
        subprocess.run([self.config.scancel, str(job_id)], check=False)

    def squeue(self, job_id: JobID, formatter: str="%i|%j|%T|%M") -> List[str]:
        """Query ``squeue`` for one job.

        Args:
            job_id: Job to query.
            formatter: ``squeue`` output format string.

        Returns:
            Fields from the first non-empty output row, split on ``"|"``.

        Raises:
            RuntimeError: If ``squeue`` is unavailable or exits with an error.
        """
        try:
            proc = subprocess.run(
                [self.config.squeue, "-j", str(job_id), "-h", "-o", formatter],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("squeue not found") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"squeue failed: {proc.stderr.strip()}")

        out = proc.stdout.strip()
        if not out:
            return []

        for line in out.splitlines():
            line = line.strip()
            if line:
                return line.split('|')

        return []

    async def squeue_async(self, job_id: JobID, formatter: str="%i|%j|%T|%M") -> List[str]:
        """Asynchronously query ``squeue`` for one job.

        Args:
            job_id: Job to query.
            formatter: ``squeue`` output format string.

        Returns:
            Fields from the first non-empty output row, split on ``"|"``.

        Raises:
            RuntimeError: If ``squeue`` is unavailable or exits with an error.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.squeue, "-j", str(job_id), "-h", "-o", formatter,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

        except FileNotFoundError as exc:
            raise RuntimeError("squeue not found") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"squeue failed: {stderr.decode().strip()}")

        out = stdout.decode().strip()
        if not out:
            return []

        for line in out.splitlines():
            line = line.strip()
            if line:
                return line.split('|')

        return []

    def sacct(self, job_id: JobID, formatter: str="JobID,JobName,State,Elapsed") -> List[str]:
        """Query ``sacct`` for one job.

        Args:
            job_id: Job to query.
            formatter: Comma-separated ``sacct`` format fields.

        Returns:
            Fields from the first non-empty output row, split on ``"|"``.

        Raises:
            RuntimeError: If ``sacct`` is unavailable or exits with an error.
        """
        try:
            proc = subprocess.run(
                [self.config.sacct, "-j", str(job_id), "-n", "-P", f"--format={formatter}"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("sacct not found") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"sacct failed: {proc.stderr.strip()}")

        out = proc.stdout.strip()
        if not out:
            return []

        for line in out.splitlines():
            line = line.strip()
            if line:
                return line.split('|')

        return []

    async def sacct_async(self, job_id: JobID, formatter: str="JobID,JobName,State,Elapsed"
                          ) -> List[str]:
        """Asynchronously query ``sacct`` for one job.

        Args:
            job_id: Job to query.
            formatter: Comma-separated ``sacct`` format fields.

        Returns:
            Fields from the first non-empty output row, split on ``"|"``.

        Raises:
            RuntimeError: If ``sacct`` is unavailable or exits with an error.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.sacct, "-j", str(job_id), "-n", "-P", f"--format={formatter}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

        except FileNotFoundError as exc:
            raise RuntimeError("sacct not found") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"sacct failed: {stderr.decode().strip()}")

        out = stdout.decode().strip()
        if not out:
            return []

        for line in out.splitlines():
            line = line.strip()
            if line:
                return line.split('|')

        return []

    def get_output(self, script: SLURMScript, job_id: JobID, poll_interval: float=0.5
                   ) -> JobOutput | None:
        """Resolve output and error filenames for a job.

        Pending jobs are polled until SLURM reports enough status information
        to expand the output and error filename patterns.

        Args:
            script: Script used to submit the job.
            job_id: Job to inspect.
            poll_interval: Delay in seconds between pending-state polls.

        Returns:
            Resolved output and error paths, or ``None`` if the job status
            cannot be resolved.
        """
        state = self.get_state(job_id)
        while state is not None and state.upper() == self.pending:
            sleep(poll_interval)
            state = self.get_state(job_id)
            if state is None or state.upper() != self.pending:
                break

        status = self.get_status(job_id)
        if status is not None and status.hostname:
            output_file = status.format_filename(script.parameters.output)
            error_file = status.format_filename(script.parameters.error)
            return JobOutput(job_id, output_file, error_file)
        return None

    @overload
    def get_status(self, job_id: JobID) -> JobStatus | None:
        ...

    @overload
    def get_status(self, job_id: List[JobID]) -> List[JobStatus | None]:
        ...

    def get_status(self, job_id: JobID | List[JobID]) -> JobStatus | List[JobStatus | None] | None:
        """Return SLURM status for one job or a batch of jobs.

        For list input, live ``squeue`` rows are preferred and missing rows are
        filled from ``sacct`` so completed array tasks can be resolved in one
        accounting query per base job set.

        Args:
            job_id: Job or jobs to query.

        Returns:
            Status for a single job, or a list aligned to the input jobs. A
            missing job is represented by ``None``.

        Raises:
            RuntimeError: If SLURM output cannot be parsed or a required query
                command fails.
        """
        if isinstance(job_id, list):
            return self._get_status_batch(job_id)

        statuses = self._get_status_batch([job_id])
        return statuses[0] if statuses else None

    async def get_status_async(self, job_id: JobID) -> JobStatus | None:
        """Asynchronously return status information for one job.

        Args:
            job_id: Job to query.

        Returns:
            Parsed status record, or ``None`` if the job is not found.

        Raises:
            RuntimeError: If SLURM output cannot be parsed or a query command
                fails.
        """
        # id | partition | name | nodelist | user | state | time_used | nodes | raw_id
        formatter = "JobID,Partition,JobName,NodeList,User,State,Elapsed,NNodes,JobIDRaw"

        try:
            values = await self.squeue_async(job_id, formatter="%i|%P|%j|%N|%u|%T|%M|%D|%A")
        except RuntimeError:
            values = await self.sacct_async(job_id, formatter=formatter)
        else:
            if not values:
                values = await self.sacct_async(job_id, formatter=formatter)

        if not values:
            return None

        if len(values) < 8:
            raise RuntimeError(f"Unexpected squeue output: {values}")
        jid, partition, name, nodelist, user, state, time_used, nodes = values[:8]
        parsed_id = JobID.from_string(jid)
        job_id_raw = int(values[8]) if len(values) > 8 and values[8] else parsed_id.id
        hostname = (nodelist.split(',')[0].strip() if nodelist else '')
        return JobStatus(id=parsed_id, partition=partition, name=name,
                         hostname=hostname, user=user, state=state, time_used=time_used,
                         nodes=int(nodes), job_id_raw=job_id_raw)

    def get_state(self, job_id: JobID) -> str | None:
        """Return the SLURM state string for one job.

        Args:
            job_id: Job to query.

        Returns:
            SLURM state string, or ``None`` if the job is not found.
        """
        try:
            values = self.squeue(job_id, formatter="%T")
        except RuntimeError:
            values = self.sacct(job_id, formatter="State")
        else:
            if not values:
                values = self.sacct(job_id, formatter="State")

        if not values:
            return None

        return values[0]

    async def get_state_async(self, job_id: JobID) -> str | None:
        """Asynchronously return the SLURM state string for one job.

        Args:
            job_id: Job to query.

        Returns:
            SLURM state string, or ``None`` if the job is not found.
        """
        try:
            values = await self.squeue_async(job_id, formatter="%T")
        except RuntimeError:
            values = await self.sacct_async(job_id, formatter="State")
        else:
            if not values:
                values = await self.sacct_async(job_id, formatter="State")

        if not values:
            return None

        return values[0]

    def is_running(self, job_id: JobID) -> bool:
        """Return whether a job is still active.

        Args:
            job_id: Job to query.

        Returns:
            ``True`` for pending, running, configuring, or completing jobs.
        """
        status = self.get_state(job_id)
        if status is None:
            return False

        return status.upper() in self.running

    async def is_running_async(self, job_id: JobID) -> bool:
        """Asynchronously return whether a job is still active.

        Args:
            job_id: Job to query.

        Returns:
            ``True`` for pending, running, configuring, or completing jobs.
        """
        status = await self.get_state_async(job_id)
        if status is None:
            return False

        return status.upper() in self.running

    async def stream_job(self, job: JobOutput, poll_interval: float = 0.1
                         ) -> AsyncGenerator[str, None]:
        """Stream lines from a job output file.

        Args:
            job: Job output descriptor returned by :meth:`get_output`.
            poll_interval: Delay in seconds between output file checks.

        Yields:
            New output lines, including newline characters.
        """
        async def read_lines(path: str, pos: int) -> Tuple[List[str], int]:
            """Read new lines from file asynchronously using a thread."""
            def read() -> Tuple[List[str], int]:
                with open(path, 'rb') as f:
                    f.seek(pos)
                    lines = f.readlines()
                    new_pos = f.tell()
                return [line.decode() for line in lines], new_pos

            return await asyncio.to_thread(read)

        pos = 0
        while not os.path.exists(job.output):
            if not await self.is_running_async(job.id):
                return
            await asyncio.sleep(poll_interval)

        while await self.is_running_async(job.id) or os.path.exists(job.output):
            if os.path.exists(job.output):
                new_lines, pos = await read_lines(job.output, pos)
                for line in new_lines:
                    yield line

            if not await self.is_running_async(job.id):
                # Flush remaining lines one last time
                new_lines, pos = await read_lines(job.output, pos)
                for line in new_lines:
                    yield line
                break

            await asyncio.sleep(poll_interval)

    def submit(self, slurm_script: SLURMScript) -> JobID:
        """Submit one SLURM batch job.

        The provided command is run with ``bash -lc <command>`` inside the job
        script so shell expansions and quoting behave as the user expects.

        Args:
            slurm_script: Script specification to submit.

        Returns:
            Submitted job id.

        Raises:
            RuntimeError: If ``sbatch`` fails or its output does not contain a
                job id.
        """
        with slurm_script.write_file() as script_file:
            result = subprocess.run([self.config.sbatch, script_file.name],
                                    capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

            # Expect output like: Submitted batch job 12345
            m = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not m:
                raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
            return JobID.from_string(m.group(1))

    def submit_all(self, scripts: List[SLURMScript], wait: bool=True, poll_interval: float = 0.5,
                   desc: str = "SLURM Jobs") -> List[JobID]:
        """Submit multiple batch jobs.

        Args:
            scripts: Scripts to submit.
            wait: If ``True``, wait until all submitted jobs finish.
            poll_interval: Delay in seconds between status polls while waiting.
            desc: Progress-bar description.

        Returns:
            Submitted job ids.
        """
        jobs = [self.submit(script) for script in scripts]
        if wait:
            self.wait_all(jobs, poll_interval=poll_interval, desc=desc)
        return jobs

    def submit_array(self, script: SLURMScript, task_ids: List[int] | range,
                     n_tasks: int | None=None, wait: bool=True, poll_interval: float = 0.5,
                     desc: str = "SLURM array") -> List[JobID]:
        """Submit a SLURM array job.

        Args:
            script: Script specification to submit as an array.
            task_ids: Array task ids.
            n_tasks: Maximum number of simultaneously running array tasks.
            wait: If ``True``, wait until all array tasks finish.
            poll_interval: Delay in seconds between status polls while waiting.
            desc: Progress-bar description.

        Returns:
            Submitted array task ids.

        Raises:
            ValueError: If ``task_ids`` is not a list or range.
            RuntimeError: If ``sbatch`` fails or its output does not contain a
                job id.
        """
        if isinstance(task_ids, list):
            array_string = ','.join(str(tid) for tid in task_ids)
        elif isinstance(task_ids, range):
            array_string = f"{task_ids.start}-{task_ids.stop - 1}:{task_ids.step}"
        else:
            raise ValueError("task_ids must be a list or range")
        if n_tasks is not None:
            array_string += f"%{n_tasks}"

        with script.write_file() as script_file:
            result = subprocess.run([self.config.sbatch, f"--array={array_string}",
                                     script_file.name],
                                    capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

            # Expect output like: Submitted batch job 12345
            m = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not m:
                raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
            job_id = int(m.group(1))
            jobs = [JobID(id=job_id, task_id=tid) for tid in task_ids]

        if wait:
            self.wait_all(jobs, poll_interval=poll_interval, desc=desc)
        return jobs

    def wait_all(self, job_ids: List[JobID], poll_interval: float = 0.1,
                 desc: str = "SLURM Jobs") -> None:
        """Wait until all jobs finish successfully.

        Args:
            job_ids: Jobs to wait for.
            poll_interval: Delay in seconds between status polls.
            desc: Progress-bar description.

        Raises:
            RuntimeError: If a job is not found or enters a failed state.
        """
        pending: List[JobID] = list(job_ids)
        completed: Set[JobID] = set()

        with tqdm(total=len(job_ids), desc=desc, unit="job") as pbar:
            while pending:
                still_pending: List[JobID] = []
                finished = 0
                statuses = self.get_status(pending)
                for job_id, status in zip(pending, statuses):
                    if status is None:
                        raise RuntimeError(f"Job {job_id} not found in squeue or sacct")

                    state = status.state
                    if state.upper() == self.completed:
                        if job_id not in completed:
                            completed.add(job_id)
                            finished += 1
                    # Check if the job failed
                    elif state.upper() in self.failed:
                        raise RuntimeError(f"Job {job_id} failed with state: {state}")
                    else:
                        still_pending.append(job_id)

                if finished:
                    pbar.update(finished)
                pending = still_pending
                if pending:
                    sleep(poll_interval)
