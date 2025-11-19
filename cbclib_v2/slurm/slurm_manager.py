"""Simple SLURM job submission and monitoring wrapper using subprocess.

This module provides `SLURMJobManager` which can submit a shell command as
an SBATCH job, poll SLURM for status, wait for completion, and cancel jobs.

It is intentionally lightweight (no external dependencies) and uses
``subprocess`` to call `sbatch`, `squeue`, `sacct` and `scancel`.
"""
import asyncio
from contextlib import contextmanager
import os
import re
from time import sleep
from shlex import quote
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper as TemporaryFileWrapper
import subprocess
from typing import AsyncGenerator, ClassVar, Dict, Iterator, List, NamedTuple, Set, Tuple
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from cbclib_v2._src.parser import Parser, get_parser
from cbclib_v2._src.data_container import Container

class SLURMConfig(NamedTuple):
    """NamedTuple holding the paths/names of SLURM binaries.

    Using a NamedTuple keeps the init arguments grouped and typed while
    remaining lightweight and tuple-like.
    """
    sbatch  : str = "sbatch"
    squeue  : str = "squeue"
    sacct   : str = "sacct"
    scancel : str = "scancel"

@dataclass
class ScriptSpec(Container):
    partition       : str = ''
    time            : str = "01:00:00"
    nodes           : int = 1
    chdir           : str = ''
    mem             : str = '0'
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
        """Return True if the value contains shell variable or command substitutions.
        """
        return bool(cls.shell_pattern.search(value))

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        return get_parser(file_or_extension, cls, 'parameters')

    @classmethod
    def read(cls, file: str) -> 'ScriptSpec':
        return cls.from_dict(**cls.parser(file).read(file))

    def add_define(self, key: str, value: str) -> None:
        self.define_macros[key] = value

    def script_header(self) -> List[str]:
        header: List[str] = []
        if self.partition:
            header.append(f"#SBATCH --partition={self.partition}\n")
        header.append(f"#SBATCH --time={self.time}\n")
        header.append(f"#SBATCH --nodes={self.nodes}\n")
        header.append(f"#SBATCH --chdir={self.chdir}\n")
        header.append(f"#SBATCH --mem={self.mem}\n")
        header.append("#SBATCH --open-mode=append\n")
        header.append(f"#SBATCH --output={self.output}\n")
        header.append(f"#SBATCH --error={self.error}\n")
        return header

    def script_body(self) -> List[str]:
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
    """Dataclass encapsulating arguments for submitting a job.

    The fields provide sensible defaults for common SLURM parameters so a
    caller only needs to set the command (and optionally a handful of
    overrides).
    """
    command         : str
    job_name        : str
    parameters      : ScriptSpec = ScriptSpec()

    def script_header(self) -> List[str]:
        header = self.parameters.script_header()
        header.append(f"#SBATCH --job-name={self.job_name}\n")

        return header

    def script_body(self) -> List[str]:
        body = self.parameters.script_body()
        body.append(f"bash -lc {quote(self.command)}\n")
        return body

    @contextmanager
    def write_file(self, directory: str | os.PathLike[str] | None=None
                   ) -> Iterator[TemporaryFileWrapper]:
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
    id      : int
    task_id : int | None = None

    @classmethod
    def from_string(cls, s: str) -> 'JobID':
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
    id          : JobID
    output      : str
    error       : str

@dataclass
class JobStatus:
    id          : JobID
    partition   : str
    name        : str
    hostname    : str
    user        : str
    state       : str
    time_used   : str
    nodes       : int

    def format_filename(self, pattern: str) -> str:
        """Format SLURM output/error filename with job status fields.

        Supports common SLURM placeholders:
        - %j: job id
        - %J: job or job step id
        - %N: hostname
        - %s: job or job step id
        - %u: user name
        - %x: job name
        A literal percent can be written as %%.

        Unknown placeholders are left unchanged.

        Args:
            pattern: The filename pattern to format.

        Returns:
            The formatted filename.
        """
        mapping = {'j': str(self.id), 'J': str(self.id), 'N': self.hostname,
                   's': str(self.id), 'u': self.user, 'x': self.name}

        def repl(m: re.Match[str]) -> str:
            ch = m.group(1)
            if ch == '%':
                return '%'
            return mapping.get(ch, '%' + ch)

        return re.sub(r'%(.)', repl, pattern)

@dataclass
class SLURMJobManager:
    """Submit and monitor SLURM jobs.

    This lightweight helper writes a small SBATCH script, submits it with
    ``sbatch``, and can poll ``squeue``/``sacct`` to determine job state.
    """
    config      : SLURMConfig = SLURMConfig()
    completed   : ClassVar[str] = "COMPLETED"
    failed      : ClassVar[Set[str]] = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
                                        "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "PREEMPTED"}
    pending     : ClassVar[str] = "PENDING"
    running     : ClassVar[Set[str]] = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}

    def cancel(self, job_id: JobID) -> None:
        """Cancel a SLURM job via scancel.
        """
        subprocess.run([self.config.scancel, str(job_id)], check=False)

    def squeue(self, job_id: JobID, formatter: str="%i|%j|%T|%M") -> List[str]:
        """Call squeue with a custom format and return output lines."""
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
        """Call squeue with a custom format and return output lines asynchronously."""
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
        """Call sacct with a custom format and return output lines."""
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
        """Call sacct with a custom format and return output lines asynchronously."""
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
        """Get the output and error files for a SLURM job.

        Args:
            script: The SLURM script used to submit the job.
            job_id: The ID of the job to retrieve output for.

        Returns:
            A JobOutput named tuple containing the job ID, output, and error.
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

    def get_status(self, job_id: JobID) -> JobStatus | None:
        """Synchronous version of get_status using squeue with a fixed format.

        Returns None if the job is not listed by squeue.
        """
        # id | partition | name | nodelist | user | state | time_used | nodes
        formatter = "JobIDRaw,Partition,JobName,NodeList,User,State,Elapsed,NNodes"

        try:
            values = self.squeue(job_id, formatter="%A|%P|%j|%N|%u|%T|%M|%D")
        except RuntimeError:
            values = self.sacct(job_id, formatter=formatter)
        else:
            if not values:
                values = self.sacct(job_id, formatter=formatter)

        if not values:
            return None

        if len(values) < 8:
            raise RuntimeError(f"Unexpected squeue output: {values}")
        jid, partition, name, nodelist, user, state, time_used, nodes = values[:8]
        hostname = (nodelist.split(',')[0].strip() if nodelist else '')
        return JobStatus(id=JobID.from_string(jid), partition=partition, name=name,
                         hostname=hostname, user=user, state=state, time_used=time_used,
                         nodes=int(nodes))

    async def get_status_async(self, job_id: JobID) -> JobStatus | None:
        """Return rich job status information using `squeue`.

        Queries squeue for a single job id with a pipe-delimited format and
        parses it into a JobStatus tuple. Returns None if the job is not found
        or if squeue yields no rows.
        """
        # id | partition | name | nodelist | user | state | time_used | nodes
        formatter = "JobID,Partition,JobName,NodeList,User,State,Elapsed,NNodes"

        try:
            values = await self.squeue_async(job_id, formatter="%i|%P|%j|%N|%u|%T|%M|%D")
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
        hostname = (nodelist.split(',')[0].strip() if nodelist else '')
        return JobStatus(id=JobID.from_string(jid), partition=partition, name=name,
                         hostname=hostname, user=user, state=state, time_used=time_used,
                         nodes=int(nodes))

    def get_state(self, job_id: JobID) -> str | None:
        """Query squeue and sacct for job status. Returns a short string or None.

        Prefers `squeue` for live jobs and falls back to `sacct` for finished
        jobs. Returns None if neither utility is available or returns no data.
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
        """Query squeue and sacct for job status. Returns a short string or None.

        Prefers `squeue` for live jobs and falls back to `sacct` for finished
        jobs. Returns None if neither utility is available or returns no data.
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
        """Return True if SLURM job is still running or pending.
        """
        status = self.get_state(job_id)
        if status is None:
            return False

        return status.upper() in self.running

    async def is_running_async(self, job_id: JobID) -> bool:
        """Return True if SLURM job is still running or pending.
        """
        status = await self.get_state_async(job_id)
        if status is None:
            return False

        return status.upper() in self.running

    async def stream_job(self, job: JobOutput, poll_interval: float = 0.1
                         ) -> AsyncGenerator[str, None]:
        """Yield lines from SLURM job output as they appear, until job finishes.
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
        """Submit a command to SLURM and return the job id.

        The provided command is run with ``bash -lc <command>`` inside the job
        script so shell expansions and quoting behave as the user expects.
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
        """Submit scripts and block synchronously until all finish using wait_all."""
        jobs = [self.submit(script) for script in scripts]
        if wait:
            self.wait_all(jobs, poll_interval=poll_interval, desc=desc)
        return jobs

    def submit_array(self, script: SLURMScript, task_ids: List[int] | range,
                     n_tasks: int | None=None, wait: bool=True, poll_interval: float = 0.5,
                     desc: str = "SLURM array") -> List[JobID]:
        """Submit a SLURM array job."""
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

    def wait_all(self, job_ids: List[JobID], poll_interval: float = 0.5,
                 desc: str = "SLURM Jobs") -> None:
        """Blocking waiter that polls get_state_sync and shows a tqdm progress bar.

        Raises:
            RuntimeError: If any job fails (enters a failed state).
        """
        pending: List[JobID] = list(job_ids)
        completed: Set[JobID] = set()

        with tqdm(total=len(job_ids), desc=desc, unit="job") as pbar:
            while pending:
                still_pending: List[JobID] = []
                finished = 0
                for job_id in pending:
                    state = self.get_state(job_id)
                    if state is None:
                        raise RuntimeError(f"Job {job_id} not found in squeue or sacct")

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
