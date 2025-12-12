from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import ClassVar, Dict, List
import pytest
from cbclib_v2.slurm import JobID, JobOutput, ScriptSpec, SLURMJobManager, SLURMScript

class TestScriptSpec:
    def test_script_body_modules(self):
        """Test body generation with module loads."""
        spec = ScriptSpec(modules=["gcc", "openmpi"])
        body = spec.script_body()
        assert "module load gcc\n" in body
        assert "module load openmpi\n" in body

    def test_script_body_env_vars(self):
        """Test body with environment variables."""
        spec = ScriptSpec(define_macros={"PATH": "/usr/bin", "TMPDIR": "/scratch"})
        body = spec.script_body()
        assert any("export PATH=" in line for line in body)
        assert any("export TMPDIR=" in line for line in body)

    def test_script_body_conda(self):
        """Test conda environment activation in body."""
        spec = ScriptSpec(conda_env="myenv", conda_source="/path/to/conda.sh")
        body = spec.script_body()
        assert "source /path/to/conda.sh\n" in body
        assert "conda activate myenv\n" in body

    def test_write_file(self, tmp_path: Path):
        """Test temporary script file creation."""
        script = SLURMScript(command="echo test", job_name="test",
                             parameters=ScriptSpec(partition="debug", modules=["gcc"]))
        with script.write_file(tmp_path) as script_file:
            content = Path(script_file.name).read_text(encoding='utf-8')
            assert "#!/bin/bash" in content
            assert "#SBATCH --partition=debug" in content
            assert "module load gcc" in content
            assert "'echo test'" in content

@dataclass
class MockOutput():
    stdout      : str = ""
    returncode  : int = 0
    stderr      : str = ""

class MockProcess:
    def __init__(self, stdout: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode

    async def communicate(self):
        return self.stdout, None

class TestJobManager:
    mock_outputs : ClassVar[Dict[str, str]] = {"sbatch": "Submitted batch job 12345",
                                               "running": "RUNNING",
                                               "completed": "COMPLETED"}

    @pytest.fixture
    def manager(self) -> SLURMJobManager:
        return SLURMJobManager()

    @pytest.fixture
    def job_id(self) -> JobID:
        return JobID(42)

    @pytest.fixture
    def script_spec(self) -> ScriptSpec:
        return ScriptSpec(time="00:05:00", partition="hour")

    def test_submit(self, manager: SLURMJobManager, monkeypatch: pytest.MonkeyPatch,
                    script_spec: ScriptSpec):
        # Test using SubmitSpec and SlurmConfig
        def mock_run(args: List[str], **kwargs) -> MockOutput:
            # emulate sbatch
            if args[0] == "sbatch":
                return MockOutput(stdout=self.mock_outputs['sbatch'], returncode=0, stderr="")
            return MockOutput(stdout="", returncode=0, stderr="")

        monkeypatch.setattr(subprocess, "run", mock_run)

        script = SLURMScript(command="echo hello", job_name="test", parameters=script_spec)
        job_id = manager.submit(script)
        assert job_id.id == 12345

    async def mock_process(self, stdout: str, returncode: int = 0) -> MockProcess:
        return MockProcess(stdout.encode(), returncode)

    @pytest.mark.asyncio
    async def test_get_state(self, manager: SLURMJobManager, job_id: JobID,
                             monkeypatch: pytest.MonkeyPatch):
        """Test get_status when job is running."""
        async def mock_run(*args: str, **kwargs) -> MockProcess:
            if ' '.join(args).startswith(' '.join((manager.config.squeue, '-j', str(job_id)))):
                return await self.mock_process(self.mock_outputs["running"])
            return await self.mock_process("")

        monkeypatch.setattr("asyncio.create_subprocess_exec", mock_run)

        status = await manager.get_state_async(job_id)
        assert status == "RUNNING"

    @pytest.mark.asyncio
    async def test_get_state_sacct_fallback(self, manager: SLURMJobManager, job_id: JobID,
                                            monkeypatch: pytest.MonkeyPatch):
        """Test is_running status check."""
        async def mock_run(*args: str, **kwargs) -> MockProcess:
            if ' '.join(args).startswith(' '.join((manager.config.squeue, '-j', str(job_id)))):
                return await self.mock_process('')
            if ' '.join(args).startswith(' '.join((manager.config.sacct, '-j', str(job_id)))):
                return await self.mock_process(self.mock_outputs["completed"])
            return await self.mock_process("")

        monkeypatch.setattr("asyncio.create_subprocess_exec", mock_run)

        status = await manager.get_state_async(job_id)
        assert status == "COMPLETED"

    @pytest.mark.asyncio
    async def test_is_running(self, job_id: JobID, monkeypatch: pytest.MonkeyPatch):
        """Test is_running status check."""
        states = ["PENDING", "RUNNING", "COMPLETED"]

        async def mock_run(*args: str, **kwargs) -> MockProcess:
            if ' '.join(args).startswith(' '.join((manager.config.squeue, '-j', str(job_id)))):
                return await self.mock_process(states.pop(0))
            return await self.mock_process("")

        monkeypatch.setattr("asyncio.create_subprocess_exec", mock_run)

        manager = SLURMJobManager()

        assert await manager.is_running_async(job_id)  # PENDING
        assert await manager.is_running_async(job_id)  # RUNNING
        assert not await manager.is_running_async(job_id)  # COMPLETED

    @pytest.fixture
    def job_output(self, job_id: JobID, tmp_path: Path) -> JobOutput:
        output_file = tmp_path / "job.out"
        output_file.write_text("line1\nline2\n")
        return JobOutput(job_id, str(output_file), "")

    @pytest.mark.asyncio
    async def test_stream_job(self, manager: SLURMJobManager, job_output: JobOutput,
                              monkeypatch: pytest.MonkeyPatch):
        """Test job output streaming."""
        # Mock is_running_async to return True once then False
        states = [True, False]
        async def mock_is_running_async(*args, **kwargs) -> bool:
            if states:
                return states.pop(0)
            return False

        monkeypatch.setattr(SLURMJobManager, "is_running_async", mock_is_running_async)

        lines = []
        async for line in manager.stream_job(job_output, poll_interval=0.1):
            lines.append(line)

        assert lines == ["line1\n", "line2\n"]
