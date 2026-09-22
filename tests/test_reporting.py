from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from extract_patch.reporting import _ParentDeathReporter


def _wait_for_process_exit(pid: int, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} did not exit")


def test_reporter_is_silent_after_clean_close(tmp_path: Path) -> None:
    log_path = tmp_path / "clean.log"
    reporter = _ParentDeathReporter(log_path, os.getpid())

    reporter.close()

    assert not log_path.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="parent monitoring is Linux-only")
def test_reporter_logs_parent_sigkill_and_exits(tmp_path: Path) -> None:
    log_path = tmp_path / "killed.log"
    supervisor_code = textwrap.dedent(
        """
        import os
        import sys
        import time
        from pathlib import Path
        from extract_patch.config import AppConfig
        from extract_patch.reporting import RunLogger

        logger = RunLogger(Path(sys.argv[1]), "killed", AppConfig())
        print(logger._death_reporter.pid, flush=True)
        while True:
            time.sleep(1)
        """
    )
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
    supervisor = subprocess.Popen(
        [sys.executable, "-c", supervisor_code, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    assert supervisor.stdout is not None
    reporter_pid = int(supervisor.stdout.readline().strip())
    try:
        os.kill(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log_path.exists() and "run_aborted" in log_path.read_text():
                break
            time.sleep(0.05)
        else:
            pytest.fail("death reporter did not append the abort record")
        message = log_path.read_text()
        assert "reason=parent_process_disappeared" in message
        assert f"parent_pid={supervisor.pid}" in message
        _wait_for_process_exit(reporter_pid)
    finally:
        try:
            os.kill(reporter_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
