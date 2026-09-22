from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from extract_patch.filters.qc import service


def _gpu(index: int, free_gib: int, uuid: str | None = None) -> service._GpuMemory:
    gib = 1024**3
    return service._GpuMemory(
        physical_index=index,
        uuid=uuid or f"GPU-{index}",
        pci_bus_id=f"00000000:{index:02X}:00.0",
        free=free_gib * gib,
        total=24 * gib,
    )


def test_pick_gpu_uses_memory_query_without_torch_cuda(monkeypatch) -> None:
    devices = [
        service._GpuMemory(**{**_gpu(0, 6).__dict__, "logical_index": 0}),
        service._GpuMemory(**{**_gpu(1, 18).__dict__, "logical_index": 1}),
    ]
    monkeypatch.setattr(service, "_visible_gpu_memory", lambda: devices)

    device, warning = service.pick_gpu(1024**3)

    assert device == "cuda:1"
    assert warning is None


def test_visible_gpu_memory_honors_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "_query_gpu_memory",
        lambda: [_gpu(0, 8), _gpu(2, 12), _gpu(5, 20, "GPU-abc123")],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2")

    devices = service._visible_gpu_memory()

    assert [(item.logical_index, item.physical_index) for item in devices] == [
        (0, 5),
        (1, 2),
    ]


def test_gpu_query_falls_back_to_nvidia_smi(monkeypatch) -> None:
    def unavailable():
        raise ImportError("no pynvml")

    expected = [_gpu(3, 10)]
    monkeypatch.setattr(service, "_query_gpu_memory_nvml", unavailable)
    monkeypatch.setattr(service, "_query_gpu_memory_smi", lambda: expected)

    assert service._query_gpu_memory() == expected


def test_qc_entry_installs_parent_death_signal_before_setup(monkeypatch) -> None:
    class Installed(Exception):
        pass

    def installed(parent_pid: int) -> None:
        assert parent_pid == 12345
        raise Installed

    monkeypatch.setattr(service, "_terminate_if_parent_dies", installed)

    with pytest.raises(Installed):
        service._qc_server_entry(None, {}, b"key", 12345)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only")
def test_parent_sigkill_terminates_child() -> None:
    child_code = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import time
        from extract_patch.concurrency import _terminate_if_parent_dies

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        _terminate_if_parent_dies(int(sys.argv[1]))
        while True:
            time.sleep(1)
        """
    )
    supervisor_code = textwrap.dedent(
        """
        import os
        import subprocess
        import sys
        import time

        child = subprocess.Popen([sys.executable, "-c", sys.argv[1], str(os.getpid())])
        print(child.pid, flush=True)
        while True:
            time.sleep(1)
        """
    )
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
    supervisor = subprocess.Popen(
        [sys.executable, "-c", supervisor_code, child_code],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert supervisor.stdout is not None
    child_pid = int(supervisor.stdout.readline().strip())
    try:
        os.kill(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            stat_path = Path(f"/proc/{child_pid}/stat")
            if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("child survived after its parent received SIGKILL")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
