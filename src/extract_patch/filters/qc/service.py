from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ...concurrency import _terminate_if_parent_dies
from .runtime import (
    QcClient,
    QcFilterSetupError,
    QcSession,
    resolve_checkpoint,
)

DEFAULT_MIN_FREE_BYTES = 1_073_741_824
DEFAULT_BATCH_SIZE = 128
DEFAULT_COLLECT_TIMEOUT_MS = 8


@dataclass(frozen=True)
class _GpuMemory:
    physical_index: int
    uuid: str
    pci_bus_id: str
    free: int
    total: int
    logical_index: int = -1


def qc_is_enabled(config: Any) -> bool:
    return "qc" in tuple(getattr(config, "post_filter_pipe", ()) or ())


def parse_device_request(value: Any) -> str:
    if value in (None, "", "null"):
        return "auto"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"cuda:{int(value)}"
    text = str(value).strip()
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _query_gpu_memory_nvml() -> list[_GpuMemory]:
    import pynvml

    pynvml.nvmlInit()
    try:
        devices: list[_GpuMemory] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            devices.append(
                _GpuMemory(
                    physical_index=index,
                    uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
                    pci_bus_id=_text(pci.busId),
                    free=int(memory.free),
                    total=int(memory.total),
                )
            )
        return devices
    finally:
        pynvml.nvmlShutdown()


def _query_gpu_memory_smi() -> list[_GpuMemory]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    devices: list[_GpuMemory] = []
    mib = 1024 * 1024
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not line.strip():
            continue
        if len(fields) != 5:
            raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
        index, uuid, pci_bus_id, free_mib, total_mib = fields
        devices.append(
            _GpuMemory(
                physical_index=int(index),
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                free=int(free_mib) * mib,
                total=int(total_mib) * mib,
            )
        )
    return devices


def _query_gpu_memory() -> list[_GpuMemory]:
    errors: list[str] = []
    try:
        return _query_gpu_memory_nvml()
    except Exception as exc:
        errors.append(f"NVML: {type(exc).__name__}: {exc}")
    try:
        return _query_gpu_memory_smi()
    except Exception as exc:
        errors.append(f"nvidia-smi: {type(exc).__name__}: {exc}")
    raise QcFilterSetupError(
        "Unable to inspect GPU memory without CUDA contexts ("
        + "; ".join(errors)
        + ")"
    )


def _matches_visible_token(device: _GpuMemory, token: str) -> bool:
    if token.isdigit():
        return device.physical_index == int(token)
    token_lower = token.lower()
    return device.uuid.lower().startswith(token_lower)


def _visible_gpu_memory() -> list[_GpuMemory]:
    devices = _query_gpu_memory()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        mapped: list[_GpuMemory] = []
        for logical_index, token in enumerate(tokens):
            match = next(
                (device for device in devices if _matches_visible_token(device, token)),
                None,
            )
            if match is None:
                continue
            mapped.append(
                _GpuMemory(
                    physical_index=match.physical_index,
                    uuid=match.uuid,
                    pci_bus_id=match.pci_bus_id,
                    free=match.free,
                    total=match.total,
                    logical_index=logical_index,
                )
            )
        return mapped

    ordered = devices
    if os.environ.get("CUDA_DEVICE_ORDER", "").upper() == "PCI_BUS_ID":
        ordered = sorted(devices, key=lambda device: device.pci_bus_id.lower())
    return [
        _GpuMemory(
            physical_index=device.physical_index,
            uuid=device.uuid,
            pci_bus_id=device.pci_bus_id,
            free=device.free,
            total=device.total,
            logical_index=logical_index,
        )
        for logical_index, device in enumerate(ordered)
    ]


def pick_gpu(min_free_bytes: int) -> tuple[Any | None, str | None]:
    try:
        devices = _visible_gpu_memory()
    except QcFilterSetupError as exc:
        return None, f"{exc}; using CPU for the qc filter"
    if not devices:
        return None, "No CUDA-visible GPU is available; using CPU for the qc filter"
    eligible: list[tuple[int, int, int]] = []
    details: list[str] = []
    for device in devices:
        details.append(
            f"cuda:{device.logical_index} free={device.free / 1e9:.2f}G "
            f"total={device.total / 1e9:.2f}G"
        )
        if device.free >= min_free_bytes:
            eligible.append((device.free, device.total, device.logical_index))
    if not eligible:
        return None, (
            "No GPU has enough free memory for the qc filter "
            f"(need {min_free_bytes / 1e9:.2f}G; {'; '.join(details)}); using CPU"
        )
    eligible.sort(reverse=True)
    _free, _total, index = eligible[0]
    return f"cuda:{index}", None


def select_device(requested: Any, min_free_bytes: int) -> tuple[Any, str | None]:
    try:
        import torch
    except ImportError as exc:
        raise QcFilterSetupError(
            "The qc post-filter requires PyTorch. Install with: pip install torch torchvision"
        ) from exc
    name = parse_device_request(requested)
    if name == "cpu":
        return torch.device("cpu"), None
    if name in {"auto", "cuda"}:
        device_name, warning = pick_gpu(min_free_bytes)
        if device_name is None:
            if name == "cuda":
                raise QcFilterSetupError(warning or "CUDA was requested but no GPU is usable")
            return torch.device("cpu"), warning
        return torch.device(device_name), warning
    device = torch.device(name)
    if device.type != "cuda":
        return device, None
    index = 0 if device.index is None else int(device.index)
    try:
        devices = _visible_gpu_memory()
    except QcFilterSetupError as exc:
        raise QcFilterSetupError(f"Cannot validate requested qc device {device}: {exc}") from exc
    memory = next(
        (item for item in devices if item.logical_index == index),
        None,
    )
    if memory is None:
        raise QcFilterSetupError(f"Requested qc device {device} does not exist")
    if memory.free < min_free_bytes:
        raise QcFilterSetupError(
            f"Requested {device} has {memory.free / 1e9:.2f}G free, "
            f"need {min_free_bytes / 1e9:.2f}G (total {memory.total / 1e9:.2f}G)"
        )
    return torch.device(f"cuda:{index}"), None


def _infer_numpy(session: QcSession, batch, max_batch: int) -> list[float]:
    scores: list[float] = []
    start = 0
    while start < len(batch):
        end = min(start + max_batch, len(batch))
        scores.extend(session.predict_numpy(batch[start:end]))
        start = end
    return scores


def _reply_connections(
    pending: list[tuple[Any, Any]],
    session: QcSession,
    max_batch: int,
) -> None:
    import numpy as np

    connections = [item[0] for item in pending]
    arrays = [item[1] for item in pending]
    try:
        scores = _infer_numpy(session, np.concatenate(arrays, axis=0), max_batch)
        offset = 0
        for connection, array in zip(connections, arrays):
            count = int(array.shape[0])
            connection.send(scores[offset : offset + count])
            offset += count
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        for connection in connections:
            try:
                connection.send(payload)
            except (EOFError, OSError, BrokenPipeError):
                pass


def _qc_server_entry(
    ready_queue,
    options: dict[str, Any],
    authkey: bytes,
    parent_pid: int,
) -> None:
    import threading
    from multiprocessing.connection import Listener, wait

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _terminate_if_parent_dies(parent_pid)

    listener = None
    try:
        device, warning = select_device(
            options.get("device", "auto"),
            int(options.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES)),
        )
        session = QcSession(
            resolve_checkpoint(options.get("checkpoint")),
            device,
            options.get("threshold"),
        )
        listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=authkey)
        host, port = listener.address
        ready_queue.put(
            {
                "ok": True,
                "device": str(session.device),
                "input_size": session.input_size,
                "threshold": session.threshold,
                "version": session.version,
                "warning": warning,
                "host": host,
                "port": int(port),
            }
        )
    except Exception as exc:
        ready_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        if listener is not None:
            listener.close()
        return

    max_batch = max(1, int(options.get("batch_size", DEFAULT_BATCH_SIZE)))
    collect_timeout = max(
        0.0,
        float(options.get("collect_timeout_ms", DEFAULT_COLLECT_TIMEOUT_MS)) / 1000.0,
    )
    connections: list[Any] = []
    conn_lock = threading.Lock()
    running = True

    def accept_loop() -> None:
        while running:
            try:
                connection = listener.accept()
            except (EOFError, OSError):
                return
            with conn_lock:
                connections.append(connection)

    acceptor = threading.Thread(target=accept_loop, name="extract-qc-accept", daemon=True)
    acceptor.start()
    try:
        while running:
            with conn_lock:
                current = list(connections)
            if not current:
                time.sleep(0.001)
                continue
            ready = wait(current, timeout=1.0)
            if not ready:
                continue
            pending: list[tuple[Any, Any]] = []
            counted = 0
            stale: list[Any] = []
            for connection in ready:
                try:
                    payload = connection.recv()
                except (EOFError, OSError):
                    stale.append(connection)
                    continue
                if payload is None:
                    running = False
                    break
                pending.append((connection, payload))
                counted += int(payload.shape[0])
            if stale:
                with conn_lock:
                    connections[:] = [item for item in connections if item not in stale]
            if not pending:
                if not running:
                    break
                continue
            deadline = time.monotonic() + collect_timeout
            while running and counted < max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                with conn_lock:
                    waiting = [
                        item
                        for item in connections
                        if item not in {pair[0] for pair in pending}
                    ]
                more = wait(waiting, timeout=remaining) if waiting else []
                if not more:
                    break
                for connection in more:
                    try:
                        payload = connection.recv()
                    except (EOFError, OSError):
                        with conn_lock:
                            if connection in connections:
                                connections.remove(connection)
                        continue
                    if payload is None:
                        running = False
                        break
                    pending.append((connection, payload))
                    counted += int(payload.shape[0])
            if pending:
                _reply_connections(pending, session, max_batch)
    finally:
        running = False
        try:
            listener.close()
        except Exception:
            pass
        # Unblock accept() on some platforms.
        try:
            from multiprocessing.connection import Client

            Client(("127.0.0.1", port), authkey=authkey).close()
        except Exception:
            pass
        for connection in list(connections):
            try:
                connection.close()
            except Exception:
                pass



@dataclass
class QcService:
    process: Any
    client: QcClient
    device: str
    warning: str | None = None

    def close(self) -> None:
        try:
            from multiprocessing.connection import Client

            connection = Client(
                (self.client.host, self.client.port),
                authkey=self.client.authkey,
            )
            connection.send(None)
            connection.close()
        except Exception:
            pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()


def start_qc_service(
    options: dict[str, Any] | None,
    *,
    slide_workers: int,
) -> QcService:
    cfg = dict(options or {})
    if "threshold" in cfg and cfg["threshold"] in ("", "null"):
        cfg["threshold"] = None
    elif cfg.get("threshold") is not None:
        cfg["threshold"] = float(cfg["threshold"])
    cfg.setdefault("device", "auto")
    cfg.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    cfg.setdefault("collect_timeout_ms", DEFAULT_COLLECT_TIMEOUT_MS)
    cfg.setdefault("min_free_bytes", DEFAULT_MIN_FREE_BYTES)
    _ = slide_workers

    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    authkey = os.urandom(16)
    process = context.Process(
        target=_qc_server_entry,
        args=(ready_queue, cfg, authkey, os.getpid()),
        name="extract-qc",
        daemon=True,
    )
    process.start()
    try:
        payload = ready_queue.get(timeout=120)
    except Exception as exc:
        process.terminate()
        process.join(timeout=5)
        raise QcFilterSetupError(f"QC service failed to start: {exc}") from exc
    if not payload.get("ok"):
        process.join(timeout=5)
        raise QcFilterSetupError(payload.get("error") or "QC service failed to start")
    client = QcClient(
        host=str(payload["host"]),
        port=int(payload["port"]),
        authkey=authkey,
        input_size=int(payload["input_size"]),
        threshold=float(payload["threshold"]),
        version=str(payload["version"]),
        device=str(payload["device"]),
    )
    return QcService(
        process=process,
        client=client,
        device=str(payload["device"]),
        warning=payload.get("warning"),
    )


def log_qc_service(logger: logging.Logger, service: QcService) -> None:
    if service.warning:
        logger.warning("%s", service.warning)
    logger.info(
        "qc_service device=%s version=%s threshold=%.4f input_size=%d",
        service.device,
        service.client.version,
        service.client.threshold,
        service.client.input_size,
    )


@contextmanager
def running_qc_service(
    config: Any,
    logger: logging.Logger | None = None,
) -> Iterator[QcClient | None]:
    from .runtime import bind_qc_client

    if not qc_is_enabled(config):
        bind_qc_client(None)
        yield None
        return
    service = start_qc_service(
        getattr(config, "post_filters", {}).get("qc", {}),
        slide_workers=getattr(getattr(config, "parallel", None), "slide_workers", 4) or 4,
    )
    if logger is not None:
        log_qc_service(logger, service)
    bind_qc_client(service.client)
    try:
        yield service.client
    finally:
        bind_qc_client(None)
        service.close()
