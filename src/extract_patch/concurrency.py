from __future__ import annotations

import ctypes
import errno
import multiprocessing as mp
import os
import queue
import signal
import threading
import traceback
from contextlib import contextmanager
from multiprocessing.connection import Connection, wait
from typing import Any, Callable, Iterable, Iterator


class WorkerSystemError(RuntimeError):
    """A worker process exited or raised outside normal per-WSI handling."""


def _terminate_if_parent_dies(expected_parent_pid: int | None = None) -> None:
    """Ask Linux to terminate this worker even if the parent receives SIGKILL.

    ``expected_parent_pid`` closes the race where the parent exits before the
    child has installed PR_SET_PDEATHSIG.  Without it, an already orphaned
    child would mistake init/systemd for its real parent and keep running.
    """
    parent_pid = (
        os.getppid() if expected_parent_pid is None else int(expected_parent_pid)
    )
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM)) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def raise_if_system_error(exc: BaseException) -> None:
    """Promote host-level failures so the supervisor stops the whole run."""
    if isinstance(exc, MemoryError):
        raise exc
    if isinstance(exc, OSError) and exc.errno in {
        errno.ENFILE,
        errno.EMFILE,
        errno.ENOMEM,
        errno.ENOSPC,
    }:
        raise exc


def _process_worker_loop(
    tasks: Any,
    results: Connection,
    worker: Callable[..., Any],
    worker_args: tuple[Any, ...],
    parent_pid: int,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _terminate_if_parent_dies(parent_pid)
    try:
        while True:
            task = tasks.get()
            if task is None:
                return
            index, item = task
            try:
                results.send(("result", index, worker(item, *worker_args)))
            except Exception:
                results.send(("error", index, traceback.format_exc()))
                return
    finally:
        results.close()


def _stop_processes(processes: list[mp.Process], timeout: float = 5.0) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=timeout)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()


def process_map(
    worker: Callable[..., Any],
    items: Iterable[Any],
    *worker_args: Any,
    max_workers: int,
    process_name: str,
    on_result: Callable[[Any], None] | None = None,
) -> list[Any]:
    """Run an iterable through a bounded process queue and reap every worker."""
    item_iterator = iter(items)
    try:
        first_item = next(item_iterator)
    except StopIteration:
        return []
    worker_count = max(1, int(max_workers))
    context = mp.get_context("fork")
    task_queue = context.Queue(maxsize=worker_count * 2)
    processes: list[mp.Process] = []
    receivers: list[Connection] = []
    senders: list[Connection] = []
    parent_pid = os.getpid()
    for index in range(worker_count):
        receiver, sender = context.Pipe(duplex=False)
        receivers.append(receiver)
        senders.append(sender)
        processes.append(
            context.Process(
                target=_process_worker_loop,
                args=(task_queue, sender, worker, worker_args, parent_pid),
                name=f"{process_name}-{index + 1}",
            )
        )

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt_parent(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    completed: dict[int, Any] = {}
    started_processes: list[mp.Process] = []
    next_item: Any = first_item
    next_index = 0
    submitted = 0
    input_exhausted = False
    sentinels_sent = 0
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        for signum in previous_handlers:
            signal.signal(signum, interrupt_parent)
        for sender in senders:
            sender.close()
        active_receivers = set(receivers)
        while (
            not input_exhausted
            or len(completed) < submitted
            or sentinels_sent < worker_count
        ):
            while not input_exhausted:
                try:
                    task_queue.put_nowait((next_index, next_item))
                except queue.Full:
                    break
                submitted += 1
                next_index += 1
                try:
                    next_item = next(item_iterator)
                except StopIteration:
                    input_exhausted = True
            while input_exhausted and sentinels_sent < worker_count:
                try:
                    task_queue.put_nowait(None)
                except queue.Full:
                    break
                sentinels_sent += 1

            if len(completed) >= submitted and input_exhausted:
                continue
            ready = wait(active_receivers, timeout=0.5) if active_receivers else []
            if not ready:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    detail = ", ".join(
                        f"{process.name} exitcode={process.exitcode}" for process in failed
                    )
                    raise WorkerSystemError(f"Worker process failed: {detail}")
                if not any(process.is_alive() for process in processes):
                    raise WorkerSystemError("All workers exited before returning every result")
                continue
            for receiver in ready:
                try:
                    kind, index, payload = receiver.recv()
                except EOFError:
                    receiver.close()
                    active_receivers.discard(receiver)
                    continue
                if kind == "error":
                    raise WorkerSystemError(
                        f"Worker task {index} escaped its WSI handler: {payload}"
                    )
                completed[index] = payload
                if on_result is not None:
                    on_result(payload)

        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            detail = ", ".join(
                f"{process.name} exitcode={process.exitcode}" for process in failed
            )
            raise WorkerSystemError(f"Worker process failed during shutdown: {detail}")
        return [completed[index] for index in range(submitted)]
    except BaseException:
        _stop_processes(started_processes)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for connection in receivers + senders:
            connection.close()
        task_queue.cancel_join_thread()
        task_queue.close()


class InflightBudget:
    """Bound both queued item count and estimated uncompressed bytes."""

    def __init__(self, max_items: int, max_bytes: int) -> None:
        if max_items <= 0 or max_bytes <= 0:
            raise ValueError("Inflight limits must be positive")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items = 0
        self._bytes = 0
        self.peak_items = 0
        self.peak_bytes = 0
        self._condition = threading.Condition()

    @contextmanager
    def reserve(self, size: int) -> Iterator[None]:
        size = max(0, min(int(size), self.max_bytes))
        with self._condition:
            self._condition.wait_for(
                lambda: self._items < self.max_items
                and (self._items == 0 or self._bytes + size <= self.max_bytes)
            )
            self._items += 1
            self._bytes += size
            self.peak_items = max(self.peak_items, self._items)
            self.peak_bytes = max(self.peak_bytes, self._bytes)
        try:
            yield
        finally:
            with self._condition:
                self._items -= 1
                self._bytes -= size
                self._condition.notify_all()
