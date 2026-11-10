"""Cooperative GPU-0 lease, using only the standard library.

The persistent lock inode must never be removed. All cooperating launchers must
use this module. An unrelated scheduler can still start a job after our check;
the monitor reports that conflict but never signals or kills any process.
"""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent  # repo root (file now lives one level down)
import sys as _sys
_sys.path.insert(0, str(ROOT))  # repo root, for analysis./inversion./clean_pipeline. imports
LOCK_PATH = ROOT / "results/clean_small/gpu0.lock"
MAX_UTILIZATION_PERCENT = 5


class GPUQueryError(RuntimeError):
    """The GPU's identity or occupancy could not be established safely."""


class GPUBusyError(RuntimeError):
    """GPU 0 or its cooperative lock is occupied."""


def query_gpu0():
    """Query both compute and graphics processes, regardless of their owner."""
    try:
        result = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-x"],
                                capture_output=True, text=True, check=True, timeout=15)
        root = ET.fromstring(result.stdout)
        gpus = root.findall("gpu")
        if root.tag != "nvidia_smi_log" or len(gpus) != 1:
            raise ValueError("Expected exactly GPU 0 from the index-restricted query")
        gpu = gpus[0]
        uuid = gpu.findtext("uuid", "").strip()
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
            raise ValueError("Missing/invalid GPU UUID")
        free = re.fullmatch(r"\s*(\d+)\s+MiB\s*", gpu.findtext("fb_memory_usage/free", ""))
        util = re.fullmatch(r"\s*(\d+)\s*%\s*", gpu.findtext("utilization/gpu_util", ""))
        if not free or not util or not 0 <= int(util[1]) <= 100:
            raise ValueError("Missing/unsupported memory or utilization reading")
        tables = gpu.findall(".//processes")
        if not tables:
            raise ValueError("Missing process table (cannot establish exclusivity)")
        for table in tables:
            if (table.text or "").strip() not in ("", "None"):
                raise ValueError("Unsupported process-table status")
            if any(child.tag != "process_info" for child in table):
                raise ValueError("Unknown process-table entry")
        processes = []
        # Do not filter by process type, user, /proc visibility, or memory usage.
        for process in gpu.findall(".//process_info"):
            pid = int(process.findtext("pid", ""))
            if pid <= 0:
                raise ValueError("Invalid process PID")
            processes.append({"pid": pid, "type": process.findtext("type", "unknown"),
                              "name": process.findtext("process_name", "unknown")})
        return {"gpu_uuid": uuid, "index": 0, "free_mib": int(free[1]),
                "utilization_percent": int(util[1]), "processes": processes}
    except (OSError, subprocess.SubprocessError, ET.ParseError, ValueError) as exc:
        raise GPUQueryError(f"GPU 0 query failed closed: {exc}") from exc


def busy_reason(snapshot, minimum_free_mib):
    others = [p for p in snapshot["processes"] if p["pid"] != os.getpid()]
    if others:
        return f"GPU 0 has other processes: {others}"
    if snapshot["free_mib"] < minimum_free_mib:
        return f"GPU 0 free memory {snapshot['free_mib']} MiB < {minimum_free_mib} MiB"
    if snapshot["utilization_percent"] > MAX_UTILIZATION_PERCENT:
        return f"GPU 0 utilization {snapshot['utilization_percent']}% > {MAX_UTILIZATION_PERCENT}%"
    return None


@contextmanager
def exclusive_lock(path):
    """Nonblocking lock on an existing, workspace-local parent directory."""
    path = Path(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or not parent.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Lock parent must be an existing workspace directory: {parent}")
    fd = os.open(parent / path.name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _monitor(stop, uuid, interval):
    previous = None
    while not stop.wait(interval):
        try:
            snapshot = query_gpu0()
            others = [p for p in snapshot["processes"] if p["pid"] != os.getpid()]
            warning = (f"External GPU conflict: {others}; UUID={snapshot['gpu_uuid']}"
                       if others or snapshot["gpu_uuid"] != uuid else None)
        except GPUQueryError as exc:
            warning = str(exc)
        if warning and warning != previous:
            print(json.dumps({"gpu_lease_warning": warning, "time": time.time(),
                              "action": "audit only; no process signalled"}), file=sys.stderr, flush=True)
        previous = warning


@contextmanager
def gpu_lease(wait=False, poll_seconds=30, timeout_seconds=None, minimum_free_mib=20000):
    """Yield {'gpu_uuid': ..., 'index': 0} while holding the cooperative lock.

    Query errors are fatal, including in wait mode. Busy attempts release the
    lock before sleeping. The optional timeout bounds acquisition, not the job.
    CUDA_VISIBLE_DEVICES must already be exactly '0'; this module never sets it.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("CUDA_VISIBLE_DEVICES must be exactly '0'")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be finite and positive")
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds < 0):
        raise ValueError("timeout_seconds must be finite and nonnegative, or None")
    if not math.isfinite(minimum_free_mib) or minimum_free_mib < 0:
        raise ValueError("minimum_free_mib must be finite and nonnegative")
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    first = True
    while True:
        if not first and deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for GPU 0")
        first = False
        # Catch lock contention only during acquisition, never an exception from
        # the caller's with-body (which must not accidentally be retried).
        lock = exclusive_lock(LOCK_PATH)
        try:
            lock.__enter__()
        except BlockingIOError:
            reason = "GPU 0 cooperative lock is held"
        else:
            try:
                snapshot = query_gpu0()
                reason = busy_reason(snapshot, minimum_free_mib)
                if reason is None:
                    if deadline is not None and timeout_seconds != 0 and time.monotonic() > deadline:
                        raise TimeoutError("GPU 0 query exceeded acquisition deadline")
                    lease = {"gpu_uuid": snapshot["gpu_uuid"], "index": 0}
                    print(json.dumps({"gpu_lease": lease, "pid": os.getpid(), "time": time.time()}),
                          file=sys.stderr, flush=True)
                    stop = threading.Event()
                    monitor = threading.Thread(target=_monitor, args=(stop, lease["gpu_uuid"],
                                                                      max(30, poll_seconds)), daemon=True)
                    monitor.start()
                    try:
                        yield lease
                    finally:
                        stop.set()
                        monitor.join()
                    return
            finally:
                lock.__exit__(None, None, None)
        if not wait:
            raise GPUBusyError(reason)
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"Timed out waiting for GPU 0: {reason}")
        print(json.dumps({"gpu_wait": reason, "pid": os.getpid(), "time": time.time()}),
              file=sys.stderr, flush=True)
        time.sleep(poll_seconds if remaining is None else min(poll_seconds, remaining))
