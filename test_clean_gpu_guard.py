"""CPU-only guard tests: synthetic XML, mocked queries, real subprocess locks."""

from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import clean_gpu_guard as guard


def xml(processes="", free="24000 MiB", utilization="0 %", uuid="GPU-test-0"):
    return (f"<nvidia_smi_log><gpu><uuid>{uuid}</uuid><fb_memory_usage><free>{free}</free>"
            f"</fb_memory_usage><utilization><gpu_util>{utilization}</gpu_util></utilization>"
            f"<processes>{processes}</processes></gpu></nvidia_smi_log>")


def process(pid, kind="C", name="root-owned-job"):
    return (f"<process_info><pid>{pid}</pid><type>{kind}</type>"
            f"<process_name>{name}</process_name></process_info>")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir="/tmp/opencode")))
        self.stack.enter_context(patch.object(guard, "ROOT", self.root))
        self.stack.enter_context(patch.object(guard, "LOCK_PATH", self.root / "gpu0.lock"))
        self.stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        self.query = self.stack.enter_context(patch.object(guard.subprocess, "run", return_value=Mock(stdout=xml())))

    def test_query_is_gpu0_xml_and_lease_returns_uuid(self):
        with guard.gpu_lease() as lease:
            self.assertEqual(lease, {"gpu_uuid": "GPU-test-0", "index": 0})
            with self.assertRaises(BlockingIOError), guard.exclusive_lock(guard.LOCK_PATH):
                self.fail("lease was not locked")
        self.query.assert_called_once_with(["nvidia-smi", "-i", "0", "-q", "-x"],
                                           capture_output=True, text=True, check=True, timeout=15)
        with guard.exclusive_lock(guard.LOCK_PATH):
            pass

    def test_compute_graphics_root_owned_and_unknown_processes_block(self):
        for kind in ("C", "G", "C+G", "unknown"):
            self.query.return_value.stdout = xml(process(1, kind))
            with self.subTest(kind=kind), self.assertRaises(guard.GPUBusyError), guard.gpu_lease():
                self.fail("another PID was allowed")

    def test_only_own_pid_is_exempt(self):
        self.query.return_value.stdout = xml(process(os.getpid()))
        with guard.gpu_lease():
            pass
        self.query.return_value.stdout = xml(process(os.getpid()) + process(os.getppid(), "G"))
        with self.assertRaises(guard.GPUBusyError), guard.gpu_lease():
            pass

    def test_free_memory_and_utilization_thresholds(self):
        for free, util in (("19999 MiB", "0 %"), ("24000 MiB", "6 %")):
            self.query.return_value.stdout = xml(free=free, utilization=util)
            with self.subTest(free=free, util=util), self.assertRaises(guard.GPUBusyError), guard.gpu_lease():
                pass
        self.query.return_value.stdout = xml(free="20000 MiB", utilization="5 %")
        with guard.gpu_lease():
            pass

    def test_malformed_or_unavailable_fields_fail_closed_even_when_waiting(self):
        samples = ["broken XML", xml(free="N/A"), xml(utilization="N/A"), xml(uuid="N/A"),
                   xml(process("N/A")), xml(process(0)), xml("N/A"), xml("<unknown/>"),
                   xml().replace("<processes></processes>", ""),
                   xml().replace("</gpu>", "</gpu><gpu/>")]
        for sample in samples:
            self.query.return_value.stdout = sample
            with self.subTest(sample=sample), patch.object(guard.time, "sleep") as sleep:
                with self.assertRaises(guard.GPUQueryError), guard.gpu_lease(wait=True):
                    pass
                sleep.assert_not_called()
                with guard.exclusive_lock(guard.LOCK_PATH):
                    pass

    def test_query_failures_never_grant_a_lease(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 15),
                      subprocess.CalledProcessError(1, "nvidia-smi")):
            self.query.side_effect = error
            with self.subTest(error=error), self.assertRaises(guard.GPUQueryError), guard.gpu_lease(wait=True):
                pass

    def test_lock_acquired_before_query_and_released_before_sleep(self):
        def query(*args, **kwargs):
            with self.assertRaises(BlockingIOError), guard.exclusive_lock(guard.LOCK_PATH):
                pass
            return Mock(stdout=xml(process(1)) if self.query.call_count == 1 else xml())

        def sleep(seconds):
            self.assertEqual(seconds, 30)
            with guard.exclusive_lock(guard.LOCK_PATH):
                pass

        self.query.side_effect = query
        with patch.object(guard.time, "sleep", side_effect=sleep) as sleeping:
            with guard.gpu_lease(wait=True):
                pass
            sleeping.assert_called_once_with(30)

    def test_busy_lock_never_queries_gpu(self):
        with guard.exclusive_lock(guard.LOCK_PATH):
            with self.assertRaises(guard.GPUBusyError), guard.gpu_lease():
                pass
        self.query.assert_not_called()

    def test_timeout_is_finite_and_sleep_is_capped(self):
        self.query.return_value.stdout = xml(process(1))
        with patch.object(guard.time, "monotonic", side_effect=[0, 0, 5]), patch.object(guard.time, "sleep") as sleep:
            with self.assertRaises(TimeoutError), guard.gpu_lease(wait=True, timeout_seconds=5):
                pass
        sleep.assert_called_once_with(5)

    def test_invalid_environment_and_wait_arguments(self):
        for visible in ("", "1", "0,1", "GPU-test-0", " 0"):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible):
                with self.assertRaises(ValueError), guard.gpu_lease():
                    pass
        for kwargs in ({"poll_seconds": 0}, {"poll_seconds": float("nan")},
                       {"timeout_seconds": -1}, {"timeout_seconds": float("inf")}, {"minimum_free_mib": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError), guard.gpu_lease(**kwargs):
                pass
        self.query.assert_not_called()

    def test_body_exception_is_not_retried_and_unlocks(self):
        with self.assertRaises(BlockingIOError), guard.gpu_lease(wait=True):
            raise BlockingIOError("body failure, not lock contention")
        self.query.assert_called_once()
        with guard.exclusive_lock(guard.LOCK_PATH):
            pass

    def test_fcntl_order(self):
        events = []
        real = fcntl.flock

        def flock(fd, operation):
            events.append(operation)
            return real(fd, operation)

        with patch.object(guard.fcntl, "flock", side_effect=flock):
            with guard.gpu_lease():
                self.assertEqual(events, [fcntl.LOCK_EX | fcntl.LOCK_NB])
        self.assertEqual(events[-1], fcntl.LOCK_UN)

    def test_lock_shared_with_subprocess_and_import_never_loads_torch(self):
        # Temporarily undo only the subprocess.run mock for an actual CPU child.
        self.stack.close()
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            code = ("import fcntl,sys; import clean_gpu_guard; "
                    "assert 'torch' not in sys.modules; "
                    "f=open(sys.argv[1],'a'); "
                    "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)")
            with patch.object(guard, "ROOT", root):
                lock = root / "gpu0.lock"
                with guard.exclusive_lock(lock):
                    result = subprocess.run([sys.executable, "-B", "-c", code, str(lock)],
                                            cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("BlockingIOError", result.stderr)
                result = subprocess.run([sys.executable, "-B", "-c", code, str(lock)],
                                        cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_external_monitor_only_reports(self):
        stop = Mock()
        stop.wait.side_effect = [False, True]
        self.query.return_value.stdout = xml(process(1, "G"))
        with patch("builtins.print") as report, patch.object(os, "kill") as kill:
            guard._monitor(stop, "GPU-test-0", 30)
        report.assert_called_once()
        kill.assert_not_called()

    def test_lock_symlink_and_missing_parent_refused(self):
        target = self.root / "elsewhere"
        target.touch()
        guard.LOCK_PATH.symlink_to(target)
        with self.assertRaises(OSError), guard.gpu_lease():
            pass
        with self.assertRaises(FileNotFoundError), guard.exclusive_lock(self.root / "missing" / "gpu0.lock"):
            pass


if __name__ == "__main__":
    unittest.main()
