"""Wall-clock instrumentation only: no sampling, loss evaluation or model changes."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid
import warnings

import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class PipelineTiming:
    """One atomic JSON journal per top-level call, retaining every stage attempt.

    Stage/phase durations include nested work. Never add a parent duration to
    its children. A hard-killed process leaves status='running', not 'complete'.
    """

    def __init__(self, directory, *, system, device):
        self.directory = Path(directory)
        self.system = system
        self.device = torch.device(device)
        self.stack = []
        self.document = None
        self.path = None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _save(self):
        # A logging failure must not replace a training exception or discard a model.
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".partial")
            temporary.write_text(json.dumps(self.document, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, ValueError, TypeError) as error:
            warnings.warn(f"Unable to save pipeline timings to {self.path}: {error}", RuntimeWarning)

    def _close_phase(self, frame, now):
        if frame["phase_started"] is not None:
            elapsed = now - frame["phase_started"]
            record = frame["record"]
            name = frame["phase"]
            record["sections"].append({"name": name, "elapsed_seconds": elapsed})
            totals = record["phase_seconds"]
            totals[name] = totals.get(name, 0.0) + elapsed

    @contextmanager
    def measure(self, name, *, kind="stage", initial_phase="setup", details=None):
        top_level = not self.stack
        if top_level:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            self.path = self.directory / f"{stamp}_{uuid.uuid4().hex}.json"
            self.document = {
                "schema_version": 1, "system": self.system, "pid": os.getpid(),
                "started_at_utc": utc_now(), "status": "running",
                "device": str(self.device), "python_version": platform.python_version(),
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "platform": platform.platform(), "gpu_name": None,
                "clock": "perf_counter wall seconds; CUDA synchronized at boundaries",
                "nested_durations": "inclusive; do not sum parent and child records",
                "events": [],
            }
        parent = self.stack[-1]["record"] if self.stack else None
        record = {
            "id": len(self.document["events"]), "parent_id": None if parent is None else parent["id"],
            "name": name, "kind": kind, "details": details or {},
            "started_at_utc": utc_now(), "finished_at_utc": None,
            "status": "running", "elapsed_seconds": None,
            "phase_seconds": {}, "sections": [], "active_phase": initial_phase,
            "final_training_loss": None,
        }
        frame = {"record": record, "started": None, "phase_started": None, "phase": initial_phase}
        self.document["events"].append(record)
        self.stack.append(frame)
        self._save()
        try:
            self._sync()
            if top_level and self.device.type == "cuda":
                self.document["gpu_name"] = torch.cuda.get_device_name(self.device)
            frame["started"] = frame["phase_started"] = time.perf_counter()
            yield record
            self._sync()
        except BaseException as error:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            try:
                self._sync()
            except Exception as sync_error:
                record["synchronization_error"] = str(sync_error)
            raise
        finally:
            now = time.perf_counter()
            self._close_phase(frame, now)
            if frame["started"] is not None:
                record["elapsed_seconds"] = now - frame["started"]
            record["finished_at_utc"] = utc_now()
            record["active_phase"] = None
            if record["status"] == "running":
                record["status"] = "complete"
            if top_level:
                self.document.update(status=record["status"], finished_at_utc=record["finished_at_utc"],
                                     elapsed_seconds=record["elapsed_seconds"])
            self._save()
            self.stack.pop()

    def phase(self, name):
        """Partition the current stage at a coarse boundary, never per optimizer step."""
        if not self.stack:
            return
        self._sync()
        now = time.perf_counter()
        frame = self.stack[-1]
        self._close_phase(frame, now)
        frame.update(phase=name, phase_started=now)
        frame["record"]["active_phase"] = name
        self._save()

    def final_loss(self, history, *, filename):
        """Copy the last existing epoch record, not a new checkpoint evaluation."""
        if not self.stack:
            return
        last = dict(history[-1]) if history else None
        nonfinite = []
        if last is not None:
            for key, value in last.items():
                if isinstance(value, float) and not math.isfinite(value):
                    nonfinite.append(key)
                    last[key] = None
        self.stack[-1]["record"]["final_training_loss"] = {
            "history_file": filename, "epochs_recorded": len(history),
            "total": None if last is None else last["total"], "last_epoch": last,
            "nonfinite_fields": nonfinite,
            "meaning": "last epoch batch-averaged weighted training loss; not final-checkpoint evaluation",
        }
        self._save()

    def skip(self, reason):
        if self.stack:
            self.stack[-1]["record"].update(status="skipped", reason=reason)
            self._save()


def timed_stage(*, initial_phase="setup", kind="stage"):
    """Cover CLI dispatch and direct method calls, including nested prerequisites."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            details = {}
            if kind == "invocation":
                details["requested_stage"] = args[0] if args else kwargs.get("stage")
            with self.timing.measure(method.__name__, kind=kind, initial_phase=initial_phase, details=details):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate
