"""Wall time and dedicated GPU allocation for selection and GRPO experiments."""
import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from checkpoints import TRACKER, atomic_text


def checkpoint_step(root):
    try:
        return int((root / TRACKER).read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


@contextmanager
def measure_stage(directory, stage, allocated_gpus, *, step=None, target_step=None):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    start_step = checkpoint_step(root) if step is None else step
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    status = "failed"
    try:
        yield
        status = "ok"
    finally:
        seconds = time.monotonic() - start
        gpu_hours = seconds * allocated_gpus / 3600 if allocated_gpus is not None else None
        event = dict(stage=stage, status=status, started_at=started, wall_seconds=seconds,
                     grpo_step_start=start_step, grpo_step_end=checkpoint_step(root), target_step=target_step,
                     allocated_gpus=allocated_gpus, gpu_hours=gpu_hours)
        log = root / "timings.jsonl"
        # The orchestrator and standalone training launcher write the same ledger.
        with (root / ".timing.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = log.read_text() if log.exists() else ""
            last = json.loads(previous.rstrip().rsplit("\n", 1)[-1]) if previous else {}
            cumulative = last.get("cumulative_gpu_hours", 0.0)
            event["cumulative_gpu_hours"] = (cumulative + gpu_hours
                                             if cumulative is not None and gpu_hours is not None else None)
            atomic_text(log, previous + json.dumps(event) + "\n")
