"""Dataset roles and run identity for online selection and direct GRPO."""
import fcntl
import hashlib
import json
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq

from checkpoints import atomic_text


@contextmanager
def run_lock(directory, *, wait=False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another experiment already owns {directory}") from exc
        yield


def _prompts(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(columns=["prompt"], batch_size=1024):
            yield from batch.column(0).to_pylist()
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)["prompt"]


def _prompt_key(prompt):
    if isinstance(prompt, list):
        prompt = [msg["content"] for msg in prompt if msg.get("role") == "user"]
        if not prompt:
            raise ValueError("Dataset row has no user prompt")
        prompt = json.dumps(prompt, ensure_ascii=False, sort_keys=True) if len(prompt) != 1 else prompt[0]
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split())
    if not normalized:
        raise ValueError("Empty dataset prompt")
    return hashlib.sha256(normalized.encode()).digest()


def describe_datasets(train, reference, heldout):
    """Check the files actually consumed. Local row indices may overlap across datasets."""
    heldout_keys = set()
    descriptions = {}
    for role, source in (("heldout_eval", heldout), ("selection_reference", reference), ("training_pool", train)):
        if source is None:
            continue
        label, path = source
        path = Path(path).expanduser().resolve()
        count = 0
        for prompt in _prompts(path):
            key = _prompt_key(prompt)
            if role == "heldout_eval":
                heldout_keys.add(key)
            elif key in heldout_keys:
                raise ValueError(f"{role} overlaps heldout_eval by normalized user prompt: {path}")
            count += 1
        if not count:
            raise ValueError(f"Empty {role} dataset: {path}")
        with path.open("rb") as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        descriptions[role] = {"dataset": label, "file": str(path), "rows": count, "sha256": digest.hexdigest()}
    return descriptions


def record_experiment(root, args, datasets):
    manifest = root / "experiment.json"
    settings = {key: value for key, value in vars(args).items()
                if key not in ("num_selections", "adopt_legacy_run", "config")}
    # Old manifests predate the opt-in FSDP/adapter flags. Preserve their
    # identity when a new invocation uses the original behavior.
    defaults = dict(adapter_mode=None, branch_rank=4, forget_grad_scale=2.0, rollout_tp=None,
                    greedy_eval=False, model_path=None, n_gpus_per_node=None, ppo_mini_batch_size=None,
                    ppo_micro_batch_size_per_gpu=None, lr=None, initial_oracle_cache=False,
                    retain_fraction=0.2, forget_fraction=0.2, rollout_max_num_batched_tokens=None, save_freq=None)
    settings = {k: v for k, v in settings.items() if k not in defaults or v != defaults[k]}
    identity = {"version": 1, "settings": settings, "datasets": datasets}
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(f"Experiment settings or data changed in {root}. Use a new --prefix.")
        return
    if (any(root.glob("global_step_*")) or (root / "chunks").exists()) and not args.adopt_legacy_run:
        raise ValueError(f"Legacy run has no experiment.json: {root}. Use a new prefix or --adopt_legacy_run.")
    atomic_text(manifest, json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
