"""Completion and resume boundaries for the local, synchronous GRPO checkpoint layout."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from contextlib import contextmanager

MARKER = "gradalign_complete.json"
TRACKER = "latest_checkpointed_iteration.txt"


def atomic_text(path, value):
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as out:
        name = out.name
        try:
            out.write(value)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
    os.replace(name, path)


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def hf_weights_ready(directory):
    if not nonempty(directory / "config.json"):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        index = directory / (name + ".index.json")
        if index.exists():
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
            except (KeyError, ValueError, TypeError, AttributeError):
                return False
            return bool(shards) and all(nonempty(directory / shard) for shard in shards)
        if nonempty(directory / name):
            return True
    return False


def packed_adapter_ready(directory):
    from safetensors import SafetensorError, safe_open

    try:
        rank = json.loads((directory / "adapter_config.json").read_text())["r"]
        if type(rank) is not int or rank < 2 or rank % 2:
            return False
        with safe_open(directory / "adapter_model.safetensors", framework="np") as weights:
            keys = set(weights.keys())
            a_keys = {k for k in keys if k.endswith(".lora_A.weight")}
            pairs = [(a, a.removesuffix(".lora_A.weight") + ".lora_B.weight") for a in a_keys]
            return bool(pairs) and keys == a_keys | {b for _, b in pairs} and all(
                weights.get_slice(a).get_shape()[0] == weights.get_slice(b).get_shape()[1] == rank
                for a, b in pairs
            )
    except (OSError, KeyError, IndexError, TypeError, ValueError, SafetensorError):
        return False


def payload_ready(directory, backend, paired=False):
    actor = directory / "actor"
    if paired and not packed_adapter_ready(actor / "graft_adapter"):
        return False
    if backend == "fsdp":
        try:
            size = json.loads((actor / "fsdp_config.json").read_text())["world_size"]
        except (OSError, KeyError, ValueError, TypeError):
            return False
        return isinstance(size, int) and size > 0 and all(
            nonempty(actor / f"{kind}_world_size_{size}_rank_{rank}.pt")
            for kind in ("model", "optim", "extra_state") for rank in range(size)
        )
    # Megatron's torch_dist backend saves tensor storage and a separate common state.
    dist = actor / "dist_ckpt"
    if not all(nonempty(dist / name) for name in ("common.pt", ".metadata", "metadata.json")):
        return False
    from torch.distributed.checkpoint import FileSystemReader
    import pickle

    try:
        if json.loads((dist / "metadata.json").read_text()).get("sharded_backend") != "torch_dist":
            return False
        metadata = FileSystemReader(dist).read_metadata()
        storage = metadata.storage_data
        if not storage:
            return False
        required_sizes = {}
        for part in storage.values():
            required_sizes[part.relative_path] = max(
                required_sizes.get(part.relative_path, 0), part.offset + part.length
            )
        if any((dist / name).stat().st_size < size for name, size in required_sizes.items()):
            return False
        keys = metadata.state_dict_metadata.keys()
        has_optimizer = any(key == "optimizer" or key.startswith("optimizer.") for key in keys)
        has_model = any(key.startswith("model") or
                        (not key.startswith("optimizer.") and key.endswith((".weight", ".bias"))) for key in keys)
        return has_optimizer and (has_model or hf_weights_ready(actor / "huggingface"))
    except (OSError, ValueError, EOFError, AttributeError, pickle.UnpicklingError):
        return False


def mark_complete(root, step, backend, paired=False):
    directory = root / f"global_step_{step}"
    if not nonempty(directory / "data.pt") or not payload_ready(directory, backend, paired):
        raise RuntimeError(f"Incomplete {backend} checkpoint at {directory}")
    atomic_text(directory / MARKER, json.dumps({"step": step, "backend": backend}) + "\n")


def latest_complete(root, interval, backend, paired=False):
    try:
        tracked = int((root / TRACKER).read_text().strip())
    except (OSError, ValueError):
        tracked = 0
    # A staging directory may exist long before its weights. A completed round
    # marker or the trainer's post-save tracker must also attest to the write.
    steps = sorted((int(p.name[12:]) for p in root.glob("global_step_*")
                    if p.is_dir() and p.name[12:].isdigit()), reverse=True)
    for step in steps:
        if step == 0:
            continue
        directory = root / f"global_step_{step}"
        marker = directory / MARKER
        try:
            marked = json.loads(marker.read_text()) == {"step": step, "backend": backend}
        except (OSError, ValueError):
            marked = False
        if not marked and step > tracked:
            continue
        # Older runs removed data.pt at a round boundary to switch datasets.
        if step % interval and not nonempty(directory / "data.pt"):
            continue
        if payload_ready(directory, backend, paired):
            if step != tracked:
                atomic_text(root / TRACKER, str(step))
            return step
    return 0


@contextmanager
def resume_arguments(root, step, boundary):
    if not step:
        yield ["--resume_mode", "disable"]
        return
    directory = root / f"global_step_{step}"
    if not boundary:
        yield ["--resume_from_path", str(directory)]
        return
    # Preserve the saved sampler for replays. A boundary uses the next Parquet
    # and its fresh dataloader while restoring the same actor and optimizer.
    with TemporaryDirectory(dir=root, prefix=".resume-") as tmp:
        view = Path(tmp) / directory.name
        view.mkdir()
        (view / "actor").symlink_to(directory / "actor", target_is_directory=True)
        yield ["--resume_from_path", str(view)]
