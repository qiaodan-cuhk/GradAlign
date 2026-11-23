#!/usr/bin/env python3

import os
import json
import argparse
import subprocess
import sys
import math
import tempfile
from itertools import islice
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple

import datasets as hfds  # type: ignore[import-not-found]

from config import add_dataset_roles, get_dataset_dir, parse_experiment_args
from experiment import describe_datasets, record_experiment, run_lock
from checkpoints import TRACKER, atomic_text, hf_weights_ready, latest_complete, mark_complete, resume_arguments
from select_data import SELECTOR_REGISTRY
from timing import measure_stage
from launch_verl_training import DEFAULT_CHECKPOINT_ROOT, DEFAULT_ROLLOUT_N, build_exp_name


def _load_train_dataset(train_dir: str, train_parquet: str):
    parquet_path = os.path.join(train_dir, train_parquet)
    jsonl_path = os.path.join(train_dir, "train.jsonl")
    if os.path.isfile(parquet_path):
        return hfds.load_dataset("parquet", data_files=parquet_path, split="train")
    if os.path.isfile(jsonl_path):
        records: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
        return hfds.Dataset.from_list(records)
    raise SystemExit(f"No train parquet or jsonl found in {train_dir}")


def write_jsonl(records: List[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def chunk_dataset(train_dir: str, exp_root: str, train_parquet: str, chunk_size: int, seed: int) -> Tuple[int, List[str]]:
    ds = _load_train_dataset(train_dir, train_parquet)
    total = len(ds)
    if total == 0:
        raise SystemExit("Empty training dataset; nothing to chunk")
    ds = ds.shuffle(seed=seed)

    if chunk_size <= 0:
        raise SystemExit("--chunk_size must be > 0")
    num_chunks = total // chunk_size
    if num_chunks <= 0:
        num_chunks = 1
        effective_chunk_size = total
    else:
        effective_chunk_size = chunk_size

    leftover = total - (num_chunks * effective_chunk_size)
    if leftover > 0:
        print(f"Info: dropping leftover {leftover} records to match floor(total/chunk_size)")

    chunk_dirs: List[str] = []
    for i in range(num_chunks):
        start = i * effective_chunk_size
        end = start + effective_chunk_size
        indices = list(range(start, min(end, total)))
        if not indices:
            continue
        chunk = ds.select(indices)
        chunk_dir = os.path.join(exp_root, f"chunk_{i}")
        os.makedirs(chunk_dir, exist_ok=True)
        out_parquet = os.path.join(chunk_dir, "train.parquet")
        chunk.to_parquet(out_parquet)
        # Also write JSONL for compatibility with inference scripts
        records = chunk.to_list()
        out_jsonl = os.path.join(chunk_dir, "train.jsonl")
        write_jsonl(records, out_jsonl)
        print(f"Wrote chunk {i}: {len(chunk)} rows -> {out_parquet} and {out_jsonl}")
        chunk_dirs.append(chunk_dir)

    # Write/update manifest
    manifest = {
        "experiment_name": os.path.basename(exp_root.rstrip("/")),
        "train_dir": train_dir,
        "total_rows": total,
        "chunk_size": effective_chunk_size,
        "num_chunks": num_chunks,
        "dropped_rows": leftover if leftover > 0 else 0,
        "seed": seed,
        "train_parquet": train_parquet,
    }
    atomic_text(Path(exp_root) / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return num_chunks, chunk_dirs


def main():
    parser = argparse.ArgumentParser(description="Dynamic selection orchestrator: chunk → infer → analyze → aggregate → select → train")
    parser.add_argument("--config", metavar="YAML", help="Experiment settings; explicit CLI arguments take precedence")
    parser.add_argument("--prefix", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--train_dataset", required=True, type=str)
    add_dataset_roles(parser)
    parser.add_argument("--adopt_legacy_run", action="store_true", help="Record settings for a pre-manifest run after verifying its original configuration")

    # Optional explicit overrides (to perfectly match experiment naming)
    parser.add_argument("--train_dir", default=None, type=str,
                        help="Override directory containing the training parquet/jsonl")
    parser.add_argument("--val_dir", default=None, type=str,
                        help="Override directory containing the validation parquet/jsonl")
    parser.add_argument("--train_parquet", default="train.parquet", type=str)

    # Chunking controls
    parser.add_argument("--chunk_size", required=True, type=int,
                        help="Number of items per chunk")
    parser.add_argument("--k", required=True, type=int,
                        help="Number of items per chunk")
    parser.add_argument("--seed", default=42, type=int)

    # Selection controls
    parser.add_argument("--mode", choices=SELECTOR_REGISTRY, required=True)
    parser.add_argument("--acc_low", type=float, default=0.2)
    parser.add_argument("--acc_high", type=float, default=0.8)
    parser.add_argument("--epochs_per_select", type=int, default=1,
                        help="Train epochs per selection iteration (we run 1 epoch per launch; this controls total iterations with total_epochs)")
    parser.add_argument("--num_selections", type=int, required=True,
                        help="Number of selection iterations to run")
    parser.add_argument("--train_batch_size", type=int, required=True,
                        help="Train batch size used to determine selection size")
    parser.add_argument("--iters_per_select", type=int, required=True,
                        help="Number of training iterations per selection (compute top N = batch_size * iters_per_select)")

    parser.add_argument("--rollout_n", type=int, default=DEFAULT_ROLLOUT_N, help="Fresh GRPO training answers per selected prompt")

    # Inference controls
    parser.add_argument("--analysis_gpus", type=int, default=8)
    parser.add_argument("--allocated_gpus", type=int, help="GPUs reserved for the duration of this run, for cost accounting")
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--n_samples_train", type=int, default=32)
    parser.add_argument("--n_samples_val", type=int, default=128)
    parser.add_argument("--min_problems", type=int, default=1)
    parser.add_argument("--max_problems", type=int, default=40000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--train_infer_batch_size", type=int, default=512)
    parser.add_argument("--val_infer_batch_size", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--use_optimizer", action="store_true", default=False)
    parser.add_argument("--use_model_judge", action="store_true", help="Use the model judge for selection rollouts")
    parser.add_argument("--reward_manager", type=str, default=None)
    parser.add_argument("--reward_path", type=str, default=None)
    
    # Checkpoint/merge controls
    parser.add_argument("--ckpt_root", type=str, default=DEFAULT_CHECKPOINT_ROOT,
                        help="Root directory where VERL checkpoints are saved")
    parser.add_argument("--merge_backend", type=str, default="megatron")

    args = parse_experiment_args(parser)
    strategy = SELECTOR_REGISTRY[args.mode]
    if args.rollout_n < 1:
        parser.error("--rollout_n must be positive")
    if args.allocated_gpus is not None and args.allocated_gpus < 0:
        parser.error("--allocated_gpus must be nonnegative")
    args.ckpt_root = os.path.abspath(os.path.expanduser(args.ckpt_root))
    if min(args.chunk_size, args.k, args.train_batch_size, args.iters_per_select, args.num_selections,
           args.n_samples_train, args.n_samples_val, args.min_problems, args.analysis_gpus) < 1 or args.max_problems < args.min_problems:
        parser.error("Batch sizes, round counts and the problem range must be positive")
    if args.chunk_size < args.k:
        parser.error("The selection ratio must leave at least one candidate")

    # Resolve dataset dirs when not explicitly provided
    train_dir = args.train_dir or get_dataset_dir(args.train_dataset, args.model)
    if strategy.reference_rollouts and not args.selection_reference:
        parser.error(f"{args.mode} requires --selection_reference")
    val_dir = args.val_dir or (get_dataset_dir(args.selection_reference) if args.selection_reference else None)
    heldout_dir = args.heldout_dir or get_dataset_dir(args.heldout_eval)
    train_dir = str(Path(train_dir).expanduser().resolve())
    val_dir = str(Path(val_dir).expanduser().resolve()) if val_dir else None
    heldout_dir = str(Path(heldout_dir).expanduser().resolve())

    # Use a fixed experiment name (prefix + model + datasets) to align with VERL checkpoints
    exp_name = build_exp_name(args.prefix, args.model, args.train_dataset, train_dir, val_dir or "no_reference")
    root = Path(args.ckpt_root) / exp_name
    with run_lock(root):
        source = Path(train_dir) / args.train_parquet
        if not source.is_file():
            source = Path(train_dir) / "train.jsonl"
        roles = describe_datasets((args.train_dataset, source),
                                  (args.selection_reference, Path(val_dir) / "train.jsonl") if val_dir else None,
                                  (args.heldout_eval, Path(heldout_dir) / args.heldout_parquet))
        record_experiment(root, args, roles)
        run_rounds(args, strategy, exp_name, train_dir, val_dir, heldout_dir)


def run_script(script, *arguments):
    cmd = [sys.executable, str(Path(__file__).with_name(script)), *map(str, arguments)]
    print("Executing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare_chunks(args, train_dir, root):
    manifest = root / "manifest.json"
    if manifest.is_file():
        info = json.loads(manifest.read_text())
        count = info["num_chunks"]
        size = min(info["total_rows"], args.chunk_size)
        if (info["seed"], info["chunk_size"], info["train_parquet"]) != (args.seed, size, args.train_parquet):
            raise ValueError(f"Chunk manifest does not match this experiment: {manifest}")
        chunks = [root / f"chunk_{i}" for i in range(count)]
        if all((chunk / "train.jsonl").is_file() and (chunk / "train.parquet").is_file()
               and pq.read_metadata(chunk / "train.parquet").num_rows == size for chunk in chunks):
            return chunks, size
    _, chunks = chunk_dataset(train_dir, str(root), args.train_parquet, args.chunk_size, args.seed)
    return list(map(Path, chunks)), pq.read_metadata(Path(chunks[0]) / "train.parquet").num_rows


def responses_complete(path, ids, samples):
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or len(records) != len(ids) * samples:
            return False
        seen = set()
        for row in records:
            key = (row["group_id"], row["sample_id"])
            if (key[0] not in ids or not 0 <= key[1] < samples or key in seen
                    or not isinstance(row.get("passed", row.get("is_correct")), bool)):
                return False
            seen.add(key)
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
    return True


def ensure_rollouts(args, dataset, prompts, directory, model_path, samples, batch_size):
    directory.mkdir(parents=True, exist_ok=True)
    with prompts.open(encoding="utf-8") as f:
        ids = {json.loads(line)["extra_info"]["index"] for line in islice(f, args.min_problems - 1, args.max_problems)}
    if not ids:
        raise ValueError(f"No prompts in requested range: {prompts}")
    sorted_file = directory / "responses_sorted.json"
    if responses_complete(sorted_file, ids, samples):
        return ids
    if responses_complete(directory / "responses.json", ids, samples):
        run_script("sort_responses.py", "--model_name", args.model, "--dataset", dataset, "--responses_dir", directory)
    else:
        options = ["--model_path", model_path] if model_path else []
        if args.use_model_judge or Path(args.reward_path or "").name == "model_reward.py":
            options.append("--use_model_judge")
        run_script("run_inference_local.py", "--model", args.model, "--resp_dataset", dataset,
                   "--prompts_file", prompts, "--output_dir", directory, "--n_samples", samples,
                   "--min_problems", args.min_problems, "--max_problems", args.max_problems,
                   "--temperature", args.temperature, "--max_tokens", args.max_tokens,
                   "--tensor_parallel_size", args.tensor_parallel_size, "--pipeline_parallel_size", args.pipeline_parallel_size,
                   "--batch_size", batch_size, "--max_model_len", args.max_model_len, "--concurrency", args.concurrency,
                   *options)
    if not responses_complete(sorted_file, ids, samples):
        raise RuntimeError(f"Incomplete rollout groups in {sorted_file}")
    return ids


def analysis_complete(part, reference, ids, mode):
    # The analyzer appends one row per problem and resumes by group_id.
    inputs = {}
    for role, folder in (("candidate", part), ("reference", reference)):
        stat = (folder / "responses_sorted.json").stat()
        inputs[role] = [stat.st_size, stat.st_mtime_ns]
    inputs["mode"] = mode
    manifest = part / "analysis_inputs.json"
    scores = part / "similarity_results_cosine_real.jsonl"
    matching = manifest.is_file() and json.loads(manifest.read_text()) == inputs
    if matching and scores.exists():
        found = set()
        try:
            with scores.open() as f:
                for row in map(json.loads, f):
                    gid = int(row["group_id"])
                    if gid in found or gid not in ids or not math.isfinite(float(row["similarity"])):
                        raise ValueError("invalid similarity")
                    found.add(gid)
            return found == ids
        except (ValueError, TypeError, KeyError):
            pass
    if scores.exists():
        backup = Path(tempfile.mkdtemp(prefix=".previous-analysis-", dir=part))
        for artifact in part.glob("similarity_results_cosine_real*"):
            artifact.rename(backup / artifact.name)
    atomic_text(manifest, json.dumps(inputs))
    return False


def select_round(args, strategy, exp_name, chunk, val_dir, step_root, round_index, output, size):
    step = round_index * args.iters_per_select
    parts = step_root / "train_split"
    part = parts / "part_0"
    options = []
    model_path = None
    if step and strategy.candidate_rollouts:
        model_path = step_root / "actor" / "huggingface"
        if not hf_weights_ready(model_path):
            model_path = step_root / "merged"
            if not hf_weights_ready(model_path):
                run_script("merge_model.py", "--experiment_name", exp_name, "--step", step, "--output_model_name", exp_name,
                           "--backend", args.merge_backend, "--ckpt_root", args.ckpt_root, "--target_dir", model_path)
        options.extend(["--model_path", model_path])
        if args.use_optimizer and strategy.gradient:
            optimizer = step_root / "opt_converted"
            run_script("convert_megatron_optimizer_to_hf.py", "--checkpoint_path", step_root / "actor",
                       "--hf-config", model_path, "--output", optimizer)
            options.extend(["--use_optimizer", "--optimizer_state_path", optimizer])
    prompts = chunk / "train.jsonl"
    if strategy.candidate_rollouts:
        ids = ensure_rollouts(args, args.train_dataset, prompts, part, model_path, args.n_samples_train,
                              args.train_infer_batch_size)
    reference = part
    if strategy.reference_rollouts:
        reference = step_root / "val_responses"
        ensure_rollouts(args, args.selection_reference, Path(val_dir) / "train.jsonl", reference, model_path,
                        args.n_samples_val, args.val_infer_batch_size)
    if strategy.gradient and not analysis_complete(part, reference, ids, strategy.gradient):
        run_script("run_parallel_analysis.py", "--size", 1, "--idx", 0, "--model", args.model,
                   "--resp_dataset", args.train_dataset, "--problem_dataset", args.train_dataset,
                   "--problem_set_path", prompts, "--val_dataset", args.selection_reference or args.train_dataset,
                   "--parts_root", parts, "--val_responses_dir", reference, "--num_gpus", args.analysis_gpus,
                   "--mini_batch_size", args.minibatch_size, "--mixed_precision", "bf16",
                   "--max_length", args.max_model_len, "--mode", strategy.gradient, *options)
        if not analysis_complete(part, reference, ids, strategy.gradient):
            raise RuntimeError(f"Incomplete candidate gradient scores in {part}")
    if strategy.gradient or strategy.use_accuracy:
        run_script("aggregate.py", "--model_name", args.model, "--dataset", args.train_dataset, "--parts_root", parts)
    run_script("select_data.py", "--mode", args.mode, "--dataset", args.train_dataset, "--model", args.model,
               "--n", size, "--output_dir", output, "--candidate_dir", chunk, "--seed", args.seed + round_index,
               "--min_problems", args.min_problems, "--max_problems", args.max_problems,
               "--parts_root", parts, "--acc_low", args.acc_low, "--acc_high", args.acc_high)


def run_rounds(args, strategy, exp_name, train_dir, val_dir, heldout_dir):
    checkpoint_dir = Path(args.ckpt_root) / exp_name
    with measure_stage(checkpoint_dir, "prepare_chunks", args.allocated_gpus):
        chunks, chunk_size = prepare_chunks(args, train_dir, checkpoint_dir / "chunks")
    candidate_count = max(0, min(chunk_size, args.max_problems) - args.min_problems + 1)
    size = candidate_count // args.k
    if size < args.train_batch_size:
        raise ValueError(f"Selected subset of {size} cannot form one batch of {args.train_batch_size}")
    resume_step = latest_complete(checkpoint_dir, args.iters_per_select, args.merge_backend)
    for i in range(resume_step // args.iters_per_select, args.num_selections):
        gstep = i * args.iters_per_select
        iter_out = checkpoint_dir / "selected" / f"iter_{i}_{args.mode}_{size}"
        if resume_step > gstep:
            if not (iter_out / "train.parquet").is_file():
                raise RuntimeError(f"Missing selected dataset for in-round resume at step {resume_step}: {iter_out}")
        else:
            with measure_stage(checkpoint_dir, "selection", args.allocated_gpus, step=gstep):
                select_round(args, strategy, exp_name, chunks[i % len(chunks)], val_dir,
                             checkpoint_dir / f"global_step_{gstep}", i, iter_out, size)

        selected_count = pq.read_metadata(iter_out / "train.parquet").num_rows
        if selected_count < args.train_batch_size:
            raise ValueError(f"{args.mode} selected {selected_count} problems for a batch of {args.train_batch_size}")
        # 5) Launch training with selected set, fixed experiment name
        end_step = gstep + args.iters_per_select
        train_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "launch_verl_training.py"),
            "--prefix", args.prefix,
            "--model", args.model,
            "--train_dataset", args.train_dataset,
            "--heldout_eval", args.heldout_eval,
            "--val_dir", heldout_dir,
            "--val_parquet", args.heldout_parquet,
            "--train_dir", str(iter_out),
            "--total_epochs", str(end_step),  # upper bound; the absolute step below controls the budget
            "--total_training_steps", str(end_step),
            "--exp_name", exp_name,
            "--rollout_n", str(args.rollout_n),
            "--ckpts_root", args.ckpt_root,
            "--backend", args.merge_backend,
            '--train_batch_size', str(args.train_batch_size),
            '--max_response_length', str(args.max_tokens),
        ]
        if args.reward_manager:
            train_cmd.extend(["--reward_manager", args.reward_manager])
        if args.reward_path:
            train_cmd.extend(["--reward_path", args.reward_path])
        if args.allocated_gpus is not None:
            train_cmd.extend(["--allocated_gpus", str(args.allocated_gpus)])
            
        with resume_arguments(checkpoint_dir, resume_step, resume_step == gstep) as resume_args:
            train_cmd.extend(resume_args)
            print("Executing:", " ".join(train_cmd))
            subprocess.run(train_cmd, check=True)
        if (checkpoint_dir / TRACKER).read_text().strip() != str(end_step):
            raise RuntimeError(f"Trainer did not reach the requested checkpoint at step {end_step}")
        mark_complete(checkpoint_dir, end_step, args.merge_backend)
        resume_step = end_step


if __name__ == "__main__":
    main()


