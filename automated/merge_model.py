#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
from pathlib import Path


def build_local_dir(experiment_name: str, step: int, ckpt_root: str) -> str:
    return f"{ckpt_root}/{experiment_name}/global_step_{step}/actor"


def build_target_dir(output_model_name: str, step: int, dest_root: str) -> str:
    return f"{dest_root}/{output_model_name}_{step}"


def run_merge(local_dir: str, target_dir: str, backend: str = "megatron") -> None:
    Path(target_dir).parent.mkdir(parents=True, exist_ok=True)
    launcher = [sys.executable]
    if backend == "megatron" and "WORLD_SIZE" not in os.environ:
        # The upstream standalone fallback binds a fixed port. Let torchrun
        # allocate and own this job's rendezvous until the merge child exits.
        launcher.extend(["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1"])
    cmd = [
        *launcher, "-m", "verl.model_merger", "merge",
        "--backend", backend,
        "--local_dir", local_dir,
        "--target_dir", target_dir,
    ]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Merge VERL checkpoints into a final model directory.")
    parser.add_argument("--experiment_name", required=True, type=str, help="Experiment name under chkpt root")
    parser.add_argument("--step", required=True, type=int, help="Global step to merge (e.g., 100)")
    parser.add_argument("--output_model_name", required=True, type=str, help="Base name for merged model output")
    parser.add_argument("--backend", default="megatron", type=str, help="Merger backend (default: megatron)")
    parser.add_argument("--target_dir", default=None, type=str, help="If set, write merged model exactly to this path")
    parser.add_argument(
        "--ckpt_root",
        default="",
        type=str,
        help="Root directory containing experiment checkpoints",
    )
    parser.add_argument(
        "--dest_root",
        default="",
        type=str,
        help="Root directory to place merged models",
    )

    args = parser.parse_args()

    local_dir = build_local_dir(args.experiment_name, args.step, args.ckpt_root)
    if args.target_dir:
        target_dir = args.target_dir
    else:
        target_dir = build_target_dir(args.output_model_name, args.step, args.dest_root)

    if not os.path.isdir(local_dir):
        raise SystemExit(f"Local checkpoint directory does not exist: {local_dir}")

    run_merge(local_dir, target_dir, backend=args.backend)
    print(f"Merged model written to: {target_dir}")


if __name__ == "__main__":
    main()


