#!/usr/bin/env python3
"""
Launcher script for parallel GRPO gradient analysis.
This script configures and launches the distributed analysis across multiple GPUs.
"""

import argparse
import subprocess
import sys
import os
import tempfile
import json
from pathlib import Path


def create_accelerate_config(num_gpus: int = 8, mixed_precision: str = "bf16"):
    """Create accelerate configuration for FSDP training."""
    config = f"""compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
{f"mixed_precision: {mixed_precision}" if mixed_precision != "None" else ""}
num_machines: 1
num_processes: {num_gpus}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    return config


def main():
    parser = argparse.ArgumentParser(description="Launch Parallel GRPO Gradient Analysis")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to analyze")
    parser.add_argument("--train_responses_dir", type=str, required=True,
                       help="Directory containing pre-generated train responses")
    parser.add_argument("--val_responses_dir", type=str, required=True,
                       help="Directory containing pre-generated validation responses")
    parser.add_argument("--output_path", type=str, 
                       default="~/data_selection/data/analysis/gradient_similarities_parallel.json",
                       help="Output file for similarity results")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="Number of GPUs to use")
    parser.add_argument("--kl_loss_coef", type=float, default=0.001,
                       help="KL loss coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini batch size for gradient computation")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       help="Mixed precision mode")
    parser.add_argument("--fsdp_auto_wrap_policy", type=str,
                       default="transformer_auto_wrap_policy",
                       help="FSDP auto wrap policy")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Offload parameters to CPU when not in use")
    parser.add_argument("--distribute_val_data", action="store_true",
                       help="Distribute validation data across processes (vs main process only)")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to accelerate config file (will create one if not provided)")
    parser.add_argument("--max_num_samples", type=int, default=None,
                       help="Maximum number of samples to load from each dataset")
    parser.add_argument("--problem_set_path", type=str, required=True,
                       help="Path to the problem set file (train.jsonl) containing ground truth")
    parser.add_argument("--norm_before_accumlation", action="store_true", default=False,
                       help="Normalize the gradients before accumulation")
    parser.add_argument("--use_optimizer", action="store_true", default=False,
                       help="Scale validation gradients with optimizer exp_avg_sq state")
    parser.add_argument("--optimizer_state_path", type=str, default=None,
                       help="Path to converted optimizer state (required when --use_optimizer)")
    parser.add_argument("--mode", default='sim', type=str)
    args = parser.parse_args()

    if args.use_optimizer and not args.optimizer_state_path:
        parser.error("--optimizer_state_path must be provided when --use_optimizer is set")

    for directory in dict.fromkeys([args.train_responses_dir, args.val_responses_dir]):
        folder = Path(directory)
        stat = (folder / "responses_sorted.json").stat()
        manifest = folder / "data" / "shards.json"
        expected = {"version": 2, "world_size": args.num_gpus, "model": args.model_path,
                    "max_length": args.max_length, "max_num_samples": args.max_num_samples,
                    "responses": [stat.st_size, stat.st_mtime_ns]}
        if (manifest.exists() and json.loads(manifest.read_text()) == expected
                and all((manifest.parent / f"data_{rank}.npz").is_file() for rank in range(args.num_gpus))):
            continue
        # Only publish the version marker after all ranks' files have been written.
        manifest.unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__).with_name("prepare.py")),
               "--responses_dir", directory, "--model_path", args.model_path,
               "--world_size", str(args.num_gpus), "--max_length", str(args.max_length)]
        if args.max_num_samples is not None:
            cmd.extend(["--max_num_samples", str(args.max_num_samples)])
        subprocess.run(cmd, check=True)
        tmp = manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(expected))
        tmp.replace(manifest)

    # Create accelerate config if not provided
    if args.config_file is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="gradalign-", delete=False) as config:
            config.write(create_accelerate_config(args.num_gpus, args.mixed_precision))
            config_file = config.name
    else:
        config_file = args.config_file
    
    # Build the accelerate launch command
    launch_cmd = [
        "accelerate", "launch",
        "--config_file", config_file,
        *(["--main_process_port", "0"] if args.config_file is None else []),
        os.path.join(os.path.dirname(__file__), "analyze_grpo_parallel.py"),
        "--model_path", args.model_path,
        "--train_responses_dir", args.train_responses_dir,
        "--val_responses_dir", args.val_responses_dir,
        "--output_path", args.output_path,
        "--kl_loss_coef", str(args.kl_loss_coef),
        "--clip_ratio", str(args.clip_ratio),
        "--mini_batch_size", str(args.mini_batch_size),
        "--max_length", str(args.max_length),
        "--mixed_precision", args.mixed_precision,
        "--fsdp_auto_wrap_policy", args.fsdp_auto_wrap_policy,
        "--problem_set_path", args.problem_set_path,
        "--mode", args.mode,
    ]
    
    if args.max_num_samples is not None:
        launch_cmd.extend(["--max_num_samples", str(args.max_num_samples)])

    # Add optional flags
    if args.norm_before_accumlation:
        launch_cmd.append("--norm_before_accumlation")
    if args.use_gradient_checkpointing:
        launch_cmd.append("--use_gradient_checkpointing")
    if args.cpu_offload:
        launch_cmd.append("--cpu_offload")
    if args.distribute_val_data:
        launch_cmd.append("--distribute_val_data")
    if args.use_optimizer:
        launch_cmd.append("--use_optimizer")
    if args.optimizer_state_path is not None:
        launch_cmd.extend(["--optimizer_state_path", args.optimizer_state_path])
    
    print("Launching parallel GRPO gradient analysis...")
    print(f"Command: {' '.join(launch_cmd)}")
    # exit()
    print(f"Using {args.num_gpus} GPUs with {args.mixed_precision} precision")
    print("="*80)
    
    # Run the command
    try:
        subprocess.run(launch_cmd, check=True)
        print("="*80)
        print("Parallel analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running parallel analysis: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    finally:
        # Clean up temporary config file if we created it
        if args.config_file is None and os.path.exists(config_file):
            os.remove(config_file)
            print(f"Cleaned up temporary config file: {config_file}")


if __name__ == "__main__":
    main() 