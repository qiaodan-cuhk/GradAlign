"""Evaluate paired LoRA checkpoints with the existing held-out generation and reward loop."""
import json
import time
from pathlib import Path

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class GraftValidationTrainer(RayPPOTrainer):
    def _validate(self):
        full = super()._validate()
        metrics = {f"val_full/{key}": value for key, value in full.items()}
        if self.config.actor_rollout_ref.graft.mode == "routed":
            try:
                self.actor_rollout_wg.set_graft_rollout_state("retain_only")
                stripped = super()._validate()
                metrics.update({f"val_retain_only/{key}": value for key, value in stripped.items()})
            finally:
                self.actor_rollout_wg.set_graft_rollout_state("full")
        graft = self.config.actor_rollout_ref.graft
        if graft.get("allocated_gpus") is not None:
            metrics["cost/cumulative_gpu_hours"] = (
                graft.prior_gpu_hours + (time.time() - graft.start_time) * graft.allocated_gpus / 3600
            )
            metrics["cost/standalone_gpu_hours"] = metrics["cost/cumulative_gpu_hours"] + graft.standalone_adjustment
        root = Path(self.config.trainer.default_local_dir)
        with (root / "heldout_metrics.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"grpo_step": self.global_steps, **metrics}) + "\n")
        return metrics
