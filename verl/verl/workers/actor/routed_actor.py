"""Opt-in vanilla GRPO actor for the synchronous, padded FSDP routing experiment."""
import math

import torch

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.graft_lora import RouteGradients


class RoutedPPOActor(DataParallelPPOActor):
    def __init__(self, *args, forget_grad_scale=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.forget_grad_scale = forget_grad_scale
        if not math.isfinite(forget_grad_scale) or forget_grad_scale <= 0:
            raise ValueError("forget_grad_scale must be positive and finite")
        if self.use_remove_padding or self.use_ulysses_sp or self.config.use_dynamic_bsz:
            raise ValueError("Routed actor requires padded, static micro-batches and sequence parallel size 1")
        if self.config.policy_loss.loss_mode != "vanilla":
            raise ValueError("Phase-one routing supports the vanilla GRPO policy loss")

    def update_policy(self, data):
        if "route_tag" not in data.batch:
            raise ValueError("Routed actor requires integer route_tag from train.parquet")
        self.actor_module.train()
        cfg = self.config
        mini_size, micro_size = cfg.ppo_mini_batch_size, cfg.ppo_micro_batch_size_per_gpu
        if mini_size % micro_size or len(data) % mini_size:
            raise ValueError("Routed actor requires complete mini-batches divisible by the per-GPU micro-batch")
        minibatches = data.split(mini_size)
        on_policy = len(minibatches) == 1 and cfg.ppo_epochs == 1 and not cfg.get("use_rollout_log_probs", False)
        loss_fn = get_policy_loss_fn("vanilla")
        metrics = {}
        for _ in range(cfg.ppo_epochs):
            for minibatch in minibatches:
                self.actor_optimizer.zero_grad(set_to_none=True)
                gradients = RouteGradients(self.actor_module, self.device_name)
                for micro in minibatch.split(micro_size):
                    batch = micro.to(self.device_name).batch
                    # Keep the route live through non-reentrant checkpoint replay.
                    with gradients.micro_batch(batch["route_tag"], self.forget_grad_scale):
                        entropy, log_prob = self._forward_micro_batch(
                            batch, data.meta_info["temperature"], calculate_entropy=cfg.entropy_coeff != 0
                        )
                        mask = batch["response_mask"]
                        pg, clipped, kl, lower = loss_fn(
                            old_log_prob=log_prob.detach() if on_policy else batch["old_log_probs"],
                            log_prob=log_prob, advantages=batch["advantages"], response_mask=mask,
                            loss_agg_mode=cfg.loss_agg_mode, config=cfg, rollout_is_weights=batch.get("rollout_is_weights"),
                        )
                        weight = micro_size / mini_size
                        loss = pg
                        if cfg.entropy_coeff:
                            loss = loss - cfg.entropy_coeff * agg_loss(entropy, mask, cfg.loss_agg_mode)
                        if cfg.use_kl_loss:
                            kld = kl_penalty(log_prob, batch["ref_log_prob"], cfg.kl_loss_type)
                            kl_loss = agg_loss(kld, mask, cfg.loss_agg_mode)
                            loss = loss + cfg.kl_loss_coef * kl_loss
                            append_to_dict(metrics, {"actor/kl_loss": weight * kl_loss.detach().item(),
                                                     "actor/kl_coef": cfg.kl_loss_coef})
                        (loss * weight).backward()
                        append_to_dict(metrics, {"actor/pg_loss": weight * pg.detach().item(),
                                                 "actor/pg_clipfrac": clipped.item(), "actor/ppo_kl": kl.item(),
                                                 "actor/pg_clipfrac_lower": lower.item()})
                append_to_dict(metrics, gradients.finish())
                append_to_dict(metrics, {"actor/grad_norm": self._optimizer_step().item()})
        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics
