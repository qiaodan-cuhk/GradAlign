"""Numerical contracts for sample routing, optimizer isolation and the deployed policy."""
import copy

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.workers.actor.graft_lora import RouteGradients, attach_adapters, packed_adapter


def tiny_base():
    torch.manual_seed(5)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config)


def tiny_pair(base, routed=True):
    cfg = LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    return attach_adapters(base, cfg, routed, 42)


def trained_pair(base):
    model = tiny_pair(base)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    return model


def test_mixed_route_matches_per_sample_full_policy_derivatives():
    base = tiny_base()
    model = trained_pair(copy.deepcopy(base))
    reference = tiny_pair(base, routed=False)
    reference.load_state_dict(model.state_dict())
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [2, 1, 5], [4, 1, 8]])
    tags = torch.tensor([2, 0, 1, 2])
    expected = {n: torch.zeros_like(p) for n, p in reference.named_parameters() if p.requires_grad}
    for row, tag in zip(tokens, tags.tolist(), strict=True):
        reference.zero_grad()
        (reference(row[None]).logits.square().sum() / tokens.numel()).backward()
        for name, param in reference.named_parameters():
            if name in expected:
                multiplier = (1, int(".default." in name), 2 * int(".forget." in name))[tag]
                expected[name].add_(param.grad, alpha=multiplier)
    gradients = RouteGradients(model, "cpu")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with gradients.micro_batch(tags, 2.0):
        (model(tokens, use_cache=False).logits.square().sum() / tokens.numel()).backward()
    metrics = gradients.finish()
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, expected[name], rtol=1e-4, atol=2e-7)
        else:
            assert p.grad is None
    assert metrics["route/forget_only_fraction"] == 0.5
    assert all(value > 0 for value in metrics.values())


def test_exclusive_step_preserves_inactive_adam_momentum_decay_and_weights():
    model = trained_pair(tiny_base())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.1)
    tokens = torch.tensor([[1, 2, 3], [4, 1, 8]])
    for tag in (0, 1, 2):
        optimizer.zero_grad(set_to_none=True)
        before = {n: (p.detach().clone(), copy.deepcopy(optimizer.state[p]))
                  for n, p in model.named_parameters() if p.requires_grad}
        gradients = RouteGradients(model, "cpu")
        with gradients.micro_batch(torch.full((2,), tag), 2.0):
            model(tokens).logits.square().sum().backward()
        gradients.finish()
        optimizer.step()
        for n, p in model.named_parameters():
            inactive = (tag == 1 and ".forget." in n) or (tag == 2 and ".default." in n)
            if inactive:
                weight, state = before[n]
                assert torch.equal(p, weight)
                assert all(torch.equal(optimizer.state[p][key], value) for key, value in state.items())


@pytest.mark.parametrize("state", ["full", "retain_only"])
def test_packed_vllm_payload_matches_actor_and_merged_hf(state):
    base = tiny_base()
    model = trained_pair(copy.deepcopy(base)).eval()
    config, weights = packed_adapter(model, state)
    inferred = get_peft_model(copy.deepcopy(base), config)
    set_peft_model_state_dict(inferred, weights)
    if state == "retain_only":
        model.base_model.set_adapter("default")
    tokens = torch.tensor([[3, 4, 5]])
    expected = model(tokens).logits
    torch.testing.assert_close(inferred(tokens).logits, expected)
    torch.testing.assert_close(inferred.merge_and_unload()(tokens).logits, expected)


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def policy_batch(tags):
    from verl import DataProto

    n = len(tags)
    ids = torch.arange(n * 5).reshape(n, 5) % 31
    return DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "position_ids": torch.arange(5).expand(n, -1),
                 "responses": ids[:, -2:], "response_mask": torch.tensor([[1, 0], [1, 1]]).repeat(n // 2, 1),
                 "old_log_probs": torch.zeros(n, 2), "ref_log_prob": torch.full((n, 2), -3.3),
                 "advantages": torch.linspace(-1, 1, n)[:, None].expand(-1, 2), "route_tag": torch.tensor(tags)},
        meta_info={"temperature": 1.0},
    )


def actor_cfg(mini=4):
    from verl.workers.config.actor import FSDPActorConfig

    return FSDPActorConfig(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=mini,
                           ppo_micro_batch_size_per_gpu=2, use_torch_compile=False,
                           entropy_coeff=0.01, use_kl_loss=True)


def test_all_both_matches_vanilla_grpo_update_with_masked_tokens(process_group):
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.actor.routed_actor import RoutedPPOActor

    base = tiny_base()
    routed = trained_pair(copy.deepcopy(base))
    vanilla = tiny_pair(base, routed=False)
    vanilla.load_state_dict(routed.state_dict())
    cfg = actor_cfg()
    for cls, model in [(DataParallelPPOActor, vanilla), (RoutedPPOActor, routed)]:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        actor = cls(cfg, model, optimizer)
        metrics = actor.update_policy(policy_batch([0] * 4))
        assert metrics["actor/grad_norm"][0] > 0
    for a, b in zip(routed.parameters(), vanilla.parameters(), strict=True):
        # The routed reduction uses fp32 for the adapter matmul; CPU autocast is bf16.
        torch.testing.assert_close(a, b, atol=4e-5, rtol=2e-3)


def _distributed_worker(rank, world, path):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from verl.workers.actor.routed_actor import RoutedPPOActor

    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world)
    try:
        base = tiny_base()
        model = trained_pair(copy.deepcopy(base))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy = get_fsdp_wrap_policy(model, is_lora=True)
        sharded = FSDP(model, device_id=torch.device("cpu"), use_orig_params=True, auto_wrap_policy=policy)
        actor = RoutedPPOActor(actor_cfg(2), sharded, torch.optim.AdamW(sharded.parameters(), lr=0.002))
        initial = packed_adapter(sharded)[1]
        tags = [[1, 1], [1, 0], [2, 2], [2, 0]][rank]
        metrics = actor.update_policy(policy_batch(tags))
        assert metrics["route/forget_only_fraction"] == [3 / 8]
        _, full = packed_adapter(sharded)
        assert any(not torch.equal(full[k], v) for k, v in initial.items())
        actor.update_policy(policy_batch([1, 1]))
        _, retain_step = packed_adapter(sharded)
        for k, v in full.items():
            axis = 0 if ".lora_A." in k else 1
            assert torch.equal(v.chunk(2, dim=axis)[1], retain_step[k].chunk(2, dim=axis)[1])
        # A real sharded optimizer + model round trip (including per-branch Adam moments).
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models
        tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
        manager = FSDPCheckpointManager(model=sharded, optimizer=actor.actor_optimizer,
                                        lr_scheduler=None, processing_class=tok)
        manager.save_checkpoint(str(path.parent / "actor"), global_step=2)
        actor.update_policy(policy_batch(tags))
        after = packed_adapter(sharded)[1]
        manager.load_checkpoint(str(path.parent / "actor"))
        actor.update_policy(policy_batch(tags))
        replay = packed_adapter(sharded)[1]
        assert all(torch.equal(v, replay[k]) for k, v in after.items())
    finally:
        dist.destroy_process_group()


def test_four_rank_cpu_fsdp_update_pack_and_resume(tmp_path):
    torch.multiprocessing.start_processes(_distributed_worker, args=(4, tmp_path / "gloo"),
                                          nprocs=4, join=True, start_method="fork")
