"""Two ordinary PEFT adapters during training, one equivalent adapter for inference."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLinear
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

BRANCHES = ("default", "forget")
STATES = ("full", "retain_only")
_route = ContextVar("graft_micro_batch", default=None)


class RouteGradients:
    """Accumulate the pre-clip contributions per tag over an entire optimizer step."""

    def __init__(self, model, device):
        self.layers = [layer for layer in model.modules() if isinstance(layer, RoutedLinear)]
        if not self.layers:
            raise ValueError("Routed actor requires trainable retain and forget adapters")
        n = 0
        for layer in self.layers:
            size = layer.in_features * layer.out_features
            layer.grad_slice = slice(n, n + size)
            n += size
        self.by_tag = torch.zeros(3, n, device=device, dtype=torch.float32)
        self.counts = torch.zeros(3, device=device, dtype=torch.long)

    @contextmanager
    def micro_batch(self, tags, forget_scale):
        if tags.ndim != 1 or tags.dtype != torch.long or not ((tags >= 0) & (tags <= 2)).all():
            raise ValueError("route_tag must contain one integer in {0,1,2} per response")
        self.counts.add_(torch.bincount(tags, minlength=3))
        indices = [torch.where(tags == i)[0] for i in range(3)]
        token = _route.set((indices, tags.numel(), forget_scale, self.by_tag))
        try:
            yield
        finally:
            _route.reset(token)

    def finish(self):
        if dist.is_initialized():
            dist.all_reduce(self.counts)
            dist.all_reduce(self.by_tag)
            self.by_tag.div_(dist.get_world_size())
        # With AdamW, even a zero gradient can move an inactive parameter via
        # momentum or decoupled decay. None skips both the update and its clock.
        counts = self.counts.tolist()
        for layer in self.layers:
            if counts[0] + counts[layer.exclusive_tag] == 0:
                layer.weight.grad = None
        metrics = {}
        for tag, name in enumerate(("both", "retain_only", "forget_only")):
            metrics[f"route/{name}_fraction"] = counts[tag] / sum(counts)
            metrics[f"route/{name}_grad_norm"] = self.by_tag[tag].norm().item()
        return metrics


class _RoutedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, layer, routing):
        ctx.save_for_backward(inputs, weight)
        ctx.layer, ctx.routing = layer, routing
        return F.linear(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        layer = ctx.layer
        indices, batch_size, forget_scale, contributions = ctx.routing
        if inputs.shape[0] != batch_size:
            raise ValueError("GRAFT requires padded batch-major activations")
        # Propagate the full policy derivative into earlier layers. Gate only
        # this linear's parameter gradient, so A and B do not scale it twice.
        grad_input = grad_output.matmul(weight.to(grad_output.dtype)) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        for tag in (0, layer.exclusive_tag):
            rows = indices[tag]
            if rows.numel():
                x = inputs.index_select(0, rows).reshape(-1, layer.in_features)
                dy = grad_output.index_select(0, rows).reshape(-1, layer.out_features)
                partial = dy.float().T @ x.float()
                if tag == 2:
                    partial.mul_(forget_scale)
                contributions[tag, layer.grad_slice].add_(partial.flatten())
                grad_weight.add_(partial)
        return grad_input, grad_weight.to(weight.dtype), None, None


class RoutedLinear(nn.Linear):
    """A PEFT linear factor with the same state-dict key and forward values."""

    def __init__(self, original, exclusive_tag):
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight = original.weight
        self.register_parameter("bias", None)
        self.exclusive_tag = exclusive_tag

    def forward(self, inputs):
        routing = _route.get()
        if routing is None:
            return F.linear(inputs, self.weight)
        return _RoutedLinearFunction.apply(inputs, self.weight, self, routing)


def attach_adapters(base, packed_config: LoraConfig, routed: bool, seed: int):
    if packed_config.r % 2 or packed_config.lora_alpha % 2:
        raise ValueError("Paired LoRA requires an even packed rank and alpha")
    config = replace(packed_config, r=packed_config.r // 2, lora_alpha=packed_config.lora_alpha // 2)
    # Initial adapter tensors take the same values in all three arms.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = get_peft_model(base, config)
        model.add_adapter("forget", replace(config))
    model.base_model.set_adapter(list(BRANCHES))
    if routed:
        for layer in model.modules():
            if isinstance(layer, PeftLinear):
                for factors in (layer.lora_A, layer.lora_B):
                    for tag, branch in enumerate(BRANCHES, start=1):
                        factors[branch] = RoutedLinear(factors[branch], tag)
    return model


@torch.no_grad()
def packed_adapter(model, state="full"):
    """Gather adapter tensors from FSDP1 and pack B_R A_R + B_F A_F at unchanged alpha/r."""
    if state not in STATES:
        raise ValueError(f"Unknown GRAFT inference state {state}")
    gather = FSDP.summon_full_params(model, writeback=False) if isinstance(model, FSDP) else nullcontext()
    with gather:
        peft_model = model.module if isinstance(model, FSDP) else model
        config = peft_model.peft_config["default"]
        packed = replace(config, r=2 * config.r, lora_alpha=2 * config.lora_alpha, inference_mode=True)
        tensors = {}
        for name, layer in peft_model.named_modules():
            if not isinstance(layer, PeftLinear):
                continue
            name = name.replace("_fsdp_wrapped_module.", "")
            ar, af = (layer.lora_A[k].weight.detach().cpu() for k in BRANCHES)
            br, bf = (layer.lora_B[k].weight.detach().cpu() for k in BRANCHES)
            if state == "retain_only":
                bf = torch.zeros_like(bf)
            tensors[f"{name}.lora_A.weight"] = torch.cat([ar, af], dim=0)
            tensors[f"{name}.lora_B.weight"] = torch.cat([br, bf], dim=1)
        if not tensors:
            raise RuntimeError("No LoRA weights found in the actor")
    return packed, tensors


def save_packed_adapter(model, actor_directory):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from safetensors.torch import save_file

    config, tensors = packed_adapter(model)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    directory = Path(actor_directory) / "graft_adapter"
    directory.mkdir(parents=True, exist_ok=True)
    # Rank shards remain the authoritative optimizer + model training checkpoint.
    # Publish the small deployment payload before the trainer advances its tracker.
    with TemporaryDirectory(dir=directory) as temp:
        stage = Path(temp)
        config.save_pretrained(stage)
        save_file(tensors, stage / "adapter_model.safetensors")
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (stage / name).replace(directory / name)
