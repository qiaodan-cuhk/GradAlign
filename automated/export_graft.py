"""Export ordinary Hugging Face policies from a paired adapter and its frozen base."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from checkpoints import hf_weights_ready


@torch.inference_mode()
def export_policy(base_path, adapter_path, destination, state):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if state not in ("full", "retain_only"):
        raise ValueError(f"Unknown adapter state {state}")
    destination = Path(destination)
    if hf_weights_ready(destination):
        return
    cfg = LoraConfig.from_pretrained(adapter_path)
    tensors = load_file(Path(adapter_path) / "adapter_model.safetensors")
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")
    model = get_peft_model(base, cfg)
    expected = {n.replace(".default.", ".") for n, _ in model.named_parameters() if ".lora_" in n}
    if not expected or set(tensors) != expected or cfg.r % 2:
        raise ValueError("Packed adapter keys or rank do not match the frozen base")
    if state == "retain_only":
        for name, value in tensors.items():
            if name.endswith(".lora_B.weight"):
                value[:, cfg.r // 2:] = 0
    set_peft_model_state_dict(model, tensors)
    merged = model.merge_and_unload(safe_merge=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "policy"
        merged.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        if not hf_weights_ready(stage):
            raise RuntimeError(f"Could not export {state} policy")
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--state", choices=["full", "retain_only"], required=True)
    args = parser.parse_args()
    export_policy(args.base_model, args.adapter_dir, args.output_dir, args.state)


if __name__ == "__main__":
    main()
