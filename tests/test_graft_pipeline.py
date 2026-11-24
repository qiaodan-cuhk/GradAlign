import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "automated"))
import aggregate
import config
import dynamic_selection as runner
import launch_verl_training as launch
import select_data
from checkpoints import TRACKER
from export_graft import export_policy
from test_graft_actor import tiny_base, trained_pair
from test_selection_baseline import read_ids, write_dataset
from verl import DataProto
from verl.trainer.ppo.graft_trainer import GraftValidationTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.dataset.routed_dataset import RoutedRLHFDataset
from verl.utils.config import validate_config
from verl.workers.actor.graft_lora import save_packed_adapter


@pytest.fixture(autouse=True)
def dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")


def tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token="<unk>")),
                                  unk_token="<unk>", pad_token="<pad>")
    tok.chat_template = "{% for m in messages %}{{m.content}}{% endfor %}"
    return tok


def test_parquet_route_survives_real_generation_pop_repeat_rebalance_and_split(tmp_path):
    rows = [{"extra_info": {"index": i}, "prompt": [{"role": "user", "content": f"Q{i}"}],
             "data_source": "math", "route_tag": tag} for i, tag in enumerate([2, 0, 1])]
    path = tmp_path / "train.parquet"
    datasets.Dataset.from_list(rows).to_parquet(str(path))
    ds = RoutedRLHFDataset(str(path), tokenizer(), OmegaConf.create(
        {"max_prompt_length": 8, "filter_overlong_prompts": False, "cache_dir": str(tmp_path / "cache")}))
    batch = DataProto.from_single_dict(collate_fn([ds[i] for i in range(3)]))
    gen = RayPPOTrainer._get_gen_batch(SimpleNamespace(async_rollout_mode=False), batch)
    assert "route_tag" not in gen.batch
    training = batch.repeat(repeat_times=16, interleave=True)
    training.reorder(torch.arange(47, -1, -1))
    for micro in training.split(2):
        for row, tag in zip(micro.non_tensor_batch["extra_info"], micro.batch["route_tag"], strict=True):
            assert tag.item() == rows[row["index"]]["route_tag"]


def test_full_and_stripped_hf_exports_reload_outside_peft(tmp_path):
    base = tiny_base()
    base.save_pretrained(tmp_path / "base")
    tokenizer().save_pretrained(tmp_path / "base")
    model = trained_pair(copy.deepcopy(base)).eval()
    save_packed_adapter(model, tmp_path / "actor")
    ids = torch.tensor([[1, 2, 3]])
    full = model(ids).logits
    model.base_model.set_adapter("default")
    retain = model(ids).logits
    assert not torch.allclose(full, retain)
    for state, expected in (("full", full), ("retain_only", retain)):
        dest = tmp_path / "merged" / state
        export_policy(tmp_path / "base", tmp_path / "actor/graft_adapter", dest, state)
        loaded = AutoModelForCausalLM.from_pretrained(dest)
        torch.testing.assert_close(loaded(ids).logits, expected)
        assert not hasattr(loaded, "peft_config") and (dest / "tokenizer.json").is_file()
    with pytest.raises(FileExistsError):
        export_policy(tmp_path / "base", tmp_path / "actor/graft_adapter", tmp_path / "merged/full", "retain_only")


@pytest.mark.parametrize("fail", [False, True])
def test_validation_restores_full_policy_and_records_both_states(tmp_path, monkeypatch, fail):
    states = ["full"]
    wg = SimpleNamespace(set_graft_rollout_state=states.append)
    graft = {"mode": "routed"}
    trainer = SimpleNamespace(actor_rollout_wg=wg, global_steps=5, config=OmegaConf.create(
        {"actor_rollout_ref": {"graft": graft}, "trainer": {"default_local_dir": str(tmp_path)}}))
    def validate(self):
        if fail and states[-1] == "retain_only":
            raise RuntimeError("generation failed")
        return {"val-core/amc23/reward/mean@1": 0.4 if states[-1] == "full" else 0.5}
    monkeypatch.setattr(RayPPOTrainer, "_validate", validate)
    # Call the real subclass method (super() requires an actual trainer instance).
    actual = object.__new__(GraftValidationTrainer)
    actual.__dict__.update(trainer.__dict__)
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            actual._validate()
    else:
        metrics = actual._validate()
        assert set(metrics.values()) == {0.4, 0.5}
        assert json.loads((tmp_path / "heldout_metrics.jsonl").read_text())["grpo_step"] == 5
    assert states == ["full", "retain_only", "full"]


def test_three_pilot_yamls_through_orchestrator_and_hydra(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(config, "BASE_DATA_DIR", str(tmp_path))
    for label, size in (("amc23", 40), ("amc22_ref16", 16), ("webinstruct_math80", 80)):
        write_dataset(tmp_path, label, range(size))
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base snapshot identity")
    roots = []
    hydrated = []
    analyses = []
    inference_models = []
    def dispatch(cmd, check):
        stage = Path(cmd[1]).name
        def option(key):
            return cmd[cmd.index(key) + 1]
        if stage == "run_inference_local.py":
            inference_models.append(Path(option("--model_path")))
            ids = read_ids(Path(option("--prompts_file")).parent)
            n = int(option("--n_samples"))
            data = [{"group_id": i, "sample_id": s, "passed": bool(s % 2), "prompt": str(i), "response": str(s)}
                    for i in ids for s in range(n)]
            (Path(option("--output_dir")) / "responses_sorted.json").write_text(json.dumps(data))
        elif stage == "run_parallel_analysis.py":
            analyses.append(option("--num_gpus"))
            ids = read_ids(Path(option("--problem_set_path")).parent)
            path = Path(option("--parts_root")) / "part_0/similarity_results_cosine_real.jsonl"
            path.write_text("".join(json.dumps({"group_id": i, "similarity": i / 80}) + "\n" for i in ids))
        elif stage in ("aggregate.py", "select_data.py", "launch_verl_training.py"):
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", cmd[1:] + (["--num_workers", "1"] if stage == "aggregate.py" else []))
                {"aggregate.py": aggregate.main, "select_data.py": select_data.main,
                 "launch_verl_training.py": launch.main}[stage]()
        elif cmd[1:3] == ["-m", "verl.trainer.main_ppo"]:
            with initialize_config_dir(config_dir=str(launch.VERL_PACKAGE / "trainer/config"), version_base=None):
                cfg = compose(config_name="ppo_trainer", overrides=cmd[3:])
            validate_config(cfg, use_reference_policy=True, use_critic=False)
            assert cfg.trainer.n_gpus_per_node == 4 and cfg.actor_rollout_ref.actor.strategy == "fsdp"
            assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
            assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 2
            assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 8 and cfg.data.train_batch_size == 8
            assert cfg.actor_rollout_ref.rollout.n == 16 and cfg.actor_rollout_ref.rollout.val_kwargs.n == 1
            assert not cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
            assert cfg.actor_rollout_ref.model.lora_rank == 8
            assert cfg.actor_rollout_ref.rollout.seed == cfg.data.seed == 42
            assert cfg.actor_rollout_ref.model.path == str(base)
            hydrated.append(cfg.actor_rollout_ref.graft.mode)
            root = Path(cfg.trainer.default_local_dir)
            roots.append(root)
            end = cfg.trainer.total_training_steps
            if end == 20:
                resume = Path(cfg.trainer.resume_from_path)
                assert resume.name == "global_step_10" and not (resume / "data.pt").exists()
                assert (resume / "actor").resolve() == root / "global_step_10/actor"
            step = root / f"global_step_{end}"
            ckpt = step / "actor"
            ckpt.mkdir(parents=True)
            (step / "data.pt").write_bytes(b"sampler")
            (ckpt / "fsdp_config.json").write_text('{"world_size": 4}')
            for rank in range(4):
                for kind in ("model", "optim", "extra_state"):
                    (ckpt / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"shard")
            adapter = ckpt / "graft_adapter"
            adapter.mkdir()
            from safetensors.torch import save_file
            (adapter / "adapter_config.json").write_text('{"r": 8}')
            save_file({"q.lora_A.weight": torch.zeros(8, 1), "q.lora_B.weight": torch.zeros(1, 8)},
                      adapter / "adapter_model.safetensors")
            (root / TRACKER).write_text(str(end))
        elif stage == "export_graft.py":
            dest = Path(option("--output_dir"))
            dest.mkdir(parents=True)
            (dest / "config.json").write_text("{}")
            (dest / "model.safetensors").write_bytes(b"merged")
            (dest / "tokenizer_config.json").write_text("{}")
            (dest / "graft_state.txt").write_text(option("--state"))
        else:
            pytest.fail(f"Unexpected command: {cmd}")
    monkeypatch.setattr(runner.subprocess, "run", dispatch)
    for arm in ("a_hard", "b_all", "c_route"):
        cfg_path = repo / "configs" / f"graft_{arm}.yaml"
        monkeypatch.setattr(sys, "argv", ["run", "--config", str(cfg_path), "--model_path", str(base),
                                         "--ckpt_root", str(tmp_path / "runs")])
        runner.main()
    assert hydrated == ["paired", "paired", "routed"] and analyses == ["4"]
    selected = [next((root / "selected").iterdir()) for root in roots]
    a, b, c = selected
    assert read_ids(b) == read_ids(c) and len(read_ids(c)) == 80
    routes = [json.loads(line) for line in (c / "train.jsonl").read_text().splitlines()]
    assert set(read_ids(a)) == {r["extra_info"]["index"] for r in routes if r["route_tag"] == 1}
    assert [sum(r["route_tag"] == i for r in routes) for i in (1, 0, 2)] == [16, 48, 16]
    assert json.loads((c / "initial_oracle.json").read_text())["reused"]
    assert (roots[2] / "global_step_10/merged/retain_only/model.safetensors").exists()
    assert inference_models == [base, base]
    monkeypatch.setattr(sys, "argv", sys.argv + ["--num_selections", "2"])
    runner.main()
    assert inference_models[2:] == [roots[2] / "global_step_10/merged/full"] * 2
    assert analyses == ["4", "4"]
    assert (roots[2] / "global_step_20/merged/retain_only/model.safetensors").exists()
