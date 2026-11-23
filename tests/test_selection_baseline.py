"""CPU contract for the real selection entry point; GPU stages are replaced at dispatch."""
import json
import sys
from pathlib import Path

import datasets
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "automated"))
import config
import dynamic_selection
import select_data


def write_dataset(root, name, ids):
    directory = root / name
    directory.mkdir()
    rows = [
        {"data_source": name, "prompt": [{"role": "user", "content": f"{name} question {i}"}],
         "reward_model": {"ground_truth": str(i)}, "extra_info": {"index": i}}
        for i in ids
    ]
    (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    datasets.Dataset.from_list(rows).to_parquet(str(directory / "train.parquet"))
    return directory


def read_ids(directory):
    return [json.loads(line)["extra_info"]["index"] for line in (directory / "train.jsonl").read_text().splitlines()]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    monkeypatch.setattr(config, "BASE_DATA_DIR", str(tmp_path))
    write_dataset(tmp_path, "train", range(100, 114))
    write_dataset(tmp_path, "reference", [0, 1])
    write_dataset(tmp_path, "heldout", [0, 1])
    return [
        "dynamic_selection", "--prefix", "toy", "--model", "toy-model", "--train_dataset", "train",
        "--val_dataset", "reference", "--verl_val_set", "heldout", "--chunk_size", "6", "--k", "2",
        "--train_batch_size", "3", "--iters_per_select", "1", "--num_selections", "1",
        "--mode", "rand", "--seed", "42", "--rollout_n", "16", "--merge_backend", "fsdp", "--ckpt_root", str(tmp_path / "checkpoints"),
    ]


def test_random_uses_only_current_chunk(experiment, monkeypatch):
    seen = []
    selections = []

    def dispatch(cmd, check):
        stage = Path(cmd[1]).name
        seen.append(stage)
        if stage == "select_data.py":
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", cmd[1:])
                select_data.main()
            pool = read_ids(Path(cmd[cmd.index("--candidate_dir") + 1]))
            out = Path(cmd[cmd.index("--output_dir") + 1])
            selected = read_ids(out)
            assert len(selected) == len(set(selected)) == 3
            assert set(selected) <= set(pool)
            assert datasets.Dataset.from_parquet(str(out / "train.parquet"))["extra_info"][0]["index"] in selected
            selections.append(selected)
        elif stage == "launch_verl_training.py":
            root = Path(cmd[cmd.index("--ckpts_root") + 1]) / cmd[cmd.index("--exp_name") + 1]
            fake_checkpoint(root, int(cmd[cmd.index("--total_training_steps") + 1]))
            assert cmd[cmd.index("--rollout_n") + 1] == "16"
            assert cmd[cmd.index("--ckpts_root") + 1] == experiment[experiment.index("--ckpt_root") + 1]
        else:
            pytest.fail(f"Random selection attempted an unexpected stage: {stage}")

    monkeypatch.setattr(dynamic_selection.subprocess, "run", dispatch)
    monkeypatch.setattr(sys, "argv", experiment)
    dynamic_selection.main()
    dynamic_selection.main()  # Complete run is a no-op.
    experiment[experiment.index("--prefix") + 1] = "independent-replay"
    dynamic_selection.main()
    assert seen == ["select_data.py", "launch_verl_training.py"] * 2
    assert selections[0] == selections[1]


@pytest.mark.parametrize("answer, expected, correct", [
    (r"work then \boxed{0.5}", r"\frac{1}{2}", True),
    (r"\boxed{0.5} then \boxed{2}", r"\frac{1}{2}", False),
    ("0.5", "0.5", False),
])
def test_math_grading_uses_complete_response(answer, expected, correct):
    import runpy
    selection_utils = runpy.run_path(str(Path(__file__).resolve().parents[1] / "select/utils.py"))
    from verl.utils.reward_score.math_reward import compute_score

    assert selection_utils["math_response_passed"](answer, expected) is correct
    assert bool(compute_score(answer, expected)) is correct


def fake_checkpoint(root, step):
    directory = root / f"global_step_{step}"
    actor = directory / "actor"
    actor.mkdir(parents=True, exist_ok=True)
    (actor / "fsdp_config.json").write_text('{"world_size": 2}')
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"shard")
    (directory / "data.pt").write_bytes(b"sampler")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    return directory


def test_checkpoint_resume_with_partial_and_completed_steps(tmp_path):
    from checkpoints import latest_complete, mark_complete, resume_arguments

    (tmp_path / "global_step_10").mkdir()
    assert latest_complete(tmp_path, 10, "fsdp") == 0
    checkpoint = fake_checkpoint(tmp_path, 6)
    with resume_arguments(tmp_path, 6, boundary=False) as args:
        assert Path(args[-1]) == checkpoint
    assert latest_complete(tmp_path, 10, "fsdp") == 6
    boundary = fake_checkpoint(tmp_path, 10)
    (boundary / "actor/optim_world_size_2_rank_1.pt").unlink()
    assert latest_complete(tmp_path, 10, "fsdp") == 6
    with pytest.raises(RuntimeError, match="Incomplete"):
        mark_complete(tmp_path, 10, "fsdp")
    fake_checkpoint(tmp_path, 10)
    mark_complete(tmp_path, 10, "fsdp")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("")
    assert latest_complete(tmp_path, 10, "fsdp") == 10
    with resume_arguments(tmp_path, 10, boundary=True) as args:
        view = Path(args[-1])
        assert (view / "actor").resolve() == boundary / "actor"
        assert not (view / "data.pt").exists()
    assert (boundary / "data.pt").read_bytes() == b"sampler"


def test_interrupted_round_restores_its_sampler_then_switches_chunks(experiment, monkeypatch):
    for flag in ("--iters_per_select", "--num_selections"):
        experiment[experiment.index(flag) + 1] = "2"
    selected_chunks = []
    attempts = []

    def dispatch(cmd, check):
        def option(flag):
            return cmd[cmd.index(flag) + 1]

        if Path(cmd[1]).name == "select_data.py":
            selected_chunks.append(Path(option("--candidate_dir")).name)
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", cmd[1:])
                select_data.main()
            return
        assert Path(cmd[1]).name == "launch_verl_training.py"
        target = int(option("--total_training_steps"))
        root = Path(option("--ckpts_root")) / option("--exp_name")
        attempts.append(target)
        if len(attempts) == 1:
            assert option("--resume_mode") == "disable"
            fake_checkpoint(root, 1)
            raise InterruptedError
        resume = Path(option("--resume_from_path"))
        if target == 2:
            assert resume == root / "global_step_1"
            assert (resume / "data.pt").is_file()
        else:
            assert target == 4
            assert (resume / "actor").resolve() == root / "global_step_2/actor"
            assert not (resume / "data.pt").exists()
            assert (root / "global_step_2/data.pt").is_file()
        fake_checkpoint(root, target)

    monkeypatch.setattr(dynamic_selection.subprocess, "run", dispatch)
    monkeypatch.setattr(sys, "argv", experiment)
    with pytest.raises(InterruptedError):
        dynamic_selection.main()
    dynamic_selection.main()
    dynamic_selection.main()
    assert attempts == [2, 2, 4]
    assert selected_chunks == ["chunk_0", "chunk_1"]


def test_megatron_checkpoint_checks_indexed_tensor_storage(tmp_path):
    import torch
    import torch.distributed.checkpoint as dcp
    from checkpoints import payload_ready

    dist = tmp_path / "actor/dist_ckpt"
    dcp.save({"model": torch.ones(2), "optimizer": torch.zeros(2)}, checkpoint_id=dist)
    (dist / "common.pt").write_bytes(b"common")
    (dist / "metadata.json").write_text('{"sharded_backend": "torch_dist"}')
    assert payload_ready(tmp_path, "megatron")
    next(dist.glob("*.distcp")).write_bytes(b"truncated")
    assert not payload_ready(tmp_path, "megatron")

    for name, tensors, expected in [
        ("weights_only", {"decoder.final_layernorm.weight": torch.ones(2)}, False),
        ("optimizer_only", {"optimizer.state.exp_avg.decoder.weight": torch.ones(2)}, False),
        ("sharded", {"decoder.final_layernorm.weight": torch.ones(2),
                     "optimizer.state.exp_avg.decoder.weight": torch.zeros(2)}, True),
    ]:
        checkpoint = tmp_path / name
        dist = checkpoint / "actor/dist_ckpt"
        dcp.save(tensors, checkpoint_id=dist)
        (dist / "common.pt").write_bytes(b"common")
        (dist / "metadata.json").write_text('{"sharded_backend": "torch_dist"}')
        assert payload_ready(checkpoint, "megatron") is expected


def test_heldout_content_and_run_identity(experiment, tmp_path, monkeypatch):
    from experiment import describe_datasets, record_experiment, run_lock
    from argparse import Namespace

    heldout = ("heldout", tmp_path / "heldout/train.parquet")
    alias_path = tmp_path / "alias.jsonl"
    alias_path.write_text(json.dumps({"prompt": [{"role": "user", "content": "heldout  question  0"}]}))
    with pytest.raises(ValueError, match="overlaps heldout_eval"):
        describe_datasets(("train", tmp_path / "train/train.jsonl"), ("alias", alias_path), heldout)
    roles = describe_datasets(("train", tmp_path / "train/train.jsonl"), None, heldout)
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with run_lock(tmp_path):
                pass
        args = Namespace(seed=42, num_selections=1, adopt_legacy_run=False)
        record_experiment(tmp_path, args, roles)
        args.num_selections = 2
        record_experiment(tmp_path, args, roles)
        args.seed = 43
        with pytest.raises(ValueError, match="settings or data changed"):
            record_experiment(tmp_path, args, roles)


def test_online_similarity_reuses_per_question_scores_on_retry(experiment, monkeypatch):
    import aggregate

    experiment[experiment.index("--mode") + 1] = "sim"
    experiment.extend(["--n_samples_train", "2", "--n_samples_val", "3"])
    seen = []
    attempts = 0

    def dispatch(cmd, check):
        nonlocal attempts
        stage = Path(cmd[1]).name
        seen.append(stage)

        def option(flag):
            return cmd[cmd.index(flag) + 1]

        if stage == "run_inference_local.py":
            prompts = Path(option("--prompts_file"))
            ids = read_ids(prompts.parent)
            out = Path(option("--output_dir")) / "responses_sorted.json"
            rows = [{"group_id": i, "sample_id": j, "is_correct": j == 0}
                    for i in ids for j in range(int(option("--n_samples")))]
            out.write_text(json.dumps(rows))
        elif stage == "run_parallel_analysis.py":
            ids = read_ids(Path(option("--problem_set_path")).parent)
            part = Path(option("--parts_root")) / "part_0"
            (part / "similarity_results_cosine_real.jsonl").write_text("".join(
                json.dumps({"group_id": i, "similarity": float(i)}) + "\n" for i in ids))
        elif stage in ("select_data.py", "aggregate.py"):
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", cmd[1:] + (["--num_workers", "1"] if stage == "aggregate.py" else []))
                (aggregate.main if stage == "aggregate.py" else select_data.main)()
        elif stage == "launch_verl_training.py":
            assert option("--heldout_eval") == "heldout"
            assert Path(option("--val_dir")).name == "heldout"
            root = Path(option("--ckpts_root")) / option("--exp_name")
            pool = read_ids(root / "chunks/chunk_0")
            assert set(read_ids(Path(option("--train_dir")))) == set(sorted(pool)[-3:])
            attempts += 1
            if attempts == 1:
                raise InterruptedError("simulated interruption before the update")
            fake_checkpoint(root, 1)
        else:
            pytest.fail(stage)

    monkeypatch.setattr(dynamic_selection.subprocess, "run", dispatch)
    monkeypatch.setattr(sys, "argv", experiment)
    with pytest.raises(InterruptedError):
        dynamic_selection.main()
    dynamic_selection.main()
    assert seen.count("run_inference_local.py") == 2
    assert seen.count("run_parallel_analysis.py") == 1


@pytest.mark.parametrize("rollouts,world", [(4, 8), (16, 8), (3, 2)])
def test_shards_run_identical_collectives_without_reusing_real_answers(rollouts, world):
    import runpy
    shard_fn = runpy.run_path(str(Path(__file__).resolve().parents[1] / "select/parallel/group_shards.py"))["split_by_problem"]
    rows = [{"group_id": gid, "global_index": gid * rollouts + sample,
             "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "response_mask": [0, 1]}
            for gid in range(3) for sample in range(rollouts)]
    shards = shard_fn(rows, world)
    order = [row["group_id"] for row in shards[0]]
    assert all([row["group_id"] for row in shard] == order for shard in shards)
    actual = [row for shard in shards for row in shard]
    assert sorted(row["global_index"] for row in actual if row["global_index"] >= 0) == list(range(len(rows)))
    assert all(not any(row["response_mask"]) for row in actual if row["global_index"] < 0)
