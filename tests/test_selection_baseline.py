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
        "--mode", "rand", "--seed", "42", "--rollout_n", "16", "--ckpt_root", str(tmp_path / "checkpoints"),
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
            assert cmd[cmd.index("--rollout_n") + 1] == "16"
            assert cmd[cmd.index("--ckpts_root") + 1] == experiment[experiment.index("--ckpt_root") + 1]
        else:
            pytest.fail(f"Random selection attempted an unexpected stage: {stage}")

    monkeypatch.setattr(dynamic_selection.subprocess, "run", dispatch)
    monkeypatch.setattr(sys, "argv", experiment)
    dynamic_selection.main()
    dynamic_selection.main()
    assert seen == ["select_data.py", "launch_verl_training.py"] * 2
    assert selections[0] == selections[1]
