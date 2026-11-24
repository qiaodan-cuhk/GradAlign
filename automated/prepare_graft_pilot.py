"""Materialize the shared 80/16/40 math pilot once, preserving upstream identities."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import datasets
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from config import BASE_DATA_DIR
from experiment import _prompt_key
from prepare_data import _build_user_prompt, _process_to_verl, _write_outputs

WEB = ("TIGER-Lab/WebInstruct-verified-unfiltered", "ac48a536b31d3cabb346e1139fb6c69ae631abd7",
       "14103d8dbf664fbac9e72e6e3d4894b4220139207ca1203bf14c9b825aa52ee6")
AMC = ("AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316",
       "4de056fd006062cee46e3900727dd5d00094b8592ed0bfd6b5dfae2267f72dd7")


def load_source(source, local):
    path = local or hf_hub_download(source[0], "data/train-00000-of-00001.parquet",
                                    repo_type="dataset", revision=source[1])
    if local:
        digest = hashlib.sha256()
        with Path(local).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != source[2]:
            raise ValueError(f"Offline parquet does not match pinned {source[0]} revision {source[1]}")
    return datasets.Dataset.from_parquet(str(path))


def build_splits(web, amc, tokenizer, seed):
    rng = np.random.default_rng(seed)
    used = set()

    def accept(problem, label):
        key = _prompt_key(problem)
        if key in used:
            return False
        messages = [{"role": "user", "content": _build_user_prompt(problem, label)}]
        if len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)) > 1024:
            return False
        used.add(key)
        return True

    splits = {}
    for label, year, size in (("amc23", "2023_AMC", 40), ("amc22_ref16", "2022_AMC", 16)):
        source = [row for row in amc if year in row["url"]]
        rows = []
        for i in rng.permutation(len(source)):
            row = source[int(i)]
            if accept(row["problem"], label):
                rows.append({"id": str(row["id"]), "problem": row["problem"], "answer": str(int(row["answer"]))})
            if len(rows) == size:
                break
        if len(rows) != size:
            raise ValueError(f"Expected {size} distinct eligible {label} prompts, found {len(rows)}")
        splits[label] = rows

    rows = []
    for i in rng.permutation(len(web)):
        row = web[int(i)]
        if row["subject"] != "Mathematics" or row["answer_type"] not in ("Integer", "Float"):
            continue
        answer = str(row["short_answer"]).strip() if row["short_answer"] is not None else ""
        if answer and accept(row["original_question"], "webinstruct_math80"):
            rows.append({"id": str(row["task_id"]), "problem": row["original_question"], "answer": answer})
        if len(rows) == 80:
            break
    if len(rows) != 80:
        raise ValueError(f"Expected 80 eligible math prompts, found {len(rows)}")
    splits["webinstruct_math80"] = rows
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=BASE_DATA_DIR)
    parser.add_argument("--model_path", required=True, help="Tokenizer from the frozen base snapshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--web_parquet", help="Offline copy of the pinned WebInstruct parquet")
    parser.add_argument("--amc_parquet", help="Offline copy of the pinned AMC parquet")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    splits = build_splits(load_source(WEB, args.web_parquet), load_source(AMC, args.amc_parquet),
                          AutoTokenizer.from_pretrained(args.model_path), args.seed)
    for label, rows in splits.items():
        destination = root / label
        file = "pilot_source.json"
        source = WEB if label == "webinstruct_math80" else AMC
        manifest = {"dataset": source[0], "revision": source[1], "seed": args.seed, "ids": [row["id"] for row in rows]}
        if destination.exists():
            if ((destination / file).is_file() and json.loads((destination / file).read_text()) == manifest
                    and all((destination / part).is_file() for part in ("train.parquet", "train.jsonl"))):
                continue
            raise FileExistsError(f"Use a fresh --output_dir for a different pilot dataset: {destination}")
        with TemporaryDirectory(dir=root) as tmp:
            stage = Path(tmp) / label
            _write_outputs(_process_to_verl(datasets.Dataset.from_list(rows), label), str(stage))
            (stage / file).write_text(json.dumps(manifest, indent=2) + "\n")
            stage.rename(destination)
        print(f"{label}: {len(rows)} rows in {destination}")


if __name__ == "__main__":
    main()
