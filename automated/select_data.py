#!/usr/bin/env python3
"""Selection strategies and their shared command-line entry point."""
import argparse
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import datasets as hfds

from config import get_dataset_dir, get_response_dir


@dataclass
class SelectionInputs:
    ids: list[int]
    accuracy: dict[int, float]
    similarity: dict[int, float]
    n: int
    rng: random.Random
    acc_low: float
    acc_high: float
    limit: int | None
    parts_root: Path | None


@dataclass
class RoutedSelection:
    """A selector can annotate every eligible prompt for a route-aware consumer."""
    route_tags: dict[int, str]
    align_scores: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Selector:
    choose: Callable[[SelectionInputs], list[int] | RoutedSelection]
    use_accuracy: bool = False
    gradient: str | None = None
    reference_rollouts: bool = False
    raw_rollouts: bool = False

    @property
    def candidate_rollouts(self):
        return self.raw_rollouts or self.use_accuracy or self.gradient is not None


def by_similarity(data):
    return sorted(data.similarity, key=data.similarity.get, reverse=True)[:data.n]


def by_negative_similarity(data):
    return sorted(data.similarity, key=data.similarity.get)[:data.n]


def by_accuracy_band(data):
    pool = [gid for gid, acc in data.accuracy.items()
            if data.acc_low <= acc <= data.acc_high and (data.limit is None or gid < data.limit)]
    return data.rng.sample(pool, min(data.n, len(pool)))


def by_accgreedy(data):
    return sorted(data.accuracy, key=lambda gid: abs(data.accuracy[gid] - 0.5))[:data.n]


def by_similarity_and_accuracy(data):
    scores = {gid: sim * 4 * data.accuracy[gid] * (1 - data.accuracy[gid])
              for gid, sim in data.similarity.items() if gid in data.accuracy}
    return sorted(scores, key=scores.get, reverse=True)[:data.n]


def random_in_chunk(data):
    pool = data.ids[:data.limit]
    return data.rng.sample(pool, min(data.n, len(pool)))


SELECTOR_REGISTRY = {
    "sim": Selector(by_similarity, gradient="sim", reference_rollouts=True),
    "dot": Selector(by_similarity, gradient="dot", reference_rollouts=True),
    "norm": Selector(by_similarity, gradient="norm"),
    "align": Selector(by_similarity, gradient="sim"),
    "negsim": Selector(by_negative_similarity, gradient="sim", reference_rollouts=True),
    "simacc": Selector(by_similarity_and_accuracy, use_accuracy=True, gradient="sim", reference_rollouts=True),
    "acc": Selector(by_accuracy_band, use_accuracy=True),
    "accgreedy": Selector(by_accgreedy, use_accuracy=True),
    "rand_in_chunk": Selector(random_in_chunk),
}
SELECTOR_REGISTRY["rand"] = SELECTOR_REGISTRY["rand_in_chunk"]


def _scores(path, key, pool):
    scores = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gid = int(row["group_id"])
            if gid in pool:
                value = float(row[key])
                if not math.isfinite(value) or gid in scores:
                    raise ValueError(f"Invalid or duplicate {key} for group {gid} in {path}")
                scores[gid] = value
    return scores


def select_from_chunk(mode, dataset_dir, parts_root, n, output_dir, *,
                      similarity_filename="similarity_results_aggregated.jsonl",
                      acc_low=0.2, acc_high=0.8, limit=None, rng=None,
                      first_problem=1, last_problem=None):
    strategy = SELECTOR_REGISTRY[mode]
    source = Path(dataset_dir) / "train.jsonl"
    with source.open(encoding="utf-8") as f:
        ids = [row["extra_info"]["index"] for row in map(json.loads, f)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate candidate IDs in {source}")
    if first_problem < 1 or (last_problem is not None and last_problem < first_problem):
        raise ValueError("Invalid candidate problem range")
    ids = ids[first_problem - 1:last_problem]
    pool = set(ids)
    if not pool:
        raise ValueError(f"Empty candidate window in {source}")
    if n <= 0:
        raise ValueError("Selection size must be positive")
    parts = Path(parts_root) if parts_root is not None else None
    if strategy.candidate_rollouts and parts is None:
        raise ValueError(f"{mode} needs candidate rollout artifacts")
    data = SelectionInputs(
        ids, _scores(parts / "accuracy_by_problem.jsonl", "accuracy", pool) if strategy.use_accuracy else {},
        _scores(parts / similarity_filename, "similarity", pool) if strategy.gradient else {},
        n, rng if rng is not None else random.Random(), acc_low, acc_high, limit, parts,
    )
    result = strategy.choose(data)
    if isinstance(result, RoutedSelection):
        if set(result.route_tags) != pool or not set(result.align_scores) <= pool:
            raise ValueError("Routing must cover exactly the current candidate pool")
        if any(tag not in ("retain", "both", "forget") for tag in result.route_tags.values()):
            raise ValueError("Invalid route_tag")
        if any(not math.isfinite(score) for score in result.align_scores.values()):
            raise ValueError("Invalid align_score")
        selected = pool
    else:
        selected = set(result)
        if not selected or len(selected) != len(result) or not selected <= pool or len(selected) > n:
            raise ValueError(f"{mode} returned invalid candidate IDs from {source}")
        if len(selected) < n:
            print(f"Selected all {len(selected)} eligible problems (requested {n})")

    with source.open(encoding="utf-8") as f:
        rows = [row for row in map(json.loads, f) if row["extra_info"]["index"] in selected]
    if isinstance(result, RoutedSelection):
        for row in rows:
            gid = row["extra_info"]["index"]
            row["route_tag"] = result.route_tags[gid]
            row["align_score"] = float(result.align_scores[gid]) if gid in result.align_scores else None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset = hfds.Dataset.from_list(rows)
    if isinstance(result, RoutedSelection):
        dataset = dataset.cast_column("align_score", hfds.Value("float64"))
    dataset.to_parquet(str(out / "train.parquet"))
    return selected


# Keep existing standalone Python callers on the same selection implementation.
def select_random(dataset_dir, n, output_dir, max_num=None, rng=random):
    return select_from_chunk("rand", dataset_dir, None, n, output_dir, limit=max_num, rng=rng)


def select_by_similarity(dataset_dir, parts_root, top_n, output_dir, filename, neg=False):
    return select_from_chunk("negsim" if neg else "sim", dataset_dir, parts_root, top_n, output_dir,
                             similarity_filename=filename)


def select_by_accuracy(dataset_dir, parts_root, n, acc_low, acc_high, output_dir, lim=None, rng=random):
    return select_from_chunk("acc", dataset_dir, parts_root, n, output_dir, acc_low=acc_low, acc_high=acc_high,
                             limit=lim, rng=rng)


def select_by_simacc(dataset_dir, parts_root, n, output_dir, filename):
    return select_from_chunk("simacc", dataset_dir, parts_root, n, output_dir, similarity_filename=filename)


def select_by_accgreedy(dataset_dir, parts_root, n, output_dir):
    return select_from_chunk("accgreedy", dataset_dir, parts_root, n, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Select training problems from a candidate chunk")
    parser.add_argument("--mode", required=True, choices=SELECTOR_REGISTRY)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate_dir", help="Directory containing the current train.jsonl")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--acc_low", type=float, default=0.2)
    parser.add_argument("--acc_high", type=float, default=0.8)
    parser.add_argument("--lim", type=int)
    parser.add_argument("--min_problems", type=int, default=1)
    parser.add_argument("--max_problems", type=int)
    parser.add_argument("--seed", type=int, help="Round-specific selection seed")
    parser.add_argument("--similarity_filename", default="similarity_results_aggregated.jsonl")
    parser.add_argument("--output_dir")
    parser.add_argument("--parts_root")
    args = parser.parse_args()

    if args.mode in ("rand", "rand_in_chunk") and not args.candidate_dir:
        parser.error("Random selection requires an explicit --candidate_dir")
    directory = args.candidate_dir or get_dataset_dir(args.dataset, args.model)
    parts = args.parts_root
    if parts is None and SELECTOR_REGISTRY[args.mode].candidate_rollouts:
        parts = get_response_dir(args.dataset, args.model) + "_split"
    name = "random" if args.mode in ("rand", "rand_in_chunk") else args.mode
    if name == "acc":
        name += f"_{args.acc_low}_{args.acc_high}"
    out = args.output_dir or Path(directory) / "selected" / args.model / f"{name}_{args.n}"
    select_from_chunk(args.mode, directory, parts, args.n, out, similarity_filename=args.similarity_filename,
                      acc_low=args.acc_low, acc_high=args.acc_high, limit=args.lim, rng=random.Random(args.seed),
                      first_problem=args.min_problems, last_problem=args.max_problems)
    print(f"Selection complete. Output: {out}")


if __name__ == "__main__":
    main()
