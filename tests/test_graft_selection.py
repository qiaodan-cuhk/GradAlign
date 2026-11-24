import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "automated"))
from select_data import select_from_chunk


def test_sim_route_matches_hard_filter_and_preserves_candidate_order(tmp_path):
    ids = [75, 11, 62, 32, 78, 4, 69, 81, 90, 43]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps({"extra_info": {"index": i}}) + "\n" for i in ids))
    # Equal scores break ties in exactly the order used by sim.
    scores = list(zip(reversed(ids), [0.9, 0.5, 0.5, 0.2, 0, 0, 0, -0.1, -0.4, -0.8], strict=True))
    source = tmp_path / "similarity_results_aggregated.jsonl"
    source.write_text("".join(json.dumps({"group_id": i, "similarity": s}) + "\n" for i, s in scores))
    hard = select_from_chunk("sim", tmp_path, tmp_path, 2, tmp_path / "hard")
    select_from_chunk("sim_route", tmp_path, tmp_path, 10, tmp_path / "route")
    table = pq.read_table(tmp_path / "route/train.parquet")
    assert str(table.schema.field("route_tag").type) == "int64"
    rows = table.to_pylist()
    assert [row["extra_info"]["index"] for row in rows] == ids
    assert {i for i, row in zip(ids, rows, strict=True) if row["route_tag"] == 1} == hard
    assert [row["route_tag"] for row in rows].count(2) == 2
    assert [row["align_score"] for row in rows] == [dict(scores)[i] for i in ids]
    with pytest.raises(ValueError, match="k=1"):
        select_from_chunk("sim_route", tmp_path, tmp_path, 2, tmp_path / "bad")
    source.write_text(source.read_text().split("\n", 1)[1])
    with pytest.raises(ValueError, match="every candidate"):
        select_from_chunk("sim_route", tmp_path, tmp_path, 10, tmp_path / "bad")
