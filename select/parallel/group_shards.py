"""Keep FSDP collectives aligned by problem, including groups smaller than the world size."""
from itertools import groupby


def split_by_problem(entries, world_size):
    if world_size < 1:
        raise ValueError("world_size must be positive")
    entries.sort(key=lambda row: (row["group_id"], len(row["input_ids"]), row["global_index"]))
    shards = [[] for _ in range(world_size)]
    for gid, group in groupby(entries, key=lambda row: row["group_id"]):
        rows = list(group)
        for offset in range(0, len(rows), world_size):
            block = rows[offset:offset + world_size]
            # A short, zero-weight forward keeps all ranks in the same collective.
            token = block[0]["input_ids"][0]
            padding = {"input_ids": [token, token], "attention_mask": [1, 1],
                       "response_mask": [0], "group_id": gid, "global_index": -1}
            for rank, shard in enumerate(shards):
                shard.append(block[rank] if rank < len(block) else padding)
    return shards
