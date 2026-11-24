"""Keep sample routes in DataProto.batch through generation, repeat and balancing."""
import torch

from verl.utils.dataset.rl_dataset import RLHFDataset


class RoutedRLHFDataset(RLHFDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        if "route_tag" in row:  # Held-out prompts do not carry selection metadata.
            tag = row["route_tag"]
            if type(tag) is not int or tag not in (0, 1, 2):
                raise ValueError(f"Invalid integer route_tag at dataset row {index}: {tag!r}")
            row["route_tag"] = torch.tensor(tag, dtype=torch.long)
        return row
