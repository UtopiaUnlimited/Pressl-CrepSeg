from __future__ import annotations

import torch


def pastis_collate_fn(batch: list[dict]) -> dict:
    samples = []
    for item in batch:
        samples.append({key: value for key, value in item.items() if key != "target"})

    return {
        "samples": samples,
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "patch_id": [item["patch_id"] for item in batch],
        "fold": torch.tensor([item["fold"] for item in batch], dtype=torch.long),
    }
