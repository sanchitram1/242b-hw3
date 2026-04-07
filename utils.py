import json
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=shuffle,
    )


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
