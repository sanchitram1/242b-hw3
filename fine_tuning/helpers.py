import torch.nn as nn
from lora import LoRALinear

from models import TinyGPT


def apply_lora_to_model(
    model: TinyGPT,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_ff: bool = False,
) -> TinyGPT:
    for block in model.blocks:
        block.qkv = LoRALinear(block.qkv, rank=rank, alpha=alpha, dropout=dropout)
        block.out_proj = LoRALinear(
            block.out_proj, rank=rank, alpha=alpha, dropout=dropout
        )

        if target_ff:
            block.ff[0] = LoRALinear(
                block.ff[0], rank=rank, alpha=alpha, dropout=dropout
            )
            block.ff[2] = LoRALinear(
                block.ff[2], rank=rank, alpha=alpha, dropout=dropout
            )
    return model


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def count_trainable_parameters(): ...
