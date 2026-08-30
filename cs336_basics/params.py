from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class Params:
    # Model architecture (Section 7.2.1)
    vocab_size: int = 10_000
    ctx: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    theta: float = 10000.0

    # Optimization & learning rate schedule
    batch: int = 32
    steps: int = 5000 # 20_000
    amax: float = 5.0e-4
    amin: float = 5.0e-5 # 0.1 * amax
    t_warm: int = 100 # 1000 # should be 1-5% of steps
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    max_l2_norm: float = 1.0

    # Data, checkpoints & logging
    input_text: str = "data/TinyStoriesV2-GPT4-train.txt"
    input_text_encoded: str = "data/TinyStoriesV2-GPT4-train.bin"
    vocab_path: str = "vocab_dict.pkl"
    merges_path: str = "merges.pkl"
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    checkpoint_pth: str = "checkpoint.pt"
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log_every: int = 10
    checkpoint_every: int = 1000
    validation_every: int = 100
    train_val_split: float = 0.9
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
