import torch
import wandb
import numpy as np
import time
from tqdm import tqdm
from cs336_basics.modules import (
    load,
    cross_entropy,
    save_checkpoint,
    gradient_clipping,
    get_lr_cosine_schedule,
    softmax
)

from dataclasses import dataclass, asdict

@dataclass
class HyperParams:
    steps: int
    lr: float
    batch: int
    ctx: int
    checkpoint_pth: str
    device: str = 'mps'
    log_every: int = 10
    checkpoint_every: int = 100
    validation_every: int = 100
    train_val_split: float = 0.9
    max_l2_norm: float = 1.0
    amax: float = 6e-4
    amin: float = 6e-5
    t_warm: int = 100
    t_c: int = 1000
    wandb_mode: str = 'offline'

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_path, p: HyperParams):
    # init
    memmap = np.memmap(data_path, dtype=np.int64, mode='r')
    split_idx = int(p.train_val_split * len(memmap))
    train_data = memmap[:split_idx]
    val_data = memmap[split_idx:]

    wandb.init(
        project="cs336-assignment1",
        config=asdict(p),
        mode=p.wandb_mode
    )

    for step in tqdm(range(p.steps)):
        # 1. Sample
        x_in, targets = load(train_data, p.batch, p.ctx, p.device)

        if torch.mps.is_available():
            torch.mps.synchronize()
        start = time.perf_counter()

        # 2. Forward & Backward pass
        logits = model(x_in)
        loss = cross_entropy(logits, targets)
        loss.backward()
        unclipped_norm = gradient_clipping(model.parameters(), p.max_l2_norm)

        optimizer.step()
        optimizer.zero_grad()
        curr_lr = optimizer.param_groups[0]['lr']
        lr = get_lr_cosine_schedule(step, p.amax, p.amin, p.t_warm, p.t_c)
        optimizer.param_groups[0]['lr'] = lr

        # Sync + log
        if torch.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        tps = p.batch * p.ctx / elapsed
        tokens_seen = (step + 1) * p.batch * p.ctx
        epoch = tokens_seen / len(train_data)

        w_val = {}
        w = {
            "train/loss" : loss.item(),
            "train/lr" : curr_lr, 
            "train/tps" : tps,
            "tokens_seen" : tokens_seen,
            "epoch" : epoch,
            "grad_norm" : unclipped_norm,
        } 
        if step % p.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, build_path(p.checkpoint_pth, step))
        if step % p.validation_every == 0:
            with torch.no_grad():
                x_val, targets_val = load(val_data, p.batch, p.ctx, p.device)
                logits = model(x_val)
                val_loss = cross_entropy(logits, targets_val)
                # Generate some text
                emb = 
                generate(model, "Once upon a time", )
            print("Validation: ", end=' ')
            print_log(step, float(val_loss), optimizer.param_groups[0]['lr'], tps)
            w_val = {"val/loss" : val_loss, "val/perplexity" : torch.exp(val_loss).item()}
            #TODO: add table with qualitative token generation to wandb table
        elif step % p.log_every == 0:
            print_log(step, float(loss), optimizer.param_groups[0]['lr'], tps)

        wandb.log(w + w_val, step=step)

    wandb.finish()


def build_path(pth: str, step: int) -> str:
    split = pth.split(".")
    beginning = '.'.join(split[:-1]) + str(step)
    ext = split[-1]
    return beginning + "." + ext

def print_log(step, loss, lr, tps):
    """
    step: Current training step (int)
    loss: Current loss value (float)
    lr: Current learning rate (float)
    tps: Tokens/samples per second (float)
    """
    # Formats:
    # - step: right-aligned with commas
    # - loss: 4 decimal places
    # - lr: scientific notation
    # - tps: 1 decimal place with comma separation
    print(f"Step: {step:>7,d} | Loss: {loss:.4f} | LR: {lr:.2e} | Speed: {tps:>8,.1f} tok/s")


def generate(
    model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    max_gen: int,
    temperature: float = 1,
    top_p : int | None = None,
    end_token : str = "<|endoftext|>"
) -> torch.Tensor:
    with torch.no_grad():
        tokens = list(prompt_tokens)
        for _ in range(max_gen):
            logits = model(tokens)
            dist = softmax(logits, -1, temperature)
            dist_idxs = torch.arange(0, len(dist))
            if top_p is not None:
                dist_idxs = torch.argsort(dist)
                dist_idxs = dist_idxs[:top_p]
                dist = dist[dist_idxs]

            selected = torch.multinomial(dist, num_samples=1)           
            token = dist_idxs[selected]
            if token == end_token:
                break
            tokens.append(token)

    return torch.Tensor(tokens)
            