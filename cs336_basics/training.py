import torch
import wandb
import numpy as np
import time
import regex as re
import os
import multiprocessing as mp
import itertools
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm
from cs336_basics.modules import (
    load,
    cross_entropy,
    save_checkpoint,
    gradient_clipping,
    get_lr_cosine_schedule,
    softmax,
    Tokenizer,
    LM,
    AdamW,
)
from dataclasses import asdict
import tyro
from cs336_basics.params import Params

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_path, p: Params, tokenizer: Tokenizer | None = None):
    # init
    memmap = np.memmap(data_path, dtype=np.uint16, mode='r')
    max_data = 40_000_000
    # max_data = len(memmap)
    split_idx = int(p.train_val_split * max_data)
    train_data = memmap[:split_idx]
    val_data = memmap[split_idx:max_data]

    # mini
    # max_data = p.batch * p.ctx # One batch
    # train_data = memmap[:max_data]
    # val_data = memmap[:max_data]
    # print(f"{len(memmap)} tokens available. Using {max_data} tokens")

    wandb.init(
        project="cs336-assignment1",
        config=asdict(p),
        mode=p.wandb_mode,
        group="lr_sweep"
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

        lr = get_lr_cosine_schedule(step, p.amax, p.amin, p.t_warm, p.steps)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()

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
            "train/lr" : lr, 
            "train/tps" : tps,
            "tokens_seen" : tokens_seen,
            "epoch" : epoch,
            "grad_norm" : unclipped_norm,
        } 
        if step % p.checkpoint_every == 0 and step != 0:
            save_checkpoint(model, optimizer, step, build_path(p.checkpoint_pth, step))
        if step % p.validation_every == 0 and step != 0:
            with torch.no_grad():
                x_val, targets_val = load(val_data, p.batch, p.ctx, p.device)
                logits = model(x_val)
                val_loss = cross_entropy(logits, targets_val)

                # Generate some text for qualitative analysis
                if tokenizer is not None:
                    prompt = "Once upon a time"
                    emb = torch.tensor(tokenizer.encode(prompt), device=p.device, dtype=torch.long)
                    out_toks = generate(model, emb, max_gen=100)
                    gen = tokenizer.decode(out_toks.cpu().tolist())

                    # Create wandb table and log
                    table = wandb.Table(columns=["step", "prompt", "generated_text"])
                    table.add_data(step, prompt, gen)
                    w_val["val/samples"] = table

            print("Validation: ", end=' ')
            print_log(step, float(val_loss), optimizer.param_groups[0]['lr'], tps)
            w_val |= {"val/loss" : val_loss, "val/perplexity" : torch.exp(val_loss).item()}

        elif step % p.log_every == 0:
            print_log(step, float(loss), optimizer.param_groups[0]['lr'], tps)

        wandb.log(w | w_val, step=step)

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
    top_p: float | None = None,
    end_token: int | None = None,
    device: str = "mps",
) -> torch.Tensor:
    # TODO: make sure we don't go above max_context length of the model
    # max_gen = min(max_gen, model.get_context_length())
    with torch.no_grad():
        tokens = prompt_tokens.tolist()
        for _ in range(max_gen):
            tokens_tensor = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0) # (batch, seq)
            logits = model(tokens_tensor) # (batch, seq, vocab_size)
            logits = logits[0, -1, :] # discard batch and other words in the seq
            dist = softmax(logits, -1, temperature) # (vocab_size)
            dist_idxs = torch.arange(0, len(dist), device=device)
            if top_p is not None:
                dist_idxs = torch.argsort(dist, descending=True)
                cumsum = torch.cumsum(dist[dist_idxs], -1)
                cutoff_idx = torch.searchsorted(cumsum, torch.tensor(top_p, device=cumsum.device)).item()
                dist_idxs = dist_idxs[:cutoff_idx + 1]
                dist = dist[dist_idxs]

            selected = torch.multinomial(dist, num_samples=1)           
            token = dist_idxs[selected]
            if token.item() == end_token:
                break
            tokens.append(token.item())

    return torch.tensor(tokens, device=device, dtype=torch.long)

def _process_chunk(input_path, start, end, special_tokens, pre_tok_pat, special_token_pat):
    pre_tok: dict[tuple[bytes, ...], int] = defaultdict(int) 

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    docs = re.split(special_token_pat, chunk)

    for doc in docs:
        # don't split up special tokens
        if doc in special_tokens:
            pre_tok[tuple((doc.encode(),))] += 1
            continue
        matches = re.finditer(pre_tok_pat, doc)
        for match in matches:
            tok = doc[match.start():match.end()]
            tok_enc = tok.encode('utf-8')
            pre_tok[tuple([bytes([i]) for i in tok_enc])] += 1

    return pre_tok

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pre_tok_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    special_tokens_esc = [re.escape(tok) for tok in special_tokens]
    special_token_pat = f"({'|'.join(special_tokens_esc)})"

    n_proc = mp.cpu_count()
    print(f"Running with {n_proc} processes")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_proc, special_tokens[0].encode("utf-8"))

    file_size = os.path.getsize(input_path)
    min_size = 5 * 1024 * 1024 # 5MB

    # Use serial for small files
    if file_size < min_size:
        pre_tok = _process_chunk(input_path, 0, file_size, special_tokens, pre_tok_pat, special_token_pat)
    else:
        chunk_args = [
            (input_path, start, end, special_tokens, pre_tok_pat, special_token_pat)
            for start, end in zip(boundaries, boundaries[1:])
        ]
        # 1. pre-tokenization
        with mp.Pool(n_proc - 2) as p:
            res = p.starmap(_process_chunk, chunk_args)
        
        # merge pre-tok dictionaries
        pre_tok: dict[tuple[bytes, ...], int] = defaultdict(int) 
        for partial_dict in res:
            for key, count in partial_dict.items():
                pre_tok[key] += count

    # 2. merges
    bp_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words = defaultdict(set)
    for tok, count in pre_tok.items():
        for pair in zip(tok, tok[1:]):
            bp_counts[pair] += count
            pair_to_words[pair].add(tok)

    merges = []
    target_merges = vocab_size - 256 - len(special_tokens)
    for i in tqdm(range(target_merges), desc="Training BPE Merges", unit="merge"):
        # Max based on value, or key if a tie
        max_bp = max(bp_counts.items(), key=lambda x: (x[1], x[0]))[0]

        merges.append(tuple(max_bp))

        # Create a snapshot of the set so we don't modify it while iterating
        words_to_update = list(pair_to_words[max_bp]) 

        for old_word in words_to_update:
            count = pre_tok[old_word]
            
            # 1. Tear down the old pairs
            for pair in zip(old_word, old_word[1:]):
                bp_counts[pair] -= count
                pair_to_words[pair].discard(old_word) # discard is safe if element doesn't exist
                
                # Optional but recommended: keep dictionaries lean
                if bp_counts[pair] <= 0:
                    del bp_counts[pair]

            # 2. Build the new word
            new_word = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and (old_word[i], old_word[i+1]) == max_bp:
                    merged_tok = old_word[i] + old_word[i+1]
                    new_word.append(merged_tok)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            
            new_word = tuple(new_word)

            # 3. Set up the new pairs
            for pair in zip(new_word, new_word[1:]):
                bp_counts[pair] += count
                pair_to_words[pair].add(new_word)

            # 4. Update the main vocab tracker
            pre_tok[new_word] = count
            del pre_tok[old_word]


    # 3. assemble vocab
    vocab = []
    vocab.extend([tok.encode('utf-8') for tok in special_tokens])
    vocab.extend([bytes([i]) for i in range(256)])
    for merge in merges:
        merge_joined = b''.join(merge)
        vocab.append(merge_joined)

    vocab_dict = {i : v for i, v in enumerate(vocab)}
    return vocab_dict, merges

def _tokenize_chunk(input_path, start, end, tokenizer: Tokenizer):
    """
    Read in a short chunk and tokenize it!
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return tokenizer.encode(chunk)

def tokenize_file(input_text: str, out: str, vocab_dict, merges, special_tokens):
    """
    input_text: .txt path where the input text is stored
    out: path to save the encoded input text file
    vocab_dict: tokenization scheme from {int : bytes}
    merges: list[tuple[bytes, bytes]] to merge
    special_tokens: e.g. <|endoftext|>
    """
    tokenizer = Tokenizer(vocab_dict, merges, special_tokens)

    n_proc = mp.cpu_count() - 2
    with open(input_text, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_proc, special_tokens[0].encode("utf-8"))

    print(f"Found {len(boundaries)} boundaries")

    chunk_args = [
        (input_text, start, end, Tokenizer(vocab_dict, merges, special_tokens))
        for start, end in zip(boundaries, boundaries[1:])
    ]
    t0 = time.time()
    with mp.Pool(n_proc - 2) as p:
        res = p.starmap(_tokenize_chunk, chunk_args)
    tokens = list(itertools.chain.from_iterable(res))
    print(f"Tokenized {os.path.getsize(input_text):0.2f} MB in {time.time() - t0:.2f}s")

    print(f"Saving encoded text to {out}")
    tokens_np = np.array(tokens, dtype=np.uint16)
    tokens_np.tofile(out)

if __name__ == "__main__":
    # Parse args into the Params dataclass
    args = tyro.cli(Params)

    if not os.path.exists(args.vocab_path) or not os.path.exists(args.merges_path):
        print(f"{args.vocab_path} or {args.merges_path} not found.")
        print("Training BPE")
        vocab_dict, merges = train_bpe(args.input_text, args.vocab_size, args.special_tokens)
        print(f"Saving to {args.vocab_path}\nand {args.merges_path}")
        torch.save(vocab_dict, args.vocab_path)
        torch.save(merges, args.merges_path)
    else:
        print("Loading vocab + merges from cache")
        vocab_dict = torch.load(args.vocab_path)
        merges = torch.load(args.merges_path)

    # Create tokenizer
    tokenizer = Tokenizer(vocab_dict, merges, args.special_tokens)

    # Tokenize the training data
    if not os.path.exists(args.input_text_encoded):
        print(f"{args.input_text_encoded} was not found")
        print(f"Encoding entire text file...")
        tokenize_file(args.input_text, args.input_text_encoded, vocab_dict, merges, args.special_tokens)

    # LR sweep
    max_lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    for max_lr in max_lrs:
        # Create Model
        model = LM(
            vocab_size=args.vocab_size,
            max_seq_len=args.ctx,
            n_layers=args.num_layers,
            d_model=args.d_model,
            n_heads=args.num_heads,
            d_ff=args.d_ff,
            theta=args.theta,
            device=args.device,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=0.0, # init
            weight_decay=args.weight_decay, # what does this do?
            betas=(args.beta1, args.beta2),
            eps=args.eps
        )

        # train(model, optimizer, args.input_text_encoded, args, tokenizer)

        max_lr = max_lr
        amin = 0.1 * max_lr
        args.amin = amin
        args.amax = max_lr
        train(model, optimizer, args.input_text_encoded, args, tokenizer)