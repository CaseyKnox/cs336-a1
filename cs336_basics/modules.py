import einops
import torch
import torch.nn as nn
import math
from jaxtyping import Bool, Float, Int
from collections.abc import Callable, Iterable
from typing import Optional

class Linear(nn.Module):
    def __init__(self, in_feat, out_feat, device=None, dtype=None):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(out_feat, in_feat, device=device, dtype=dtype))
        self.weights = nn.init.trunc_normal_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weights, "... d_in, ... d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, device=None, dtype=None):
        """
        vocab_size : number or words in the vocabulary
        emb_dim : (aka d_model) vector size of the embedding dimension
        """
        super().__init__()

        self.embedding = nn.init.trunc_normal_(nn.Parameter(torch.empty(vocab_size, emb_dim, device=device, dtype=dtype)))

    def forward(self , token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: maps from word to vocab index. shaped (batch_size, sequence_length)
        """
        assert (token_ids > 0).all()
        assert token_ids.dtype == torch.int64, f"token_ids has type {token_ids.dtype} != int"
        return self.embedding[token_ids] # (batch, sequence_len, emb_dim)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int - hidden dim of the model
        eps: float = 1e-5 - Epsilon value for numerical stability
        device: torch.device | None = None - Device to store the parameters on
        dtype: torch.dtype | None = None - Data type of the params
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g = nn.init.trunc_normal_(nn.Parameter(torch.empty(self.d_model, device=device, dtype=dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the RMS Norm
        x: torch.Tensor - input tensor with shape (batch, seq_len, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32) # upcast for stability

        denom = einops.reduce(torch.pow(x, 2), "b s d -> b s 1", "sum") / self.d_model # (B, S, 1)
        denom += self.eps
        denom = torch.sqrt(denom)

        # Element-wise multiplication
        numerator = einops.einsum(x, self.g, "b s d, d -> b s d")

        out = numerator / denom # (b s d)

        return out.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        """
        super().__init__()
        self.w1 = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype)))
        self.w2 = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype)))
        self.w3 = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype)))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """
        x (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.
        """
        w1x = einops.einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        w3x = einops.einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")

        silu = einops.einsum(w1x, torch.sigmoid(w1x), "... d_ff, ... d_ff -> ... d_ff")

        # Element wise multiplication
        glu = einops.einsum(silu, w3x, "... d_ff, ... d_ff -> ... d_ff")
        return einops.einsum(glu, self.w2, "... d_ff, d_model d_ff -> ... d_model")

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        """
        Rotational positional embeddings

        theta (float): value for the RoPE
        d_k (int): dimension of query and key vectors
        max_seq_len (int): Maximum seq length that will be input
        device (torch.device | None): Device to store the buffer on
        """
        super().__init__()
        # Note using arange(0,d,2) is equivalent to 2k-2 for [1,d//2]
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)) # (d//2,)
        positions = torch.arange(max_seq_len, dtype=torch.float32) # (max_seq_len,)
        thetas = torch.einsum("i, j -> i j", positions, freqs) # (max_seq_len, d//2)
        thetas = thetas.repeat_interleave(2, dim=-1) # (max_seq_len, d)

        self.register_buffer("cos", torch.cos(thetas), persistent=False)
        self.register_buffer("sin", torch.sin(thetas), persistent=False)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2] # evens
        x2 = x[..., 1::2] # odds
        # we want [-x2, x1, ...]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x (torch.Tensor): shape=(..., seq_len, d_k): input tensor
        token_positions (torch.Tensor): shape=(..., seq_len): token positions of x along the sequence dimension

        returns torch.Tensor shape=(..., seq_len, d_k)
        """
        x_rotated = self._rotate_half(x)
        cos = self.cos[token_positions] # (seq_len, d)
        sin = self.sin[token_positions] # (seq_len, d)

        # Handle arbitrary dims in x
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        return x * cos + x_rotated * sin

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        """
        d_model (int): Dimensionality of the transformer block inputs
        n_heads: number of heads to use in mult-headed self attention
        """
        super().__init__()
        self.Q = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        self.K = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        self.V = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        self.O = nn.init.trunc_normal_(nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype)))

        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = self.d_model // n_heads
        self.device = device
        self.dtype = dtype

        # Optional Rope
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.dk, max_seq_len, device)
        else:
            self.rope = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x torch.Tensor: input of shape (B, seq, d_model)
        """
        B, seq, dm = x.shape
        assert dm == self.d_model, f"input tensor does not have appropriate last dim: {dm} != {self.d_model}"

        # m = d_model
        # d = d_model
        wq = torch.einsum("d m, b s m -> b s d", self.Q, x)
        wk = torch.einsum("d m, b s m -> b s d", self.K, x)
        wv = torch.einsum("d m, b s m -> b s d", self.V, x)

        # Reshape to multiple heads
        wq = torch.reshape(wq, (B, seq, self.n_heads, self.dk)).transpose(1, 2) # (b h s dk)
        wk = torch.reshape(wk, (B, seq, self.n_heads, self.dk)).transpose(1, 2) # (b h s dk)
        wv = torch.reshape(wv, (B, seq, self.n_heads, self.dk)).transpose(1, 2) # (b h s dk)

        # RoPE
        if self.rope is not None:
            seq_len = wq.shape[2]
            wq = self.rope(wq, torch.arange(0, seq_len))
            wk = self.rope(wk, torch.arange(0, seq_len))

        # s = sq 
        # t = sk
        # now d = dk
        qk = torch.einsum("b h s d, b h t d -> b h s t", wq, wk) / (self.dk ** 0.5)

        # mask inf before softmax for causal self-attn
        mask = torch.triu(
            torch.full((seq, seq), -torch.inf, device=self.device, dtype=self.dtype), 
            diagonal=1
        )
        qk = qk + mask
        qk = softmax(qk, -1) # softmax along the seq dim

        qkv = torch.einsum("b h s t, b h t d -> b h s d", qk, wv).transpose(1, 2) # (b s h dk)
        qkv = qkv.reshape((B, seq, self.d_model))
        return torch.einsum("d m, b s m -> b s d", self.O, qkv)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    softmax(v) = exp(v_i) / sum(exp(v_j))
    """
    # Subtract a constant for stability
    dim_max = torch.amax(x, dim, keepdim=True)
    x = x - dim_max

    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.rms = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, n_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre norm
        x += self.attn(self.rms(x)) # use rope
        x += self.swiglu(self.rms2(x))
        return x

class LM(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 max_seq_len: int, 
                 n_layers: int, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int,
                 theta: float, 
                 device=None, 
                 dtype=None):
        """
        vocab_size (int): Size of the vocabulary for the emb matrix
        max_seq_len (int): max context length
        n_layers (int): number of transformer blocks to use
        """
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, theta, max_seq_len, device, dtype) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x) # (B, S, D)

        for block in self.blocks:
            x = block(x) # (B S D)

        x = self.norm(x) # B S D
        x = self.linear(x) # B S V
        return x
        # return torch.softmax(x, -1)

def calculate_flops(batch, seq_len, d_model, d_ff, vocab_size, n_blocks):
    block = 8 * batch * seq_len * d_model**2 + 4 * batch * seq_len**2 * d_model + 6 * batch * seq_len * d_ff * d_model

    lm_head = 2 * batch * seq_len * vocab_size * d_model

    return n_blocks * block + lm_head

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: output from model of shape (B, seq, vocab_size)
    targets: expected output of shape (B, seq)

    returns cross entropy of shape (B, 1)
    """
    # Subtract the maximum for stability
    # All exp(x) will fall in (0,1] since x_i <= 0
    c = torch.amax(logits, -1, keepdim=True)
    logits = logits - c

    # log rules cancel out the numerator and becomes subtraction
    # ln(e(x)) = x
    # ln(x/y) = ln(x) - ln(y)
    # equation becomes x - denom
    denom = torch.log(torch.sum(torch.exp(logits), -1, keepdim=True))

    logits_plucked = logits.gather(-1, targets.unsqueeze(-1)) # (B, S)
    return -(logits_plucked - denom).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid alpha: {lr}")
        defaults = {
            "a" : lr,
            "b1" : betas[0],
            "b2" : betas[1],
            "eps" : eps,
            "l" : weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            a = group["a"]
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]
            l = group["l"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Extract values
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data # grad of loss wrt p

                # Algorithm
                a_t = a * (1 - b2**t)**0.5 / (1-b1**t)
                p.data -= a * l * p.data
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2
                p.data -= a_t * m / (v**0.5 + eps)

                # Save
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

def calculate_activations(b,h,s,d,d_f,v,n):
    return n *(8*b*s*d + 2*b*h*s**2 + 4*b*s*d_f) + b*s*d + b*s + b*s*v

def calc_params(d,d_f,v,n):
    return d * (2*v + n*(2+4*d+3*d_f) + 1)

def calc_grads(d,d_f,v,n):
    return calc_params(d,d_f,v,n)

def calc_optim(d,d_f,v,n):
    return 2 * calc_params(d,d_f,v,n)

def calc_total_memory(b,h,s,d,d_f,v,n):
    total = calculate_activations(b,h,s,d,d_f,v,n) + calc_params(d,d_f,v,n) + calc_grads(d,d_f,v,n) + calc_optim(d,d_f,v,n)
    return total

def get_lr_cosine_schedule(
    t: int,
    amax: float,
    amin: float,
    t_warm: int,
    t_c: int,
):
    # warm-up
    if t < t_warm:
        return t / t_warm * amax
    elif t_warm <= t <= t_c:
        # cosine annealing phase
        return amin + 0.5 * (1 + math.cos((t-t_warm) / (t_c - t_warm) * math.pi)) * (amax - amin)
    else:
        # post annealing
        return amin

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    parameters = list(parameters)
    # Calculate the l2 norm of "g" the concat of all p.grad
    with torch.no_grad():
        norm = 0
        for p in parameters:
            if p.grad is None:
                continue
            norm += torch.sum(p.grad**2)
        norm = torch.sqrt(norm)

        # Do nothing
        if norm < max_l2_norm:
            return

        # Clip the grads
        scale = max_l2_norm / (norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad *= scale
