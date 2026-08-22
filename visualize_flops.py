# /// script
# dependencies = [
#     "matplotlib>=3.8.0",
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Interactive FLOPs Visualizer and Breakdown Tool for Transformer Language Models.
Based on CS336 Transformer architecture and FLOP analysis.

Usage:
    uv run visualize_flops.py            # Opens interactive GUI (and prints analysis)
    uv run visualize_flops.py --cli      # Prints analysis and tables to terminal
    uv run visualize_flops.py --save     # Saves static analysis figures to disk
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Dict, Any


def compute_decomposed_flops(
    batch_size: int,
    seq_len: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    num_layers: int,
    ffn_type: str = "swiglu",  # "swiglu" (3 matmuls) or "standard" (2 matmuls)
) -> Dict[str, Any]:
    """
    Computes exact decomposed FLOP counts for a Transformer LM forward pass.

    Matrix Multiplications Breakdown (FLOPs = 2 * M * K * N per GEMM):
    1. Attention QKV Projections:
       - Q: (B, S, d_model) x (d_model, d_model) -> 2 * B * S * d_model^2
       - K: (B, S, d_model) x (d_model, d_model) -> 2 * B * S * d_model^2
       - V: (B, S, d_model) x (d_model, d_model) -> 2 * B * S * d_model^2
       Total QKV per layer = 6 * B * S * d_model^2

    2. Attention Core MatMuls (QK^T and Attn * V):
       - Scores QK^T: (B, H, S, d_k) x (B, H, d_k, S) -> 2 * B * H * S * S * d_k = 2 * B * S^2 * d_model
       - Context Attn * V: (B, H, S, S) x (B, H, S, d_k) -> 2 * B * H * S * S * d_k = 2 * B * S^2 * d_model
       Total Attn MatMuls per layer = 4 * B * S^2 * d_model

    3. Attention Output Projection:
       - O: (B, S, d_model) x (d_model, d_model) -> 2 * B * S * d_model^2

    4. Feed-Forward Network (FFN):
       - SwiGLU: W1 (gate: d_model -> d_ff), W3 (up: d_model -> d_ff), W2 (down: d_ff -> d_model)
         Total SwiGLU per layer = 3 * (2 * B * S * d_model * d_ff) = 6 * B * S * d_model * d_ff
       - Standard MLP: W1 (up: d_model -> d_ff), W2 (down: d_ff -> d_model)
         Total Standard per layer = 2 * (2 * B * S * d_model * d_ff) = 4 * B * S * d_model * d_ff

    5. LM Head / Unembedding:
       - Logits: (B, S, d_model) x (d_model, vocab_size) -> 2 * B * S * d_model * vocab_size
    """
    B, S, d, V, L = batch_size, seq_len, d_model, vocab_size, num_layers

    # Layer components (multiplied by num_layers)
    qkv_flops = L * (6 * B * S * (d**2))
    attn_core_flops = L * (4 * B * (S**2) * d)
    attn_out_flops = L * (2 * B * S * (d**2))
    attn_total_flops = qkv_flops + attn_core_flops + attn_out_flops

    if ffn_type == "swiglu":
        ffn_flops = L * (6 * B * S * d * d_ff)
    else:
        ffn_flops = L * (4 * B * S * d * d_ff)

    transformer_blocks_flops = attn_total_flops + ffn_flops
    lm_head_flops = 2 * B * S * d * V

    total_flops = transformer_blocks_flops + lm_head_flops

    # Parameter count estimation
    attn_params = L * (4 * (d**2))  # Q, K, V, O
    ffn_params = L * (3 * d * d_ff if ffn_type == "swiglu" else 2 * d * d_ff)
    lm_head_params = d * V
    emb_params = V * d
    total_params = attn_params + ffn_params + lm_head_params + emb_params

    return {
        "qkv_proj": qkv_flops,
        "attn_core": attn_core_flops,
        "attn_out": attn_out_flops,
        "attn_total": attn_total_flops,
        "ffn_total": ffn_flops,
        "blocks_total": transformer_blocks_flops,
        "lm_head": lm_head_flops,
        "total_flops": total_flops,
        "total_params": total_params,
        "attn_params": attn_params,
        "ffn_params": ffn_params,
        "lm_head_params": lm_head_params,
        "emb_params": emb_params,
        # Proportions
        "pct_qkv": (qkv_flops / total_flops) * 100,
        "pct_attn_core": (attn_core_flops / total_flops) * 100,
        "pct_attn_out": (attn_out_flops / total_flops) * 100,
        "pct_attn_total": (attn_total_flops / total_flops) * 100,
        "pct_ffn": (ffn_flops / total_flops) * 100,
        "pct_lm_head": (lm_head_flops / total_flops) * 100,
    }


PRESETS = {
    "GPT-2 Small": {
        "d_model": 768,
        "num_layers": 12,
        "d_ff": int(8 * 768 / 3),
        "vocab_size": 50257,
        "seq_len": 1024,
        "batch_size": 1,
    },
    "GPT-2 Medium": {
        "d_model": 1024,
        "num_layers": 24,
        "d_ff": int(8 * 1024 / 3),
        "vocab_size": 50257,
        "seq_len": 1024,
        "batch_size": 1,
    },
    "GPT-2 Large": {
        "d_model": 1280,
        "num_layers": 36,
        "d_ff": int(8 * 1280 / 3),
        "vocab_size": 50257,
        "seq_len": 1024,
        "batch_size": 1,
    },
    "GPT-2 XL (1k)": {
        "d_model": 1600,
        "num_layers": 48,
        "d_ff": 4288,
        "vocab_size": 50257,
        "seq_len": 1024,
        "batch_size": 1,
    },
    "GPT-2 XL (16k)": {
        "d_model": 1600,
        "num_layers": 48,
        "d_ff": 4288,
        "vocab_size": 50257,
        "seq_len": 16384,
        "batch_size": 1,
    },
}


def print_cli_analysis():
    """Prints tabular breakdown and deliverable answers for CS336."""
    print("=" * 96)
    print("  CS336 TRANSFORMER FLOPs & ARCHITECTURAL SCALING BREAKDOWN")
    print("=" * 96)

    header = (
        f"{'Model / Setting':<18} | {'SeqLen':<7} | {'Total (GFLOPs)':<15} | "
        f"{'Attn Total':<12} | {'(Core QK+SV)':<13} | {'FFN (SwiGLU)':<13} | {'LM Head':<10}"
    )
    print(header)
    print("-" * 96)

    for name, cfg in PRESETS.items():
        res = compute_decomposed_flops(
            batch_size=cfg["batch_size"],
            seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            vocab_size=cfg["vocab_size"],
            num_layers=cfg["num_layers"],
        )
        total_gflops = res["total_flops"] / 1e9
        row = (
            f"{name:<18} | {cfg['seq_len']:<7} | {total_gflops:>14.2f}  | "
            f"{res['pct_attn_total']:>10.2f}% | {res['pct_attn_core']:>11.2f}% | "
            f"{res['pct_ffn']:>11.2f}% | {res['pct_lm_head']:>8.2f}%"
        )
        print(row)
    print("=" * 96)

    print("\n" + "=" * 96)
    print("  DETAILED ANSWERS TO ASSIGNMENT DELIVERABLES")
    print("=" * 96)

    print("\n[Deliverable: List of Matrix Multiplies and Total FLOPs Formula]")
    print(
        "• QKV Projections (per layer): 3 GEMMs of (B*S, d_model) x (d_model, d_model) -> 6 * B * S * d_model^2 FLOPs\n"
        "• Attention Scores QK^T (per layer): (B*H, S, d_k) x (B*H, d_k, S) -> 2 * B * S^2 * d_model FLOPs\n"
        "• Attention Context A*V (per layer): (B*H, S, S) x (B*H, S, d_k) -> 2 * B * S^2 * d_model FLOPs\n"
        "• Output Projection (per layer): (B*S, d_model) x (d_model, d_model) -> 2 * B * S * d_model^2 FLOPs\n"
        "• SwiGLU FFN (per layer): 3 GEMMs (W1, W3, W2) between d_model & d_ff -> 6 * B * S * d_model * d_ff FLOPs\n"
        "• LM Head / Unembedding: (B*S, d_model) x (d_model, vocab_size) -> 2 * B * S * d_model * vocab_size FLOPs\n"
        "\nTotal Forward FLOPs Formula:\n"
        "  FLOPs = L * [ 8*B*S*d_model^2 + 4*B*S^2*d_model + 6*B*S*d_model*d_ff ] + 2*B*S*d_model*V\n"
    )

    print("[Deliverable (c): Which parts require the most FLOPs?]")
    print(
        "For standard sequence lengths (e.g. S=1024), the Feed-Forward Network (FFN/SwiGLU) consumes the largest\n"
        "share of FLOPs (~40%–58%), followed by the Multi-Head Attention projections and the LM Head (especially in smaller models)."
    )

    print("\n[Deliverable (d): Model scaling across GPT-2 Small, Medium, Large]")
    print(
        "As model width (d_model) and depth (layers) increase, the FFN and QKV/Output projections scale quadratically\n"
        "with d_model and linearly with layers (O(L * d_model^2)), causing their proportional share to increase,\n"
        "while the LM Head (which does not scale with L) and context-attention matmuls take up proportionally less of the total FLOPs."
    )

    print("\n[Deliverable (e): Increasing GPT-2 XL context length to 16,384]")
    print(
        "Increasing context length from 1,024 to 16,384 increases total FLOPs by ~38x (from 3.52 TFLOPs to 133.58 TFLOPs);\n"
        "because the core attention matmuls (QK^T and AV) scale quadratically with context length (O(S^2)), core attention\n"
        "dominates the compute, surging from 9.16% to 61.73% of total FLOPs, while FFN's relative contribution falls from 57.53% to 24.24%."
    )
    print("=" * 96 + "\n")


def build_interactive_dashboard():
    """Builds and launches an interactive Matplotlib dashboard."""
    fig = plt.figure(figsize=(16, 9), facecolor="#1e1e24")
    try:
        fig.canvas.manager.set_window_title("Transformer FLOPs Explorer & Component Visualizer")
    except Exception:
        pass

    # Layout: left 2 columns for graphs, right/bottom for sliders
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.06,
        right=0.96,
        bottom=0.28,
        top=0.92,
        wspace=0.35,
        hspace=0.35,
    )

    ax_bar = fig.add_subplot(gs[0, 0], facecolor="#2a2b36")
    ax_pie = fig.add_subplot(gs[0, 1], facecolor="#2a2b36")
    ax_seq_scale = fig.add_subplot(gs[0, 2], facecolor="#2a2b36")
    ax_dmodel_scale = fig.add_subplot(gs[1, :], facecolor="#2a2b36")

    # Theme colors
    colors = {
        "qkv": "#38bdf8",       # Sky blue
        "attn_core": "#f43f5e", # Bright Rose / Red (O(S^2))
        "attn_out": "#818cf8",  # Indigo
        "ffn": "#34d399",       # Emerald green
        "lm_head": "#fbbf24",   # Amber
    }

    # Initial state (GPT-2 Small)
    state = {
        "d_model": 768,
        "num_layers": 12,
        "seq_len": 1024,
        "vocab_size": 50257,
        "d_ff_mult": 8 / 3,  # d_ff = mult * d_model
        "batch_size": 1,
    }

    def update_plots(_=None):
        d_model = int(slider_dmodel.val)
        num_layers = int(slider_layers.val)
        seq_len = int(slider_seq.val)
        vocab_size = int(slider_vocab.val)
        d_ff = int(slider_dff_ratio.val * d_model)

        res = compute_decomposed_flops(
            batch_size=1,
            seq_len=seq_len,
            d_model=d_model,
            d_ff=d_ff,
            vocab_size=vocab_size,
            num_layers=num_layers,
        )

        total_gflops = res["total_flops"] / 1e9

        # --- 1. Subplot: Bar Chart (Absolute GFLOPs) ---
        ax_bar.clear()
        labels = ["QKV Proj", "QK & SV (S²)", "Attn Out", "FFN (SwiGLU)", "LM Head"]
        values = [
            res["qkv_proj"] / 1e9,
            res["attn_core"] / 1e9,
            res["attn_out"] / 1e9,
            res["ffn_total"] / 1e9,
            res["lm_head"] / 1e9,
        ]
        bar_colors = [colors["qkv"], colors["attn_core"], colors["attn_out"], colors["ffn"], colors["lm_head"]]

        bars = ax_bar.bar(labels, values, color=bar_colors, edgecolor="#ffffff", linewidth=0.5)
        ax_bar.set_title(f"Component FLOPs (Total: {total_gflops:,.1f} GFLOPs)", color="white", fontsize=11, fontweight="bold")
        ax_bar.set_ylabel("GFLOPs (10⁹ ops)", color="white", fontsize=9)
        ax_bar.tick_params(colors="white", labelsize=8)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.2, color="white")
        plt.setp(ax_bar.get_xticklabels(), rotation=20, ha="right")

        # Label values on bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + (max(values) * 0.02),
                    f"{h:.1f}G",
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=8,
                )
        ax_bar.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

        # --- 2. Subplot: Donut / Pie Chart (Proportions) ---
        ax_pie.clear()
        pie_labels = [
            f"QKV ({res['pct_qkv']:.1f}%)",
            f"Attn MatMul ({res['pct_attn_core']:.1f}%)",
            f"Attn Out ({res['pct_attn_out']:.1f}%)",
            f"FFN ({res['pct_ffn']:.1f}%)",
            f"LM Head ({res['pct_lm_head']:.1f}%)",
        ]
        wedges, _ = ax_pie.pie(
            values,
            labels=None,
            colors=bar_colors,
            startangle=140,
            wedgeprops=dict(width=0.45, edgecolor="#1e1e24", linewidth=1.5),
        )
        ax_pie.set_title("Relative Compute Share (%)", color="white", fontsize=11, fontweight="bold")
        ax_pie.legend(
            wedges,
            pie_labels,
            loc="center left",
            bbox_to_anchor=(0.95, 0.5),
            fontsize=7.5,
            frameon=False,
            labelcolor="white",
        )

        # --- 3. Subplot: Scaling vs Context Length (S) ---
        ax_seq_scale.clear()
        seq_range = np.logspace(np.log10(128), np.log10(32768), 50).astype(int)
        qkv_s, core_s, ffn_s, head_s = [], [], [], []

        for s in seq_range:
            r = compute_decomposed_flops(1, s, d_model, d_ff, vocab_size, num_layers)
            tot = r["total_flops"]
            qkv_s.append(r["qkv_proj"] / tot * 100)
            core_s.append(r["attn_core"] / tot * 100)
            ffn_s.append(r["ffn_total"] / tot * 100)
            head_s.append(r["lm_head"] / tot * 100)

        ax_seq_scale.plot(seq_range, core_s, label="Attn MatMul (O(S²))", color=colors["attn_core"], linewidth=2)
        ax_seq_scale.plot(seq_range, ffn_s, label="FFN (O(S))", color=colors["ffn"], linewidth=2)
        ax_seq_scale.plot(seq_range, qkv_s, label="QKV Proj (O(S))", color=colors["qkv"], linewidth=1.5)
        ax_seq_scale.plot(seq_range, head_s, label="LM Head (O(S))", color=colors["lm_head"], linewidth=1.5)

        ax_seq_scale.axvline(seq_len, color="#f59e0b", linestyle=":", linewidth=1.5, label=f"Current S={seq_len}")
        ax_seq_scale.set_xscale("log")
        ax_seq_scale.set_title("Compute % vs Sequence Length", color="white", fontsize=11, fontweight="bold")
        ax_seq_scale.set_xlabel("Context Length (log scale)", color="white", fontsize=9)
        ax_seq_scale.set_ylabel("% of Total FLOPs", color="white", fontsize=9)
        ax_seq_scale.tick_params(colors="white", labelsize=8)
        ax_seq_scale.grid(True, linestyle="--", alpha=0.2, color="white")
        ax_seq_scale.legend(fontsize=7, framealpha=0.3, loc="center left", labelcolor="white")

        # --- 4. Subplot: Scaling vs d_model and Layers ---
        ax_dmodel_scale.clear()
        d_range = np.linspace(256, 4096, 40).astype(int)
        attn_d, ffn_d, head_d = [], [], []

        for dm in d_range:
            dff_val = int(slider_dff_ratio.val * dm)
            r = compute_decomposed_flops(1, seq_len, dm, dff_val, vocab_size, num_layers)
            tot = r["total_flops"]
            attn_d.append(r["attn_total"] / tot * 100)
            ffn_d.append(r["ffn_total"] / tot * 100)
            head_d.append(r["lm_head"] / tot * 100)

        ax_dmodel_scale.plot(d_range, ffn_d, label="FFN Share % (O(L·d²))", color=colors["ffn"], linewidth=2.2)
        ax_dmodel_scale.plot(d_range, attn_d, label="Total Attention Share %", color=colors["qkv"], linewidth=2.2)
        ax_dmodel_scale.plot(d_range, head_d, label="LM Head Share % (O(d·V))", color=colors["lm_head"], linewidth=2.2)
        ax_dmodel_scale.axvline(d_model, color="#f59e0b", linestyle=":", linewidth=1.5, label=f"Current d_model={d_model}")

        ax_dmodel_scale.set_title("Proportional FLOPs vs Model Width (d_model)", color="white", fontsize=11, fontweight="bold")
        ax_dmodel_scale.set_xlabel("d_model", color="white", fontsize=9)
        ax_dmodel_scale.set_ylabel("% of Total FLOPs", color="white", fontsize=9)
        ax_dmodel_scale.tick_params(colors="white", labelsize=8)
        ax_dmodel_scale.grid(True, linestyle="--", alpha=0.2, color="white")
        ax_dmodel_scale.legend(fontsize=8, framealpha=0.3, loc="upper right", labelcolor="white")

        fig.canvas.draw_idle()

    # --- UI Controls Area (Bottom) ---
    slider_color = "#38bdf8"
    bg_slider = "#2a2b36"

    ax_sl_dmodel = fig.add_axes([0.10, 0.18, 0.35, 0.025], facecolor=bg_slider)
    ax_sl_layers = fig.add_axes([0.10, 0.13, 0.35, 0.025], facecolor=bg_slider)
    ax_sl_seq = fig.add_axes([0.10, 0.08, 0.35, 0.025], facecolor=bg_slider)
    ax_sl_vocab = fig.add_axes([0.10, 0.03, 0.35, 0.025], facecolor=bg_slider)

    ax_sl_dff = fig.add_axes([0.55, 0.18, 0.18, 0.025], facecolor=bg_slider)

    slider_dmodel = Slider(ax_sl_dmodel, "d_model", 128, 4096, valinit=state["d_model"], valstep=64, color=slider_color)
    slider_layers = Slider(ax_sl_layers, "Layers", 1, 96, valinit=state["num_layers"], valstep=1, color=slider_color)
    slider_seq = Slider(ax_sl_seq, "Context (S)", 128, 32768, valinit=state["seq_len"], valstep=128, color=slider_color)
    slider_vocab = Slider(ax_sl_vocab, "Vocab Size", 1000, 128000, valinit=state["vocab_size"], valstep=1000, color=slider_color)
    slider_dff_ratio = Slider(ax_sl_dff, "d_ff / d_model", 1.0, 4.0, valinit=state["d_ff_mult"], valstep=0.3333, color=slider_color)

    for sl in [slider_dmodel, slider_layers, slider_seq, slider_vocab, slider_dff_ratio]:
        sl.label.set_color("white")
        sl.valtext.set_color("white")
        sl.on_changed(update_plots)

    # Preset Buttons
    presets_names = list(PRESETS.keys())
    btn_axes = []
    buttons = []
    btn_start_x = 0.77
    btn_w = 0.18
    btn_h = 0.035

    for i, name in enumerate(presets_names):
        b_ax = fig.add_axes([btn_start_x, 0.18 - i * 0.04, btn_w, btn_h])
        btn = Button(b_ax, name, color="#2a2b36", hovercolor="#3b82f6")
        btn.label.set_color("white")
        btn.label.set_fontsize(8)

        def make_preset_handler(cfg):
            def handler(event):
                slider_dmodel.set_val(cfg["d_model"])
                slider_layers.set_val(cfg["num_layers"])
                slider_seq.set_val(cfg["seq_len"])
                slider_vocab.set_val(cfg["vocab_size"])
                slider_dff_ratio.set_val(cfg["d_ff"] / cfg["d_model"])
            return handler

        btn.on_clicked(make_preset_handler(PRESETS[name]))
        btn_axes.append(b_ax)
        buttons.append(btn)

    update_plots()
    plt.show()


def save_static_figures(output_path: str = "flops_analysis.png"):
    """Generates and saves static comparison figures for report and deliverables."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # Figure 1: Breakdown across GPT-2 Models
    ax1 = axes[0, 0]
    models = ["GPT-2 Small", "GPT-2 Medium", "GPT-2 Large", "GPT-2 XL (1k)", "GPT-2 XL (16k)"]
    attn_shares = []
    ffn_shares = []
    head_shares = []

    for name in models:
        cfg = PRESETS[name]
        r = compute_decomposed_flops(1, cfg["seq_len"], cfg["d_model"], cfg["d_ff"], cfg["vocab_size"], cfg["num_layers"])
        attn_shares.append(r["pct_attn_total"])
        ffn_shares.append(r["pct_ffn"])
        head_shares.append(r["pct_lm_head"])

    x = np.arange(len(models))
    w = 0.55
    ax1.bar(x, ffn_shares, w, label="FFN (SwiGLU)", color="#10b981")
    ax1.bar(x, attn_shares, w, bottom=ffn_shares, label="Attention (Total)", color="#0ea5e9")
    ax1.bar(x, head_shares, w, bottom=np.array(ffn_shares) + np.array(attn_shares), label="LM Head", color="#f59e0b")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("% of Total FLOPs")
    ax1.set_title("Compute Allocation Across Model Sizes", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Figure 2: Sequence Length Scaling (Quadratic vs Linear)
    ax2 = axes[0, 1]
    s_vals = np.logspace(np.log10(128), np.log10(32768), 50).astype(int)
    core_pcts = []
    ffn_pcts = []
    for s in s_vals:
        r = compute_decomposed_flops(1, s, 1600, 4288, 50257, 48)  # GPT-2 XL
        core_pcts.append(r["pct_attn_core"])
        ffn_pcts.append(r["pct_ffn"])

    ax2.plot(s_vals, core_pcts, label="Attention MatMuls QK^T + SV (O(S²))", color="#ef4444", linewidth=2)
    ax2.plot(s_vals, ffn_pcts, label="FFN (O(S))", color="#10b981", linewidth=2)
    ax2.axvline(1024, linestyle=":", color="gray", label="1k Context")
    ax2.axvline(16384, linestyle="--", color="purple", label="16k Context")
    ax2.set_xscale("log")
    ax2.set_xlabel("Context Length (Tokens)")
    ax2.set_ylabel("% of Total FLOPs")
    ax2.set_title("GPT-2 XL: Attn MatMul Explosion with Context", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Figure 3: Model Width Scaling
    ax3 = axes[1, 0]
    dm_vals = np.linspace(256, 3072, 50).astype(int)
    ffn_p = []
    head_p = []
    for dm in dm_vals:
        r = compute_decomposed_flops(1, 1024, dm, int(8 * dm / 3), 50257, 24)
        ffn_p.append(r["pct_ffn"])
        head_p.append(r["pct_lm_head"])

    ax3.plot(dm_vals, ffn_p, label="FFN Share % (O(L·d²))", color="#10b981", linewidth=2)
    ax3.plot(dm_vals, head_p, label="LM Head Share % (O(d·V))", color="#f59e0b", linewidth=2)
    ax3.set_xlabel("d_model (Model Width)")
    ax3.set_ylabel("% of Total FLOPs")
    ax3.set_title("Impact of Width Scaling (Fixed L=24, S=1024)", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Figure 4: Total GFLOPs Comparison
    ax4 = axes[1, 1]
    gflops = [
        compute_decomposed_flops(
            1,
            PRESETS[m]["seq_len"],
            PRESETS[m]["d_model"],
            PRESETS[m]["d_ff"],
            PRESETS[m]["vocab_size"],
            PRESETS[m]["num_layers"],
        )["total_flops"]
        / 1e9
        for m in models
    ]
    bars = ax4.bar(models, gflops, color=["#38bdf8", "#818cf8", "#c084fc", "#f43f5e", "#fb7185"])
    ax4.set_yscale("log")
    ax4.set_ylabel("Total Forward FLOPs (GFLOPs, Log Scale)")
    ax4.set_title("Total FLOPs per Forward Pass (Batch Size = 1)", fontweight="bold")
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax4.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, h * 1.15, f"{h:,.0f}G", ha="center", va="bottom", fontsize=7.5)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Saved high-resolution comparison figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transformer FLOPs Breakdown & Visualizer")
    parser.add_argument("--cli", action="store_true", help="Print tabular FLOPs breakdown and deliverable text to console")
    parser.add_argument("--save", action="store_true", help="Save static breakdown figure as PNG")
    parser.add_argument("--output", type=str, default="flops_analysis.png", help="Path for saved figure")
    args = parser.parse_args()

    if args.save:
        save_static_figures(args.output)
    elif args.cli:
        print_cli_analysis()
    else:
        # Default behavior: Print analysis and try to launch GUI
        print_cli_analysis()
        try:
            build_interactive_dashboard()
        except Exception as e:
            print(f"[Note] Could not open GUI window (headless/no display): {e}")
            print("Run with '--save' to generate PNG plots or '--cli' for terminal analysis.")


if __name__ == "__main__":
    main()
