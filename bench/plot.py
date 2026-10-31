"""Charts for the README from results/scheduling.json (writes figures/scheduling.png)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = {"sequential": "#eb6834", "static": "#eda100", "continuous": "#2a78d6"}
LABELS = {"sequential": "One request at a time", "static": "Static batching (16)", "continuous": "Continuous batching (16)"}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel("Arrival rate (requests per second)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


runs = json.loads((ROOT / "results" / "scheduling.json").read_text())["runs"]
rates = [0.5, 1.0, 2.0, "burst"]
xs = range(len(rates))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)
fig.set_facecolor(SURFACE)
for policy in ("sequential", "static", "continuous"):
    rows = {r["arrival_rate"]: r for r in runs if r["policy"] == policy}
    ttft = [rows[r]["ttft_p50_s"] for r in rates]
    thr = [rows[r]["throughput_tok_s"] for r in rates]
    for ax, ys in ((axes[0], ttft), (axes[1], thr)):
        ax.plot(xs, ys, color=COLORS[policy], linewidth=2, marker="o", markersize=5, label=LABELS[policy])
axes[0].set_yscale("log")
axes[0].set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
axes[0].minorticks_off()
axes[1].set_ylim(0, 115)
style(axes[0], "Median time to first token (log scale)", "Seconds")
style(axes[1], "Throughput", "Generated tokens per second")
for ax in axes:
    ax.set_xticks(list(xs), [str(r) for r in rates])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle("Qwen2.5-0.5B on 4 CPU cores, 48 requests, 32-128 output tokens", color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(ROOT / "figures" / "scheduling.png", facecolor=SURFACE)
