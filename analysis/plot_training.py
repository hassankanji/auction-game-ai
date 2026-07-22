"""Plot training curves from models/train_log.json -> analysis/training_curves.png."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    with open(os.path.join(ROOT, "models", "train_log.json")) as f:
        log = json.load(f)
    it = [r["iter"] for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for key, label in [("vs_random", "vs random"), ("vs_value", "vs value"),
                       ("vs_strategic", "vs strategic"), ("vs_mixed", "vs mixed pool")]:
        ax1.plot(it, [r[key] for r in log], label=label, linewidth=2)
    ax1.axhline(0.20, color="gray", ls="--", lw=1, label="chance (0.20)")
    ax1.set_title("Win-share vs reference opponents")
    ax1.set_xlabel("iteration"); ax1.set_ylabel("win-share"); ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(it, [r["entropy"] for r in log], color="#7c5cff", lw=2, label="policy entropy")
    ax2.plot(it, [r["qual_rate"] for r in log], color="#3ad29f", lw=2, label="qualification rate")
    ax2.set_title("Policy entropy & qualification rate")
    ax2.set_xlabel("iteration"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "training_curves.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
