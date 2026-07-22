"""
Pick the strongest trained checkpoint, export it for the web app, and regenerate the
in-browser parity reference.

    python -m training.finalize

Compares every training/checkpoints/*.pt on a large, stable evaluation (win-share vs the
strategic heuristic + mixed pool) and exports the winner to models/policy.json, then dumps
models/parity_ref.json from the same weights.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from .policy import PolicyValueNet, export_weights
from .evaluate import evaluate_suite

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_net(path: str) -> PolicyValueNet:
    ck = torch.load(path, map_location="cpu")
    sd = ck["net"]
    hidden = sd["body.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("body.") and k.endswith(".weight") and sd[k].dim() == 2)
    net = PolicyValueNet(hidden=hidden, depth=max(1, depth))
    net.load_state_dict(sd)
    net.eval()
    return net


def score_net(net, n_games=800, seed=7):
    ev = evaluate_suite(net, n_games=n_games, seed=seed)
    ev["blend"] = 0.6 * ev["vs_strategic"] + 0.4 * ev["vs_mixed"]
    return ev


def main():
    cands = sorted(glob.glob(os.path.join(HERE, "checkpoints", "*.pt")))
    # Prefer explicit best checkpoints; always include them if present.
    if not cands:
        raise SystemExit("no checkpoints found — train first")

    print("Evaluating candidates (win-share; 0.20 = chance):")
    results = []
    for path in cands:
        name = os.path.basename(path)
        try:
            net = load_net(path)
        except Exception as e:
            print(f"  {name}: skipped ({e})")
            continue
        ev = score_net(net)
        results.append((ev["blend"], path, net, ev))
        print(f"  {name:16s} blend={ev['blend']:.3f}  vs_strat={ev['vs_strategic']:.3f} "
              f"vs_mix={ev['vs_mixed']:.3f} vs_val={ev['vs_value']:.3f} vs_rand={ev['vs_random']:.3f}")

    results.sort(key=lambda r: -r[0])
    best_blend, best_path, best_net, best_ev = results[0]
    print(f"\nWinner: {os.path.basename(best_path)}  (blend={best_blend:.3f})")

    # Export for the web app.
    out = os.path.join(ROOT, "models", "policy.json")
    with open(out, "w") as f:
        json.dump(export_weights(best_net), f)
    print(f"exported -> {out}")

    # Record the chosen model's stats.
    with open(os.path.join(ROOT, "models", "final_eval.json"), "w") as f:
        json.dump({"winner": os.path.basename(best_path), **best_ev}, f, indent=2)

    # Regenerate parity reference from the winning net.
    from .features import features_batch
    from engine.game import N_PLAYERS
    rng = np.random.default_rng(12345)
    cases = []
    for _ in range(40):
        r = int(rng.integers(0, 10))
        cash = rng.uniform(0, 500, size=N_PLAYERS)
        won = rng.random(N_PLAYERS) < 0.6
        proj = rng.uniform(0, 1500, size=N_PLAYERS)
        me = int(rng.integers(0, N_PLAYERS))
        feats = features_batch(cash[None], won[None], proj[None], r)[0, me]
        with torch.no_grad():
            logits, value = best_net(torch.as_tensor(feats[None], dtype=torch.float32))
        cases.append({"cash": cash.tolist(), "won": won.astype(int).tolist(),
                      "proj": proj.tolist(), "round": r, "me": me,
                      "features": feats.astype(float).tolist(),
                      "logits": logits[0].numpy().astype(float).tolist(),
                      "value": float(value[0])})
    with open(os.path.join(ROOT, "models", "parity_ref.json"), "w") as f:
        json.dump({"cases": cases}, f)
    print("wrote models/parity_ref.json")


if __name__ == "__main__":
    main()
