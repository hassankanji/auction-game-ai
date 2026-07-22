"""
Dump reference (features, logits, value) for random states from the trained net, so the
in-browser TypeScript port can verify it reproduces them exactly.

    python -m training.dump_parity            # uses checkpoints/best.pt

Writes models/parity_ref.json (copied into web/public by the web build's copy-model step).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from .policy import PolicyValueNet
from .features import features_batch
from engine.game import N_PLAYERS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(n_cases: int = 40, seed: int = 12345):
    ckpt = os.path.join(HERE, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(HERE, "checkpoints", "last.pt")
    state = torch.load(ckpt, map_location="cpu")
    # Infer architecture from the checkpoint.
    sd = state["net"]
    hidden = sd["body.0.weight"].shape[0]
    # body layout is [Linear, LayerNorm, Tanh] * depth -> count the Linear weights.
    depth = sum(1 for k in sd if k.startswith("body.") and k.endswith(".weight")
                and sd[k].dim() == 2)
    net = PolicyValueNet(hidden=hidden, depth=max(1, depth))
    net.load_state_dict(sd)
    net.eval()

    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        r = int(rng.integers(0, 10))
        cash = rng.uniform(0, 500, size=N_PLAYERS)
        won = (rng.random(N_PLAYERS) < 0.6)
        proj = rng.uniform(0, 1500, size=N_PLAYERS)
        me = int(rng.integers(0, N_PLAYERS))
        feats = features_batch(cash[None, :], won[None, :], proj[None, :], r)[0, me]
        with torch.no_grad():
            logits, value = net(torch.as_tensor(feats[None, :], dtype=torch.float32))
        cases.append({
            "cash": cash.tolist(),
            "won": won.astype(int).tolist(),
            "proj": proj.tolist(),
            "round": r,
            "me": me,
            "features": feats.astype(float).tolist(),
            "logits": logits[0].numpy().astype(float).tolist(),
            "value": float(value[0]),
        })

    out = os.path.join(ROOT, "models", "parity_ref.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {len(cases)} parity cases -> {out}")


if __name__ == "__main__":
    main()
