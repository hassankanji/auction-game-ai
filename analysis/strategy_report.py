"""
Probe the trained policy at canonical decision points and print what it does. This is how
we read the strategy out of the bot in human terms (used in the strategy writeup).

    python -m analysis.strategy_report
"""

from __future__ import annotations

import os

import numpy as np
import torch

from training.policy import PolicyValueNet, BID_FRACTIONS
from training.features import features_batch
from engine.game import RETURNS, N_PLAYERS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["net"]
    hidden = sd["body.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("body.") and k.endswith(".weight") and sd[k].dim() == 2)
    net = PolicyValueNet(hidden=hidden, depth=max(1, depth))
    net.load_state_dict(sd); net.eval()
    return net


def dist(net, cash, won, proj, r, me):
    feats = features_batch(np.array(cash)[None], np.array(won)[None], np.array(proj)[None], r)[0, me]
    with torch.no_grad():
        logits, value = net(torch.as_tensor(feats[None], dtype=torch.float32))
    probs = torch.softmax(logits, -1)[0].numpy()
    mx = int(np.floor(cash[me]))
    bids = np.floor(BID_FRACTIONS * mx).astype(int)
    # merge by bid
    agg = {}
    for b, p in zip(bids, probs):
        agg[b] = agg.get(b, 0) + p
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:6]
    return top, float(value[0])


def show(title, net, cash, won, proj, r, me):
    top, val = dist(net, cash, won, proj, r, me)
    ev = RETURNS[r]
    print(f"\n### {title}")
    print(f"    round {r+1} (project ${ev}m), you=seat{me} cash=${cash[me]:.0f}m "
          f"qualified={bool(won[me])}  value~{val:+.2f}")
    line = "    bid mix: " + "  ".join(f"${b}m:{p*100:4.1f}%" for b, p in top)
    print(line)


def main():
    ckpt = os.path.join(ROOT, "training", "checkpoints", "best.pt")
    net = load(ckpt)

    # 1. Round 1, nobody qualified yet - how much to pay for a cheap qualifying win?
    show("R1 opening - nobody qualified", net,
         cash=[500]*5, won=[0]*5, proj=[0]*5, r=0, me=0)

    # 2. Round 6, you're the ONLY one not qualified - desperation to qualify.
    show("R6 - you are the only one not yet qualified", net,
         cash=[500, 470, 460, 480, 465], won=[0, 1, 1, 1, 1], proj=[0, 25, 50, 100, 50], r=5, me=0)

    # 3. Round 6, you're already qualified and it's expensive - do you pass?
    show("R6 - already qualified, preserve cash?", net,
         cash=[450]*5, won=[1]*5, proj=[50, 50, 100, 25, 200], r=5, me=0)

    # 4. Round 10 ($1000) with a decisive budget lead.
    show("R10 finale - you hold the biggest budget", net,
         cash=[400, 250, 180, 120, 90], won=[1]*5, proj=[300]*5, r=9, me=0)

    # 5. Round 10 ($1000), you trail on budget.
    show("R10 finale - you are short on cash", net,
         cash=[90, 400, 350, 300, 250], won=[1]*5, proj=[300]*5, r=9, me=0)

    # 6. Round 7 ($500), all qualified, even budgets.
    show("R7 - first big round, even budgets", net,
         cash=[450]*5, won=[1]*5, proj=[80, 60, 100, 40, 120], r=6, me=0)

    print("\n(Values are the net's win-share signal for that seat; bid mix is the equilibrium "
          "distribution the bot samples from.)")


if __name__ == "__main__":
    main()
