"""
PSRO / double-oracle robustification: make the bot hard to *exploit*, not just good against
fixed opponents.

Self-play PPO finds a policy that beats fixed heuristics but can still be countered by a
dedicated best response. Here we close that gap:

    repeat R rounds:
      1. train a best-response "exploiter" against the current (frozen) bot,
      2. add that exploiter to the bot's opponent pool,
      3. retrain the bot against the pool (past selves + all exploiters found so far),
    then export the robustified bot and report how much its exploitability fell.

    python -m training.robustify --rounds 3

Starts from training/checkpoints/best.pt and writes the robustified model to
training/checkpoints/robust.pt (and, if it evaluates stronger overall, you can promote it
with training/finalize).
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch

from .policy import PolicyValueNet, export_weights
from .selfplay import rollout, ppo_update

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_net(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["net"]
    hidden = sd["body.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("body.") and k.endswith(".weight") and sd[k].dim() == 2)
    net = PolicyValueNet(hidden=hidden, depth=max(1, depth))
    net.load_state_dict(sd); net.eval()
    return net, hidden, max(1, depth)


def snapshot(net, hidden, depth):
    s = PolicyValueNet(hidden=hidden, depth=depth)
    s.load_state_dict(copy.deepcopy(net.state_dict())); s.eval()
    return s


def best_response(main, hidden, rng, iters, games, lr=5e-4):
    """Train an exploiter (seat 0) vs 4 frozen copies of `main`. Returns (exploiter, win-share)."""
    ex = PolicyValueNet(hidden=hidden)
    opt = torch.optim.Adam(ex.parameters(), lr=lr)
    for _ in range(iters):
        b = rollout(ex, games, rng, frozen_pool=[main], frozen_prob=1.0)
        ppo_update(ex, opt, b)
    ev = rollout(ex, 4000, rng, frozen_pool=[main], frozen_prob=1.0)
    return ex, ev.win_rate_learner


def measure_exploitability(main, hidden, rng, iters, games):
    _, wr = best_response(main, hidden, rng, iters, games)
    return wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best.pt"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--br-iters", type=int, default=90)     # exploiter training per round
    ap.add_argument("--defend-iters", type=int, default=160)  # bot retraining per round
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--frozen-prob", type=float, default=0.7)
    ap.add_argument("--ent-coef", type=float, default=0.008)
    ap.add_argument("--pool-cap", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    main_net, hidden, depth = load_net(args.ckpt)
    opt = torch.optim.Adam(main_net.parameters(), lr=args.lr)

    # Baseline exploitability of the starting model.
    base = measure_exploitability(main_net, hidden, rng, args.br_iters, args.games)
    print(f"[baseline] best-response win-share vs start model = {base:.3f} "
          f"(0.20=chance; gap={base-0.2:+.3f})", flush=True)

    pool = [snapshot(main_net, hidden, depth)]
    exploit_history = [base]

    for r in range(args.rounds):
        # 1. Find a best response to the current bot.
        ex, ex_wr = best_response(main_net, hidden, rng, args.br_iters, args.games)
        ex.eval()
        print(f"[round {r+1}] found exploiter with win-share {ex_wr:.3f}", flush=True)
        pool.append(ex)

        # 2. Retrain the bot against the pool (past selves + all exploiters).
        for it in range(args.defend_iters):
            b = rollout(main_net, args.games, rng, frozen_pool=pool, frozen_prob=args.frozen_prob,
                        win_weight=1.0, rank_weight=0.3)
            ppo_update(main_net, opt, b, ent_coef=args.ent_coef)
        pool.append(snapshot(main_net, hidden, depth))
        if len(pool) > args.pool_cap:
            pool = pool[-args.pool_cap:]

        # 3. Re-measure exploitability after defending.
        wr = measure_exploitability(main_net, hidden, rng, args.br_iters, args.games)
        exploit_history.append(wr)
        print(f"[round {r+1}] after defending, best-response win-share = {wr:.3f} "
              f"(gap={wr-0.2:+.3f})", flush=True)

        torch.save({"net": main_net.state_dict(), "round": r + 1},
                   os.path.join(HERE, "checkpoints", "robust.pt"))

    print("\n==== Robustification summary ====")
    print("best-response win-share over rounds:", [f"{x:.3f}" for x in exploit_history])
    print(f"start gap {exploit_history[0]-0.2:+.3f} -> final gap {exploit_history[-1]-0.2:+.3f}")

    # Export robustified model for finalize/compare.
    with open(os.path.join(ROOT, "models", "policy_robust.json"), "w") as f:
        import json
        json.dump(export_weights(main_net), f)
    print("saved training/checkpoints/robust.pt and models/policy_robust.json")


if __name__ == "__main__":
    main()
