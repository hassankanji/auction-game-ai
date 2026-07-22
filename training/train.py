"""
Train the self-play auction bot.

Example:
    python -m training.train --iters 1200 --games 2048 --snapshot-every 40 \
        --out models/policy.json

Produces:
    training/checkpoints/last.pt           latest weights (resumable)
    training/checkpoints/best.pt           best-by-eval weights
    models/policy.json                     best weights exported for the web app
    models/train_log.json                  metrics per evaluated iteration
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from .policy import PolicyValueNet, export_weights
from .selfplay import rollout, ppo_update
from .evaluate import evaluate_suite

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_DIR = os.path.join(HERE, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--win-weight", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--frozen-prob", type=float, default=0.25)
    ap.add_argument("--pool-size", type=int, default=6)
    ap.add_argument("--snapshot-every", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "models", "policy.json"))
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    net = PolicyValueNet(hidden=args.hidden, depth=args.depth)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume and os.path.exists(os.path.join(CKPT_DIR, "last.pt")):
        ck = torch.load(os.path.join(CKPT_DIR, "last.pt"), map_location="cpu")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck.get("iter", 0)
        print(f"resumed from iter {start_iter}")

    frozen_pool: list = []
    log = []
    best_metric = -1e9
    t0 = time.time()

    for it in range(start_iter, args.iters):
        use_pool = frozen_pool if it > 0 else None
        batch = rollout(net, args.games, rng, frozen_pool=use_pool,
                        frozen_prob=args.frozen_prob if frozen_pool else 0.0,
                        temperature=args.temperature,
                        win_weight=args.win_weight, rank_weight=args.rank_weight)
        stats = ppo_update(net, opt, batch, epochs=args.ppo_epochs,
                           minibatch=args.minibatch, clip=args.clip,
                           ent_coef=args.ent_coef)

        if (it + 1) % args.snapshot_every == 0:
            snap = PolicyValueNet(hidden=args.hidden, depth=args.depth)
            snap.load_state_dict(copy.deepcopy(net.state_dict()))
            snap.eval()
            frozen_pool.append(snap)
            if len(frozen_pool) > args.pool_size:
                frozen_pool.pop(0)

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            ev = evaluate_suite(net, n_games=args.eval_games, seed=args.seed + it)
            # Select the best checkpoint by a STABLE blend that weights the strongest fixed
            # reference (the strategic heuristic) plus the mixed pool. vs_mixed alone is noisy
            # (it includes random opponents), which made earlier runs pick lucky peaks.
            metric = 0.6 * ev["vs_strategic"] + 0.4 * ev["vs_mixed"]
            elapsed = time.time() - t0
            print(f"[{it+1:5d}/{args.iters}] "
                  f"selfplay_win={batch.win_rate_learner:.3f} "
                  f"qual={batch.qual_rate:.2f} "
                  f"R={batch.mean_reward:+.3f} "
                  f"ent={stats.get('entropy', 0):.3f} "
                  f"| vs_rand={ev['vs_random']:.2f} vs_val={ev['vs_value']:.2f} "
                  f"vs_strat={ev['vs_strategic']:.2f} vs_mix={ev['vs_mixed']:.2f} "
                  f"| pool={len(frozen_pool)} {elapsed:.0f}s", flush=True)
            log.append({"iter": it + 1, **ev,
                        "selfplay_win": batch.win_rate_learner,
                        "qual_rate": batch.qual_rate,
                        "mean_reward": batch.mean_reward,
                        "entropy": stats.get("entropy", 0.0)})
            with open(os.path.join(ROOT, "models", "train_log.json"), "w") as f:
                json.dump(log, f, indent=2)

            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": it + 1},
                       os.path.join(CKPT_DIR, "last.pt"))
            if metric > best_metric:
                best_metric = metric
                torch.save({"net": net.state_dict(), "iter": it + 1},
                           os.path.join(CKPT_DIR, "best.pt"))
                with open(args.out, "w") as f:
                    json.dump(export_weights(net), f)
                print(f"        new best blend={metric:.3f} -> exported {args.out}",
                      flush=True)

    # Always export final too (in case best == final).
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": args.iters},
               os.path.join(CKPT_DIR, "last.pt"))
    print(f"done in {time.time()-t0:.0f}s. best vs_mixed={best_metric:.3f}")


if __name__ == "__main__":
    main()
