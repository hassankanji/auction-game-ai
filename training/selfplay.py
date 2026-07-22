"""
Vectorised self-play rollouts + PPO update for the auction bot.

Design notes
------------
* All 5 seats share one policy (a symmetric equilibrium is the target). By default every
  seat is controlled by the current learner (pure self-play) and every eligible decision is
  recorded. A pool of *frozen* past snapshots can be mixed in (fictitious self-play): some
  opponent seats are played by a frozen net, and only the learner's decisions are recorded.
  This makes the policy robust / hard to exploit rather than tuned to one opponent.

* Reward is terminal only. Faithful to the rules ("highest final score wins") the primary
  reward is the tournament win-share (1 for a sole winner, 1/k for a k-way tie for 1st,
  else 0), plus a small linear-rank shaping term so 2nd is better than 5th. With gamma=1 and
  no intermediate reward, the Monte-Carlo return for every decision a seat made equals that
  seat's terminal reward, and the value head is a learned baseline. PPO clips the policy
  ratio; advantages are the (return - value) normalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from engine.game import RETURNS, N_ROUNDS, N_PLAYERS, QUALIFY_ROUND_IDX, HAIRCUT
from .features import features_batch, FEATURE_DIM
from .policy import PolicyValueNet, actions_to_bids, N_ACTIONS

FINAL_CASH_MULT = 0.9


def compute_rewards(final_score: np.ndarray, win_weight: float = 1.0,
                    rank_weight: float = 0.3) -> np.ndarray:
    """final_score (B,5) -> reward (B,5).

    win_share: 1 for a sole top score, 1/k for a k-way tie at the top, else 0.
    linear_rank: in [-1,1], expected value if ties are coin-flipped.
    """
    B = final_score.shape[0]
    s = final_score
    top = s.max(axis=1, keepdims=True)
    is_top = np.isclose(s, top)
    k = is_top.sum(axis=1, keepdims=True)
    win_share = np.where(is_top, 1.0 / k, 0.0)

    # linear rank
    gt = (s[:, :, None] > s[:, None, :]).sum(axis=2)          # strictly beaten
    eq = np.isclose(s[:, :, None], s[:, None, :]).sum(axis=2) - 1  # ties (excl self)
    linear = (gt + 0.5 * eq) / (N_PLAYERS - 1) * 2.0 - 1.0

    return (win_weight * win_share + rank_weight * linear).astype(np.float32)


@dataclass
class Batch:
    feats: np.ndarray      # (M, F)
    actions: np.ndarray    # (M,)
    logp: np.ndarray       # (M,)
    values: np.ndarray     # (M,)
    returns: np.ndarray    # (M,)  terminal reward for that seat
    # diagnostics
    win_rate_learner: float
    mean_reward: float
    qual_rate: float


def rollout(net: PolicyValueNet, n_games: int, rng: np.random.Generator,
            frozen_pool: Optional[list] = None, frozen_prob: float = 0.0,
            temperature: float = 1.0, win_weight: float = 1.0,
            rank_weight: float = 0.3) -> Batch:
    """Play `n_games` in parallel and return recorded learner decisions."""
    B = n_games
    cash = np.full((B, N_PLAYERS), 500.0)
    won = np.zeros((B, N_PLAYERS), dtype=bool)
    proj = np.zeros((B, N_PLAYERS))

    # Controller assignment: 0 = learner; >=1 index into frozen_pool (+1). Seat 0 is always
    # the learner so we always record data; other seats may be frozen opponents.
    controller = np.zeros((B, N_PLAYERS), dtype=np.int64)
    if frozen_pool and frozen_prob > 0.0:
        for seat in range(1, N_PLAYERS):
            use_frozen = rng.random(B) < frozen_prob
            idx = rng.integers(0, len(frozen_pool), size=B) + 1
            controller[:, seat] = np.where(use_frozen, idx, 0)

    rec_feats, rec_actions, rec_logp, rec_values, rec_game, rec_seat = [], [], [], [], [], []

    for r in range(N_ROUNDS):
        feats = features_batch(cash, won, proj, r)          # (B,5,F)
        if r < QUALIFY_ROUND_IDX:
            eligible = np.ones((B, N_PLAYERS), dtype=bool)
        else:
            eligible = won.copy()

        actions = np.zeros((B, N_PLAYERS), dtype=np.int64)
        logp = np.zeros((B, N_PLAYERS), dtype=np.float32)
        values = np.zeros((B, N_PLAYERS), dtype=np.float32)

        # Each controller (0 = learner, >=1 = frozen snapshot) acts on its eligible seats.
        n_controllers = 1 + (len(frozen_pool) if frozen_pool else 0)
        for cid in range(n_controllers):
            mask = (controller == cid) & eligible
            if not mask.any():
                continue
            f = feats[mask]                                  # (m,F)
            model = net if cid == 0 else frozen_pool[cid - 1]
            a, lp, v = model.act(f, greedy=False, temperature=temperature)
            actions[mask] = a
            logp[mask] = lp
            values[mask] = v

        bids = actions_to_bids(actions, cash)                # (B,5) int
        bids = np.where(eligible, bids, -1)

        # Winner per game (random tie-break among top eligible bids).
        max_bid = bids.max(axis=1)                           # (B,)
        is_top = eligible & (bids == max_bid[:, None]) & (max_bid[:, None] >= 0)
        key = np.where(is_top, rng.random((B, N_PLAYERS)) + 1.0, -1.0)
        winner = key.argmax(axis=1)                          # (B,)
        gi = np.arange(B)
        win_bid = bids[gi, winner]

        # Apply outcome.
        cash[gi, winner] -= win_bid
        proj[gi, winner] += RETURNS[r]
        if r < QUALIFY_ROUND_IDX:
            won[gi, winner] = True

        # Record learner decisions (eligible, controller==0).
        rmask = (controller == 0) & eligible
        if rmask.any():
            gg, ss = np.where(rmask)
            rec_feats.append(feats[rmask])
            rec_actions.append(actions[rmask])
            rec_logp.append(logp[rmask])
            rec_values.append(values[rmask])
            rec_game.append(gg)
            rec_seat.append(ss)

        if r == 4:                                           # haircut after round 5
            cash = cash * HAIRCUT

    final_score = proj + FINAL_CASH_MULT * cash              # (B,5)
    reward = compute_rewards(final_score, win_weight, rank_weight)  # (B,5)

    feats = np.concatenate(rec_feats, axis=0)
    actions = np.concatenate(rec_actions, axis=0)
    logp = np.concatenate(rec_logp, axis=0)
    values = np.concatenate(rec_values, axis=0)
    game = np.concatenate(rec_game, axis=0)
    seat = np.concatenate(rec_seat, axis=0)
    returns = reward[game, seat]

    # Diagnostics: learner is seat 0 (always learner). Win rate of seat 0.
    top = final_score.max(axis=1, keepdims=True)
    is_top0 = np.isclose(final_score[:, 0:1], top).squeeze(1)
    k = np.isclose(final_score, top).sum(axis=1)
    win_rate = float(np.mean(is_top0 / k))
    qual_rate = float(won.mean())

    return Batch(feats, actions, logp, values, returns,
                 win_rate_learner=win_rate,
                 mean_reward=float(returns.mean()),
                 qual_rate=qual_rate)


def ppo_update(net: PolicyValueNet, opt: torch.optim.Optimizer, batch: Batch,
               epochs: int = 4, minibatch: int = 8192, clip: float = 0.2,
               vf_coef: float = 0.5, ent_coef: float = 0.01,
               max_grad_norm: float = 0.5) -> dict:
    feats = torch.as_tensor(batch.feats, dtype=torch.float32)
    actions = torch.as_tensor(batch.actions, dtype=torch.long)
    old_logp = torch.as_tensor(batch.logp, dtype=torch.float32)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32)

    adv = returns - torch.as_tensor(batch.values, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    n = feats.shape[0]
    idx = np.arange(n)
    stats = {}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch):
            mb = idx[start:start + minibatch]
            mb_t = torch.as_tensor(mb, dtype=torch.long)
            logits, value = net(feats[mb_t])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(actions[mb_t])
            ratio = torch.exp(new_logp - old_logp[mb_t])
            a = adv[mb_t]
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            pi_loss = -torch.min(unclipped, clipped).mean()
            v_loss = ((value - returns[mb_t]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            stats = {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
                     "entropy": ent.item()}
    return stats
