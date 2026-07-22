"""
State -> feature-vector encoding, used identically by:
  * self-play training (batched, numpy),
  * Python evaluation, and
  * the in-browser bot (ported to TypeScript in web/src/bot/features.ts).

Keep this deterministic and simple; any change here MUST be mirrored in the TS port,
and tests/test_features_parity.py checks a few reference vectors match.

Feature layout (length = FEATURE_DIM):
  [0:10]   round one-hot (round index 0..9)
  [10]     project return / 1000
  [11]     rounds left / 10
  [12]     my cash / 500
  [13]     my project value / 2000
  [14]     my qualified flag (won any of rounds 1-6)
  [15]     my provisional score / 2000            (score = proj + 0.9*cash)
  [16:32]  4 opponents sorted by score desc, each [cash/500, proj/2000, qual, score/2000]
  [32]     fraction of players qualified / 1
  [33]     am I the current score leader (1/0)
  [34]     fraction of opponents with more cash than me
  [35]     leader gap = (best opponent score - my score) / 2000  (positive if behind)
"""

from __future__ import annotations

import numpy as np

from engine.game import RETURNS, N_ROUNDS, N_PLAYERS

CASH_NORM = 500.0
PROJ_NORM = 2000.0
RET_NORM = 1000.0
SCORE_NORM = 2000.0
FINAL_CASH_MULT = 0.9

FEATURE_DIM = 36

# Static per-seat opponent index table: OPP_IDX[me] = the other four seats, in order.
OPP_IDX = np.array(
    [[p for p in range(N_PLAYERS) if p != me] for me in range(N_PLAYERS)],
    dtype=np.int64,
)  # shape (5, 4)


def features_batch(cash: np.ndarray, won_any: np.ndarray, proj: np.ndarray,
                   round_idx: int) -> np.ndarray:
    """
    Vectorised features for every seat of every game.

    Args:
      cash:    (B, 5) float
      won_any: (B, 5) bool/float
      proj:    (B, 5) float
      round_idx: int in [0, 9]
    Returns:
      (B, 5, FEATURE_DIM) float32
    """
    B = cash.shape[0]
    cash = cash.astype(np.float64)
    won = won_any.astype(np.float64)
    proj = proj.astype(np.float64)

    score = proj + FINAL_CASH_MULT * cash               # (B,5)

    feats = np.zeros((B, N_PLAYERS, FEATURE_DIM), dtype=np.float64)

    # Round one-hot + globals (broadcast across seats).
    feats[:, :, round_idx] = 1.0
    feats[:, :, 10] = RETURNS[round_idx] / RET_NORM
    feats[:, :, 11] = (N_ROUNDS - round_idx) / 10.0

    # My own block.
    feats[:, :, 12] = cash / CASH_NORM
    feats[:, :, 13] = proj / PROJ_NORM
    feats[:, :, 14] = won
    feats[:, :, 15] = score / SCORE_NORM

    # Opponents sorted by score desc.
    opp_scores = score[:, OPP_IDX]        # (B,5,4)
    opp_cash = cash[:, OPP_IDX]           # (B,5,4)
    opp_proj = proj[:, OPP_IDX]           # (B,5,4)
    opp_won = won[:, OPP_IDX]             # (B,5,4)
    order = np.argsort(-opp_scores, axis=2, kind="stable")  # (B,5,4)

    def take(a):
        return np.take_along_axis(a, order, axis=2)

    s_cash = take(opp_cash)
    s_proj = take(opp_proj)
    s_won = take(opp_won)
    s_score = take(opp_scores)

    for j in range(4):
        base = 16 + 4 * j
        feats[:, :, base + 0] = s_cash[:, :, j] / CASH_NORM
        feats[:, :, base + 1] = s_proj[:, :, j] / PROJ_NORM
        feats[:, :, base + 2] = s_won[:, :, j]
        feats[:, :, base + 3] = s_score[:, :, j] / SCORE_NORM

    # Derived globals.
    feats[:, :, 32] = won.sum(axis=1, keepdims=True) / N_PLAYERS
    best_other = s_score[:, :, 0]                      # top opponent score per seat
    feats[:, :, 33] = (score >= best_other).astype(np.float64)  # leader if >= best opp
    feats[:, :, 34] = (opp_cash > cash[:, :, None]).sum(axis=2) / 4.0
    feats[:, :, 35] = (best_other - score) / SCORE_NORM

    return feats.astype(np.float32)


def features_single(state, me: int) -> np.ndarray:
    """Non-batched convenience wrapper for evaluation / debugging."""
    cash = state.cash[None, :]
    won = state.won_any[None, :].astype(np.float64)
    proj = state.proj_value[None, :]
    return features_batch(cash, won, proj, state.round)[0, me]
