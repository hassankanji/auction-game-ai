"""
Baseline / heuristic agents.

These serve three purposes:
  1. Opponents and sanity checks for the engine.
  2. A population of opponents to train the self-play RL bot against (robustness).
  3. Human-readable encodings of strategic ideas, used in the strategy writeup.

The flagship "top-difficulty" bot is the self-play-trained policy in ../training;
these heuristics are strong-but-explainable references.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .game import (
    GameState, RETURNS, N_ROUNDS, N_PLAYERS, QUALIFY_ROUND_IDX,
    HAIRCUT, FINAL_CASH_MULT, max_bid_for,
)
from .runner import Agent


class RandomAgent(Agent):
    """Bids a uniform random legal integer. Weak; useful as a floor."""
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def bid(self, state, me, is_resubmit=False, tied=None) -> int:
        mx = max_bid_for(state.cash[me])
        # Bias toward small bids so games are not degenerate.
        if mx <= 0:
            return 0
        return int(self.rng.integers(0, max(1, min(mx, RETURNS[state.round]) + 1)))


class ValueAgent(Agent):
    """
    Simple first-price bidder: bids a fixed fraction ('shade') of a project's
    return, plus a qualification top-up when at risk of being locked out of
    rounds 7-10. Ignores opponents' exact cash. A reasonable intermediate bot.
    """
    name = "value"

    def __init__(self, shade: float = 0.55, qual_shade: float = 0.9, seed: Optional[int] = None):
        self.shade = shade
        self.qual_shade = qual_shade
        self.rng = np.random.default_rng(seed)

    def bid(self, state: GameState, me, is_resubmit=False, tied=None) -> int:
        r = state.round
        ret = RETURNS[r]
        cash = state.cash[me]
        mx = max_bid_for(cash)
        if mx <= 0:
            return 0

        base = self.shade * ret

        # Qualification urgency: if not qualified and rounds 1-6 are running out,
        # be willing to pay more for a cheap early win.
        if r < QUALIFY_ROUND_IDX and not state.won_any[me]:
            rounds_left_to_qualify = QUALIFY_ROUND_IDX - r
            urgency = 1.0 / max(1, rounds_left_to_qualify)
            base = max(base, self.qual_shade * ret * (0.5 + 0.5 * urgency))

        bid = int(round(base))
        if is_resubmit:
            bid = min(mx, bid + int(self.rng.integers(1, 5)))  # nudge up to break tie
        return max(0, min(mx, bid))


class StrategicAgent(Agent):
    """
    Strong explainable heuristic. Core ideas:

      * Cash-to-score exchange rate. A dollar kept to the end scores 0.9 (post round-5
        haircut) or 0.81 (if held from before round 6). So a project is "worth winning"
        only if its return exceeds the end-value of the cash spent -- BUT cash also has
        option value (it lets you win bigger future projects), captured via a simple
        budget plan for the rounds you can realistically contest.

      * Qualification. Access to rounds 7-10 (80% of all returns) is gated on winning one
        of rounds 1-6. The agent tracks its risk of missing qualification and pays a
        premium for a cheap early win when the risk is real.

      * Opponent-cash awareness. Because winning bids are public, every player's cash is
        known exactly. The agent bids to beat the strongest *plausible* competitor for the
        projects it targets, and refuses to overpay when it holds a decisive budget lead.

    It targets a subset of the ten projects (a "plan") that maximises expected end score
    given its budget, and concentrates spending there.
    """
    name = "strategic"

    def __init__(self, aggression: float = 1.0, seed: Optional[int] = None):
        self.aggression = aggression
        self.rng = np.random.default_rng(seed)

    # ---- helpers -------------------------------------------------------------

    def _cash_end_value(self, state: GameState) -> float:
        """End-score value of $1 of current cash held to the end."""
        if state.round <= 4:
            return HAIRCUT * FINAL_CASH_MULT   # 0.81, will be haircut once more after R5
        return FINAL_CASH_MULT                 # 0.9

    def _future_target_value(self, state: GameState, me: int) -> float:
        """
        Rough option value of cash: the extra score obtainable from future rounds we can
        realistically win, per dollar. Approximated by the best return-per-dollar among
        remaining rounds we are eligible for, scaled by how contested they are.
        """
        r = state.round
        cash = state.cash[me]
        if cash <= 0:
            return self._cash_end_value(state)
        best = self._cash_end_value(state)
        for rr in range(r + 1, N_ROUNDS):
            eligible_future = (rr < QUALIFY_ROUND_IDX) or state.won_any[me]
            if not eligible_future:
                continue
            # crude price we'd expect to pay ~ 0.5 * return; value per dollar if we win
            approx_price = max(1.0, 0.5 * RETURNS[rr])
            per_dollar = RETURNS[rr] / approx_price
            best = max(best, per_dollar)
        return best

    def _strongest_rival_budget(self, state: GameState, me: int) -> float:
        r = state.round
        rivals = []
        for p in range(N_PLAYERS):
            if p == me:
                continue
            if r >= QUALIFY_ROUND_IDX and not state.won_any[p]:
                continue  # can't bid in the big rounds
            rivals.append(state.cash[p])
        return max(rivals) if rivals else 0.0

    # ---- policy --------------------------------------------------------------

    def bid(self, state: GameState, me: int, is_resubmit=False, tied=None) -> int:
        r = state.round
        ret = RETURNS[r]
        cash = float(state.cash[me])
        mx = max_bid_for(cash)
        if mx <= 0:
            return 0

        # Value of winning THIS project, in end-score units, is just `ret` (permanent).
        # Value of the cash we'd spend, per dollar, is the max of hoarding value and the
        # option value of deploying it on a better future round.
        mvc = self._future_target_value(state, me)  # marginal value of cash, >= hoarding
        # We are willing to pay up to ret / mvc dollars (indifference), then shade for the
        # first-price auction.
        indiff = ret / max(mvc, 1e-6)

        # ---- Qualification logic (rounds 1-6) --------------------------------
        must_qualify_premium = 0.0
        if r < QUALIFY_ROUND_IDX and not state.won_any[me]:
            rounds_left = QUALIFY_ROUND_IDX - r
            # Expected value of qualifying: access to rounds 7-10. Conservatively value it
            # at a fraction of the average big-round surplus we could capture.
            qual_ev = 0.18 * sum(RETURNS[QUALIFY_ROUND_IDX:])  # tunable estimate of surplus
            # Spread that willingness across the remaining cheap rounds, front-loaded as
            # the window closes.
            urgency = 1.0 / rounds_left
            must_qualify_premium = qual_ev * urgency * 0.15

        target = indiff + must_qualify_premium

        # ---- First-price shading against the strongest plausible rival -------
        rival_budget = self._strongest_rival_budget(state, me)
        # If we massively out-budget everyone in a big round, we don't need to pay near
        # our value -- just enough to beat what a rational rival would pay.
        # A rival's willingness is bounded by their cash and by the same indifference logic.
        rival_willing = min(rival_budget, ret / max(FINAL_CASH_MULT, 1e-6))

        # Bid a shade above what beats the strongest rival, but never above our target.
        beat = min(rival_willing + 1, target)
        shade = 0.85 * self.aggression
        bid = max(shade * target, beat)

        # Don't spend so much we can't contest a clearly better upcoming round.
        bid = min(bid, target, cash)

        bid = int(round(bid))
        if is_resubmit:
            # In a tie resubmit, push up toward our true ceiling.
            ceiling = int(round(min(target, cash)))
            bump = int(self.rng.integers(1, 6))
            bid = min(mx, max(bid + bump, min(ceiling, mx)))
        return max(0, min(mx, bid))


def make_population(seed: int = 0) -> list[Agent]:
    """A mixed population of reference opponents for evaluation/training."""
    rng = np.random.default_rng(seed)
    return [
        RandomAgent(seed=int(rng.integers(1 << 30))),
        ValueAgent(shade=0.45, seed=int(rng.integers(1 << 30))),
        ValueAgent(shade=0.6, seed=int(rng.integers(1 << 30))),
        StrategicAgent(aggression=0.9, seed=int(rng.integers(1 << 30))),
        StrategicAgent(aggression=1.1, seed=int(rng.integers(1 << 30))),
    ]
