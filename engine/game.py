"""
Core game engine for the 5-Player Global Tournament Auction Game.

Rules (from the official brief):
  - 5 investors, each starts with $500m.
  - 10 rounds. Each round auctions ONE renewable-energy project with a fixed return.
  - Bids are sealed, simultaneous, and integer. You may bid 0 up to your remaining cash.
    You cannot borrow (bid <= floor(cash)).
  - Highest bidder wins the project, pays their bid, and gains the project's return.
  - Ties for the highest bid: only the tied players resubmit, repeatedly, until broken.
  - Qualification: to bid in rounds 7-10 you must have won at least one of rounds 1-6.
  - After round 5, every player's remaining cash is reduced by 10% (x0.9).
  - At scoring, remaining cash is reduced by a further 10% (x0.9).
  - Final score = value of projects won + 0.9 * remaining cash.
  - Highest final score wins.

Important structural fact that the engine exposes to agents:
  Money is spent ONLY by winning, and winning bids are announced. Therefore every
  player's cash is *common knowledge* (fully derivable from public history). The only
  hidden information is the current round's simultaneous bids. Agents here receive the
  full public state; they do NOT receive opponents' current-round bids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

N_PLAYERS: int = 5
START_CASH: float = 500.0
RETURNS: tuple[int, ...] = (25, 50, 50, 100, 200, 350, 500, 750, 900, 1000)
N_ROUNDS: int = len(RETURNS)

# Round indices are 0-based internally (round 1 in the brief == index 0).
HAIRCUT_AFTER_ROUND_IDX: int = 4   # cash x0.9 applied AFTER resolving index 4 (round 5)
HAIRCUT: float = 0.9
FINAL_CASH_MULT: float = 0.9       # 90% of remaining cash counts at scoring
QUALIFY_ROUND_IDX: int = 6         # rounds with index >= 6 (rounds 7-10) require qualification


def max_bid_for(cash: float) -> int:
    """Largest legal integer bid for a given cash balance (cannot borrow)."""
    return int(math.floor(cash + 1e-9))


@dataclass
class GameState:
    """Full public state of a game in progress."""
    cash: np.ndarray                       # shape (5,), float
    won_any: np.ndarray                    # shape (5,), bool  -> won at least one of rounds 1-6
    proj_value: np.ndarray                 # shape (5,), float -> sum of returns of projects won
    round: int = 0                         # index of the NEXT round to play (0..10)
    history: list = field(default_factory=list)  # list of RoundResult

    @staticmethod
    def new() -> "GameState":
        return GameState(
            cash=np.full(N_PLAYERS, START_CASH, dtype=float),
            won_any=np.zeros(N_PLAYERS, dtype=bool),
            proj_value=np.zeros(N_PLAYERS, dtype=float),
            round=0,
            history=[],
        )

    @property
    def done(self) -> bool:
        return self.round >= N_ROUNDS

    def eligible(self) -> list[int]:
        """Players allowed to bid in the current round."""
        if self.done:
            return []
        if self.round < QUALIFY_ROUND_IDX:
            return list(range(N_PLAYERS))          # everyone may bid in rounds 1-6
        return [p for p in range(N_PLAYERS) if self.won_any[p]]  # rounds 7-10: qualified only

    def copy(self) -> "GameState":
        return GameState(
            cash=self.cash.copy(),
            won_any=self.won_any.copy(),
            proj_value=self.proj_value.copy(),
            round=self.round,
            history=list(self.history),
        )

    def final_scores(self) -> np.ndarray:
        """Final scores assuming the game is over (or scoring current cash as-is)."""
        return self.proj_value + FINAL_CASH_MULT * self.cash


@dataclass
class RoundResult:
    round: int
    return_value: int
    winner: int
    winning_bid: int
    bids: dict            # {player: bid} for eligible players (the ORIGINAL sealed bids)
    tie_broken: bool = False
    tie_resolution_bids: list = field(default_factory=list)  # list of {player: bid} per resubmit


def resolve_bids(bids: dict[int, int], rng: Optional[np.random.Generator] = None,
                 tie_resubmit_fn=None, max_resubmits: int = 8):
    """
    Given a dict {player: integer_bid} for the eligible players, determine the winner.

    Tie handling follows the brief: only the tied top bidders resubmit. `tie_resubmit_fn`,
    if provided, is called as tie_resubmit_fn(tied_players) -> {player: new_bid} to gather
    resubmitted bids. If not provided (or ties persist past max_resubmits), the tie is
    broken uniformly at random.

    Returns (winner, winning_bid, tie_broken, resubmit_log).
    """
    if rng is None:
        rng = np.random.default_rng()
    if not bids:
        raise ValueError("resolve_bids called with no eligible bidders")

    current = dict(bids)
    resubmit_log: list = []
    tie_broken = False

    for attempt in range(max_resubmits + 1):
        top = max(current.values())
        winners = [p for p, b in current.items() if b == top]
        if len(winners) == 1:
            return winners[0], top, tie_broken, resubmit_log
        # Tie among `winners`.
        tie_broken = True
        if tie_resubmit_fn is not None and attempt < max_resubmits:
            new_bids = tie_resubmit_fn(list(winners))
            # Only tied players participate; ignore anyone else.
            current = {p: int(new_bids[p]) for p in winners}
            resubmit_log.append(dict(current))
        else:
            winner = int(rng.choice(winners))
            return winner, top, tie_broken, resubmit_log

    # Fallback (should be unreachable): random among final tied set.
    top = max(current.values())
    winners = [p for p, b in current.items() if b == top]
    return int(rng.choice(winners)), top, True, resubmit_log


def apply_round(state: GameState, winner: int, winning_bid: int, bids: dict,
                tie_broken: bool = False, resubmit_log: Optional[list] = None) -> GameState:
    """Return a NEW state with the given round outcome applied (immutably)."""
    if state.done:
        raise ValueError("Game already finished")
    r = state.round
    ret = RETURNS[r]
    ns = state.copy()

    ns.cash[winner] -= winning_bid
    ns.proj_value[winner] += ret
    if r < QUALIFY_ROUND_IDX:
        ns.won_any[winner] = True

    ns.history.append(RoundResult(
        round=r, return_value=ret, winner=winner, winning_bid=winning_bid,
        bids=dict(bids), tie_broken=tie_broken,
        tie_resolution_bids=list(resubmit_log or []),
    ))
    ns.round = r + 1

    # Cash haircut after round 5 (index 4).
    if r == HAIRCUT_AFTER_ROUND_IDX:
        ns.cash = ns.cash * HAIRCUT

    return ns


def validate_bid(cash: float, bid) -> int:
    """Coerce/validate a single bid to a legal integer in [0, floor(cash)]."""
    if bid is None:
        bid = 0
    b = int(round(float(bid)))
    if b < 0:
        b = 0
    mx = max_bid_for(cash)
    if b > mx:
        b = mx
    return b
