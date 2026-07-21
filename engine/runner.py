"""
Game runner: drives 5 agents through a full game, including tie resubmissions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .game import (
    GameState, RoundResult, N_PLAYERS,
    resolve_bids, apply_round, validate_bid,
)


class Agent:
    """
    Base agent interface. An agent sees the full public state and its own index.

    bid() is called once per round for every eligible player (simultaneously in spirit;
    the runner collects all bids before revealing anything). On a tie for the top bid,
    only tied players are asked again with is_resubmit=True and `tied` set.
    """
    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of each game."""
        pass

    def bid(self, state: GameState, me: int, is_resubmit: bool = False,
            tied: Optional[list[int]] = None) -> int:
        raise NotImplementedError


def play_game(agents: list[Agent], rng: Optional[np.random.Generator] = None,
              seat_permutation: Optional[list[int]] = None) -> tuple[np.ndarray, GameState]:
    """
    Play one full game with the given 5 agents (agents[i] sits in seat i).

    Returns (final_scores, final_state). final_scores[i] is the score for seat i.
    `seat_permutation` is unused here but reserved; shuffle the agents list yourself to
    randomise seating.
    """
    assert len(agents) == N_PLAYERS, "need exactly 5 agents"
    if rng is None:
        rng = np.random.default_rng()

    for a in agents:
        a.reset()

    state = GameState.new()

    while not state.done:
        elig = state.eligible()

        # Collect sealed bids from all eligible players.
        bids: dict[int, int] = {}
        for p in elig:
            raw = agents[p].bid(state, p, is_resubmit=False, tied=None)
            bids[p] = validate_bid(state.cash[p], raw)

        if not bids:
            # No eligible bidders (can happen in rounds 7-10 if nobody qualified):
            # project goes unsold; advance the round with no winner.
            r = state.round
            ns = state.copy()
            ns.history.append(RoundResult(
                round=r, return_value=state_return(state), winner=-1,
                winning_bid=0, bids={}, tie_broken=False,
            ))
            ns.round = r + 1
            from .game import HAIRCUT_AFTER_ROUND_IDX, HAIRCUT
            if r == HAIRCUT_AFTER_ROUND_IDX:
                ns.cash = ns.cash * HAIRCUT
            state = ns
            continue

        def tie_resubmit_fn(tied_players: list[int]) -> dict[int, int]:
            out = {}
            for p in tied_players:
                raw = agents[p].bid(state, p, is_resubmit=True, tied=list(tied_players))
                out[p] = validate_bid(state.cash[p], raw)
            return out

        winner, winning_bid, tie_broken, resub_log = resolve_bids(
            bids, rng=rng, tie_resubmit_fn=tie_resubmit_fn,
        )
        state = apply_round(state, winner, winning_bid, bids,
                            tie_broken=tie_broken, resubmit_log=resub_log)

    return state.final_scores(), state


def state_return(state: GameState) -> int:
    from .game import RETURNS
    return RETURNS[state.round]


def rank_of_scores(scores: np.ndarray) -> np.ndarray:
    """Return 0-based rank of each seat (0 = winner). Ties share the better rank."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    for rank, seat in enumerate(order):
        ranks[seat] = rank
    return ranks
