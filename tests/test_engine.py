"""Engine correctness tests. Run with:  python -m pytest tests -q  (from repo root)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import (
    GameState, RETURNS, N_ROUNDS, N_PLAYERS, START_CASH, HAIRCUT,
    FINAL_CASH_MULT, apply_round, resolve_bids, validate_bid, max_bid_for,
)
from engine.runner import play_game, Agent, rank_of_scores
from engine.agents import RandomAgent, ValueAgent, StrategicAgent, make_population


def test_constants():
    assert RETURNS == (25, 50, 50, 100, 200, 350, 500, 750, 900, 1000)
    assert sum(RETURNS) == 3925
    assert N_ROUNDS == 10 and N_PLAYERS == 5 and START_CASH == 500


def test_max_bid_and_validate():
    assert max_bid_for(500.0) == 500
    assert max_bid_for(423.0) == 423
    assert validate_bid(100.0, 250) == 100      # cannot exceed cash
    assert validate_bid(100.0, -5) == 0         # no negative
    assert validate_bid(100.0, 30.4) == 30      # integer coercion


def test_resolve_no_tie():
    winner, bid, tie, log = resolve_bids({0: 10, 1: 5, 2: 7})
    assert winner == 0 and bid == 10 and not tie


def test_resolve_tie_resubmit():
    # players 0 and 1 tie at 10, then 1 outbids on resubmit
    calls = {"n": 0}

    def resubmit(tied):
        calls["n"] += 1
        return {0: 10, 1: 11}

    winner, bid, tie, log = resolve_bids({0: 10, 1: 10, 2: 3}, tie_resubmit_fn=resubmit)
    assert winner == 1 and bid == 11 and tie and calls["n"] == 1


def test_resolve_tie_random_fallback():
    rng = np.random.default_rng(0)
    winner, bid, tie, log = resolve_bids({0: 10, 1: 10}, rng=rng)  # no resubmit fn
    assert winner in (0, 1) and bid == 10 and tie


def test_apply_round_basic_and_qualification():
    s = GameState.new()
    # Round 1 (index 0): player 2 wins with 30.
    s = apply_round(s, winner=2, winning_bid=30, bids={0: 5, 1: 10, 2: 30})
    assert s.cash[2] == 470
    assert s.proj_value[2] == 25
    assert s.won_any[2] and not s.won_any[0]
    assert s.round == 1


def test_haircut_after_round_5():
    s = GameState.new()
    for r in range(5):  # rounds index 0..4
        s = apply_round(s, winner=0, winning_bid=0, bids={0: 0})
    # After round index 4 resolves, all cash x0.9
    assert np.allclose(s.cash, START_CASH * HAIRCUT)
    assert s.round == 5


def test_qualification_gate_only_rounds_1_6():
    s = GameState.new()
    # nobody wins rounds 1-6 except player 0 winning round 6 (index 5)
    for r in range(5):
        s = apply_round(s, winner=0, winning_bid=1, bids={0: 1})  # player 0 wins early anyway
    # player 0 qualified; check eligibility in round 7 (index 6)
    # fast-forward to round index 6
    s = apply_round(s, winner=0, winning_bid=1, bids={0: 1})  # round index 5
    assert s.round == 6
    elig = s.eligible()
    assert elig == [0]  # only player 0 won something in rounds 1-6


def test_final_score_formula():
    s = GameState.new()
    s.proj_value[0] = 1000
    s.cash[0] = 200
    # score = proj + 0.9*cash
    assert s.final_scores()[0] == 1000 + FINAL_CASH_MULT * 200


def test_full_game_runs_and_conserves_structure():
    agents = make_population(seed=1)
    scores, state = play_game(agents, rng=np.random.default_rng(1))
    assert state.done
    assert len(state.history) == N_ROUNDS
    assert scores.shape == (5,)
    # Every round has a winner among eligible (or -1 if unsold), and winning bid <= that
    # player's cash at the time is guaranteed by construction.
    ranks = rank_of_scores(scores)
    assert set(ranks.tolist()) <= set(range(5))


def test_cannot_bid_more_than_cash_over_game():
    # Play a game; verify no player's cash ever goes negative.
    class Greedy(Agent):
        def bid(self, state, me, is_resubmit=False, tied=None):
            return 10_000  # will be clamped
    agents = [Greedy() for _ in range(5)]
    scores, state = play_game(agents, rng=np.random.default_rng(3))
    assert np.all(state.cash >= -1e-9)


def test_unqualified_cannot_win_big_rounds():
    # Construct a game where only player 0 ever wins rounds 1-6.
    class Seat0Wins(Agent):
        def __init__(self, idx): self.idx = idx
        def bid(self, state, me, is_resubmit=False, tied=None):
            # Seat 0 wins every early round cheaply (bid 1 vs others' 0) so only it
            # qualifies, then keeps bidding in the big rounds.
            if me == 0:
                return 1
            return 0
    agents = [Seat0Wins(i) for i in range(5)]
    scores, state = play_game(agents, rng=np.random.default_rng(5))
    # In rounds 7-10 only player 0 is eligible, so player 0 wins all of them.
    big_winners = [h.winner for h in state.history if h.round >= 6]
    assert all(w in (0, -1) for w in big_winners)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([os.path.dirname(__file__), "-q"]))
