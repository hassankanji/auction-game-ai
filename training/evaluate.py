"""Evaluation utilities: win-rate of the trained net vs reference opponents and vs itself."""

from __future__ import annotations

import numpy as np

from engine.runner import play_game
from engine.agents import RandomAgent, ValueAgent, StrategicAgent
from .net_agent import NetAgent


def _win_credit(scores: np.ndarray, seat: int) -> float:
    top = scores.max()
    winners = np.isclose(scores, top)
    if winners[seat]:
        return 1.0 / winners.sum()
    return 0.0


def winrate_vs(net, make_opponent, n_games: int = 400, seed: int = 0,
               greedy: bool = False) -> float:
    """Net plays one seat; the other four seats are filled by make_opponent().

    The net's seat is rotated across games for fairness. Returns average win credit
    (1/k for a k-way tie at the top), where 0.20 is the neutral 5-player baseline.
    """
    rng = np.random.default_rng(seed)
    credits = []
    for g in range(n_games):
        seat = g % 5
        agents = [make_opponent(rng) for _ in range(5)]
        agents[seat] = NetAgent(net, greedy=greedy, seed=int(rng.integers(1 << 30)))
        scores, _ = play_game(agents, rng=np.random.default_rng(int(rng.integers(1 << 30))))
        credits.append(_win_credit(scores, seat))
    return float(np.mean(credits))


def evaluate_suite(net, n_games: int = 300, seed: int = 0) -> dict:
    """Win-rate against several opponent archetypes. 0.20 == neutral baseline."""
    rng = lambda r: r
    out = {}
    out["vs_random"] = winrate_vs(
        net, lambda r: RandomAgent(seed=int(r.integers(1 << 30))), n_games, seed)
    out["vs_value"] = winrate_vs(
        net, lambda r: ValueAgent(shade=float(r.uniform(0.4, 0.7)),
                                  seed=int(r.integers(1 << 30))), n_games, seed + 1)
    out["vs_strategic"] = winrate_vs(
        net, lambda r: StrategicAgent(aggression=float(r.uniform(0.85, 1.15)),
                                      seed=int(r.integers(1 << 30))), n_games, seed + 2)

    def mixed(r):
        pick = r.integers(0, 3)
        if pick == 0:
            return RandomAgent(seed=int(r.integers(1 << 30)))
        if pick == 1:
            return ValueAgent(shade=float(r.uniform(0.4, 0.7)), seed=int(r.integers(1 << 30)))
        return StrategicAgent(aggression=float(r.uniform(0.85, 1.15)),
                              seed=int(r.integers(1 << 30)))
    out["vs_mixed"] = winrate_vs(net, mixed, n_games, seed + 3)
    return out
