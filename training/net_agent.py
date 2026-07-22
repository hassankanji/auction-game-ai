"""Wrap a trained PolicyValueNet as an engine Agent (for evaluation and human play)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from engine.runner import Agent
from engine.game import max_bid_for
from .features import features_single
from .policy import BID_FRACTIONS


class NetAgent(Agent):
    name = "neural"

    def __init__(self, net, greedy: bool = False, temperature: float = 1.0,
                 seed: Optional[int] = None):
        self.net = net
        self.greedy = greedy
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def bid(self, state, me, is_resubmit=False, tied=None) -> int:
        mx = max_bid_for(state.cash[me])
        if mx <= 0:
            return 0
        feats = features_single(state, me)[None, :]
        a, _, _ = self.net.act(feats, greedy=self.greedy, temperature=self.temperature)
        frac = BID_FRACTIONS[int(a[0])]
        bid = int(np.floor(frac * mx + 1e-9))
        bid = max(0, min(mx, bid))
        if is_resubmit:
            bid = min(mx, bid + int(self.rng.integers(1, max(2, mx // 50 + 2))))
        return bid
