"""
Action space + policy/value network for the self-play bot.

Action space: a fixed grid of bid fractions of current cash. Because every bid is
sealed and simultaneous, the strategically meaningful decision is a *distribution* over
how much of your budget to commit -- a discretised fraction grid matches this exactly and
keeps the action set fixed (state-independent), which the network handles cleanly. All
fractions are <= 1, so every action is always a legal bid (bid = floor(frac * floor(cash))).

The grid is denser near 0 so cheap qualification bids and shades are expressible.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_DIM

BID_FRACTIONS = np.array([
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
    0.15, 0.18, 0.22, 0.27, 0.32, 0.38, 0.45, 0.52, 0.60, 0.70,
    0.80, 0.90, 1.00,
], dtype=np.float64)
N_ACTIONS = len(BID_FRACTIONS)


def actions_to_bids(action_idx: np.ndarray, cash: np.ndarray) -> np.ndarray:
    """Map action indices -> integer bids given each seat's cash. Vectorised.

    action_idx, cash: same shape. Returns integer bids (floor), clipped to [0, floor(cash)].
    """
    frac = BID_FRACTIONS[action_idx]
    max_bid = np.floor(cash + 1e-9)
    bids = np.floor(frac * max_bid + 1e-9)
    bids = np.clip(bids, 0, max_bid)
    return bids.astype(np.int64)


class PolicyValueNet(nn.Module):
    """Small MLP with a categorical policy head over BID_FRACTIONS and a scalar value head."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 256, n_actions: int = N_ACTIONS,
                 depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Tanh()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        # Small init on heads for stable early training.
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, feats: np.ndarray, greedy: bool = False, temperature: float = 1.0):
        """feats: (N, FEATURE_DIM) -> (action_idx (N,), logp (N,), value (N,))."""
        x = torch.as_tensor(feats, dtype=torch.float32)
        logits, value = self.forward(x)
        if temperature != 1.0:
            logits = logits / temperature
        if greedy:
            a = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        logp = torch.log_softmax(logits, dim=-1).gather(-1, a[:, None]).squeeze(-1)
        return a.numpy(), logp.numpy(), value.numpy()


def export_weights(net: PolicyValueNet) -> dict:
    """Serialise the network to plain nested lists for the JS/TS in-browser bot."""
    sd = net.state_dict()
    out = {"format": "mlp-pv-v1", "in_dim": None, "hidden": None, "n_actions": N_ACTIONS,
           "bid_fractions": BID_FRACTIONS.tolist(), "layers": []}

    # Reconstruct the ordered layer list: body is [Linear, LayerNorm, Tanh] * depth.
    body = net.body
    for module in body:
        if isinstance(module, nn.Linear):
            out["layers"].append({
                "type": "linear",
                "weight": module.weight.detach().cpu().numpy().tolist(),
                "bias": module.bias.detach().cpu().numpy().tolist(),
            })
        elif isinstance(module, nn.LayerNorm):
            out["layers"].append({
                "type": "layernorm",
                "weight": module.weight.detach().cpu().numpy().tolist(),
                "bias": module.bias.detach().cpu().numpy().tolist(),
                "eps": module.eps,
            })
        elif isinstance(module, nn.Tanh):
            out["layers"].append({"type": "tanh"})
    out["pi"] = {"weight": net.pi.weight.detach().cpu().numpy().tolist(),
                 "bias": net.pi.bias.detach().cpu().numpy().tolist()}
    out["value"] = {"weight": net.v.weight.detach().cpu().numpy().tolist(),
                    "bias": net.v.bias.detach().cpu().numpy().tolist()}
    out["in_dim"] = net.body[0].in_features
    out["hidden"] = net.body[0].out_features
    return out
