// Bot decision layer: turn network output into an integer bid, and expose the full
// bid distribution for the advisor. The trained self-play policy is a *mixed* strategy
// (a distribution over how much budget to commit); sampling from it is the
// game-theoretically correct play. Difficulty tweaks the sampling temperature.

import { GameState, maxBidFor } from "../engine/game";
import { PolicyNet } from "./net";
import { featuresFromState } from "./features";

export type Difficulty = "easy" | "medium" | "hard" | "expert";

// In a SEALED-bid simultaneous game the hardest-to-exploit play is the trained mixed
// strategy at its natural temperature — sharpening it (temp < 1) makes the bot predictable
// and easy to out-bid by +1. So "expert" plays the full-strength trained policy (temp 1.0)
// and the easier levels get progressively softer/noisier (weaker).
const TEMP: Record<Difficulty, number> = {
  easy: 3.0,
  medium: 1.7,
  hard: 1.15,
  expert: 1.0,
};
// Easy/medium blend in uniform noise to be beatable; expert stays full-strength.
const NOISE: Record<Difficulty, number> = {
  easy: 0.4,
  medium: 0.15,
  hard: 0.04,
  expert: 0.0,
};

function applyTemp(probs: number[], temp: number, noise: number): number[] {
  let p = probs.map((v) => Math.pow(Math.max(v, 1e-12), 1 / temp));
  const s = p.reduce((a, b) => a + b, 0);
  p = p.map((v) => v / s);
  if (noise > 0) p = p.map((v) => (1 - noise) * v + noise / p.length);
  return p;
}

export function fracToBid(frac: number, cash: number): number {
  const mx = maxBidFor(cash);
  return Math.max(0, Math.min(mx, Math.floor(frac * mx + 1e-9)));
}

function sample(probs: number[], rnd: () => number): number {
  const u = rnd();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return probs.length - 1;
}

export interface BidBreakdownRow {
  frac: number;
  bid: number;
  prob: number;
}

export interface BotDecision {
  bid: number;
  actionIdx: number;
  value: number; // net's estimate of this seat's expected reward (win-share based)
  distribution: BidBreakdownRow[]; // merged by integer bid, sorted by prob desc
  recommended: number; // argmax bid (the single best point recommendation)
}

// Full analysis of a seat's decision: used both to make a bot move and to advise the human.
export function analyzeDecision(
  net: PolicyNet,
  state: GameState,
  me: number,
  difficulty: Difficulty = "hard"
): BotDecision {
  const feat = featuresFromState(state, me);
  const { probs, value } = net.evaluate(feat);
  const cash = state.cash[me];
  const tempered = applyTemp(probs, TEMP[difficulty], NOISE[difficulty]);

  // Merge fractions that map to the same integer bid.
  const merged = new Map<number, number>();
  net.bidFractions.forEach((frac, i) => {
    const bid = fracToBid(frac, cash);
    merged.set(bid, (merged.get(bid) ?? 0) + tempered[i]);
  });
  const distribution: BidBreakdownRow[] = [...merged.entries()]
    .map(([bid, prob]) => ({ bid, prob, frac: bid / Math.max(1, maxBidFor(cash)) }))
    .sort((a, b) => b.prob - a.prob);

  const recommended = distribution[0].bid;
  return { bid: recommended, actionIdx: -1, value, distribution, recommended };
}

// Small off-grid jitter so bots don't all land on the same grid value (which a human could
// snipe by bidding grid+1). Slightly upward-biased: raising a winning bid a touch is cheap
// insurance, and it approximates the continuous spread of the true mixed strategy.
function jitter(bid: number, cash: number, rnd: () => number): number {
  if (bid <= 0) return 0;
  const j = Math.min(6, Math.max(1, Math.round(0.05 * bid)));
  const delta = Math.round((rnd() * 2 - 0.8) * j); // range ~ [-0.8j, +1.2j]
  return Math.max(0, Math.min(maxBidFor(cash), bid + delta));
}

// Bot move: sample from the (tempered) policy for equilibrium-style mixed play.
export function botBid(
  net: PolicyNet,
  state: GameState,
  me: number,
  difficulty: Difficulty,
  rnd: () => number = Math.random
): number {
  const feat = featuresFromState(state, me);
  const { probs } = net.evaluate(feat);
  const cash = state.cash[me];
  const tempered = applyTemp(probs, TEMP[difficulty], NOISE[difficulty]);
  const idx = sample(tempered, rnd);
  const base = fracToBid(net.bidFractions[idx], cash);
  // Expert/hard get jitter for un-snipeability; easy/medium already noisy enough.
  return difficulty === "expert" || difficulty === "hard"
    ? jitter(base, cash, rnd)
    : base;
}

// Bot move on a tie resubmit: nudge upward toward its ceiling to try to break the tie.
export function botResubmit(
  net: PolicyNet,
  state: GameState,
  me: number,
  difficulty: Difficulty,
  rnd: () => number = Math.random
): number {
  const base = botBid(net, state, me, difficulty, rnd);
  const mx = maxBidFor(state.cash[me]);
  const bump = 1 + Math.floor(rnd() * Math.max(2, Math.floor(mx / 50)));
  return Math.min(mx, base + bump);
}
