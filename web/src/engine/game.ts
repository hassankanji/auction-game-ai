// TypeScript port of engine/game.py. MUST stay in lock-step with the Python engine
// (rules, haircut timing, qualification gate, scoring). See ../../../engine/game.py.

export const RETURNS = [25, 50, 50, 100, 200, 350, 500, 750, 900, 1000] as const;
export const N_ROUNDS = RETURNS.length;
export const N_PLAYERS = 5;
export const START_CASH = 500;
export const HAIRCUT = 0.9; // cash x0.9 after round 5
export const HAIRCUT_AFTER_ROUND_IDX = 4;
export const FINAL_CASH_MULT = 0.9; // 90% of remaining cash counts at scoring
export const QUALIFY_ROUND_IDX = 6; // rounds with index >= 6 require qualification

export interface RoundResult {
  round: number;
  returnValue: number;
  winner: number; // -1 if unsold
  winningBid: number;
  bids: Record<number, number>; // original sealed bids by eligible seat
  tieBroken: boolean;
}

export interface GameState {
  cash: number[];
  wonAny: boolean[];
  projValue: number[];
  round: number; // index of the NEXT round to play (0..10)
  history: RoundResult[];
}

export function newGame(): GameState {
  return {
    cash: Array(N_PLAYERS).fill(START_CASH),
    wonAny: Array(N_PLAYERS).fill(false),
    projValue: Array(N_PLAYERS).fill(0),
    round: 0,
    history: [],
  };
}

export function cloneState(s: GameState): GameState {
  return {
    cash: [...s.cash],
    wonAny: [...s.wonAny],
    projValue: [...s.projValue],
    round: s.round,
    history: s.history.map((h) => ({ ...h, bids: { ...h.bids } })),
  };
}

export const isDone = (s: GameState) => s.round >= N_ROUNDS;

export function maxBidFor(cash: number): number {
  return Math.floor(cash + 1e-9);
}

export function eligible(s: GameState): number[] {
  if (isDone(s)) return [];
  if (s.round < QUALIFY_ROUND_IDX) return [0, 1, 2, 3, 4];
  return [0, 1, 2, 3, 4].filter((p) => s.wonAny[p]);
}

export function validateBid(cash: number, bid: number): number {
  let b = Math.round(bid);
  if (!isFinite(b) || b < 0) b = 0;
  const mx = maxBidFor(cash);
  if (b > mx) b = mx;
  return b;
}

// Provisional / final score = projects won + 0.9 * cash.
export function scores(s: GameState): number[] {
  return s.cash.map((c, i) => s.projValue[i] + FINAL_CASH_MULT * c);
}

// Current score used for ranking during the game (same formula).
export const provisionalScores = scores;

export interface RankInfo {
  rank: number; // 0 = best
  score: number;
}
export function ranks(s: GameState): RankInfo[] {
  const sc = scores(s);
  const order = sc.map((v, i) => i).sort((a, b) => sc[b] - sc[a]);
  const out: RankInfo[] = sc.map((v) => ({ rank: 0, score: v }));
  order.forEach((seat, r) => (out[seat].rank = r));
  return out;
}

// Apply a resolved round outcome, returning a new state (immutable).
export function applyRound(
  s: GameState,
  winner: number,
  winningBid: number,
  bids: Record<number, number>,
  tieBroken = false
): GameState {
  const ns = cloneState(s);
  const r = s.round;
  const ret = RETURNS[r];
  if (winner >= 0) {
    ns.cash[winner] -= winningBid;
    ns.projValue[winner] += ret;
    if (r < QUALIFY_ROUND_IDX) ns.wonAny[winner] = true;
  }
  ns.history.push({
    round: r,
    returnValue: ret,
    winner,
    winningBid: winner >= 0 ? winningBid : 0,
    bids: { ...bids },
    tieBroken,
  });
  ns.round = r + 1;
  if (r === HAIRCUT_AFTER_ROUND_IDX) ns.cash = ns.cash.map((c) => c * HAIRCUT);
  return ns;
}

export interface Resolution {
  winner: number;
  winningBid: number;
  tied: number[]; // seats tied at the top (length > 1 means a resubmit is needed)
}

// Determine the top bid and who is tied there. Caller handles resubmission.
export function topOfBids(bids: Record<number, number>): Resolution {
  const seats = Object.keys(bids).map(Number);
  if (seats.length === 0) return { winner: -1, winningBid: 0, tied: [] };
  let top = -Infinity;
  for (const p of seats) top = Math.max(top, bids[p]);
  const tied = seats.filter((p) => bids[p] === top);
  return { winner: tied.length === 1 ? tied[0] : -1, winningBid: top, tied };
}

// Total returns in the game (for progress bars etc.).
export const TOTAL_RETURNS = RETURNS.reduce((a, b) => a + b, 0);
