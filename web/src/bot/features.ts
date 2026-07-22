// Port of training/features.py :: features_single. MUST match exactly so the exported
// network weights behave identically in the browser. Parity is checked at runtime by
// bot/parity.ts against reference vectors dumped from Python.

import { GameState, RETURNS, N_ROUNDS, N_PLAYERS } from "../engine/game";

export const FEATURE_DIM = 36;
const CASH_NORM = 500;
const PROJ_NORM = 2000;
const RET_NORM = 1000;
const SCORE_NORM = 2000;
const FINAL_CASH_MULT = 0.9;

export function features(
  cash: number[],
  wonAny: boolean[] | number[],
  proj: number[],
  round: number,
  me: number
): number[] {
  const won: number[] = wonAny.map((w) => (w ? 1 : 0));
  const score = cash.map((c, i) => proj[i] + FINAL_CASH_MULT * c);

  const f = new Array(FEATURE_DIM).fill(0);
  f[round] = 1.0; // round one-hot [0:10]
  f[10] = RETURNS[round] / RET_NORM;
  f[11] = (N_ROUNDS - round) / 10.0;

  f[12] = cash[me] / CASH_NORM;
  f[13] = proj[me] / PROJ_NORM;
  f[14] = won[me];
  f[15] = score[me] / SCORE_NORM;

  // Opponents in ascending seat order, then stable-sorted by score desc.
  const opp: number[] = [];
  for (let p = 0; p < N_PLAYERS; p++) if (p !== me) opp.push(p);
  const ordered = opp
    .map((p, idx) => ({ p, idx, s: score[p] }))
    .sort((a, b) => (b.s !== a.s ? b.s - a.s : a.idx - b.idx))
    .map((o) => o.p);

  for (let j = 0; j < 4; j++) {
    const p = ordered[j];
    const base = 16 + 4 * j;
    f[base + 0] = cash[p] / CASH_NORM;
    f[base + 1] = proj[p] / PROJ_NORM;
    f[base + 2] = won[p];
    f[base + 3] = score[p] / SCORE_NORM;
  }

  const nQual = won.reduce((a, b) => a + b, 0);
  f[32] = nQual / N_PLAYERS;
  const bestOther = score[ordered[0]];
  f[33] = score[me] >= bestOther ? 1.0 : 0.0;
  let moreCash = 0;
  for (const p of opp) if (cash[p] > cash[me]) moreCash++;
  f[34] = moreCash / 4.0;
  f[35] = (bestOther - score[me]) / SCORE_NORM;

  return f;
}

export function featuresFromState(s: GameState, me: number): number[] {
  return features(s.cash, s.wonAny, s.projValue, s.round, me);
}
