import { GameState, RETURNS, N_ROUNDS, QUALIFY_ROUND_IDX, scores, maxBidFor } from "../engine/game";
import { BidBreakdownRow } from "../bot/bot";

export const SEAT_NAMES = ["You", "Ada", "Boris", "Chen", "Dara"];
export const SEAT_COLORS = ["#56b3ff", "#ff9f6b", "#7c5cff", "#3ad29f", "#ff6b81"];

export const fmt = (v: number) => "$" + Math.round(v).toLocaleString() + "m";
export const fmt1 = (v: number) =>
  "$" + (Math.round(v * 10) / 10).toLocaleString(undefined, { minimumFractionDigits: 0 }) + "m";

export function RoundPips({ state }: { state: GameState }) {
  return (
    <div className="pips">
      {RETURNS.map((r, i) => {
        const cls =
          i < state.round ? "pip done" : i === state.round ? "pip cur" : "pip";
        const big = i >= QUALIFY_ROUND_IDX ? " big" : "";
        return (
          <div key={i} className={cls + big} title={`Round ${i + 1}: ${fmt(r)}`}>
            {i + 1}
          </div>
        );
      })}
    </div>
  );
}

export function Prize({ round }: { round: number }) {
  const done = round >= N_ROUNDS;
  return (
    <div className="prize">
      <div className="ret">{done ? "—" : fmt(RETURNS[round])}</div>
      <div className="lbl">{done ? "game over" : `Round ${round + 1} prize`}</div>
    </div>
  );
}

export function PlayersTable({
  state,
  meSeat = 0,
  showScores = true,
}: {
  state: GameState;
  meSeat?: number;
  showScores?: boolean;
}) {
  const sc = scores(state);
  const maxScore = Math.max(...sc);
  const leader = sc.indexOf(maxScore);
  const qualifyPhase = state.round >= QUALIFY_ROUND_IDX;
  return (
    <div className="players">
      {SEAT_NAMES.map((name, i) => {
        const qualified = state.wonAny[i];
        const barPct = Math.max(2, (state.cash[i] / 500) * 100);
        const cls = `pl${i === meSeat ? " me" : ""}${i === leader ? " lead" : ""}`;
        return (
          <div className={cls} key={i}>
            <div className="dot" style={{ background: SEAT_COLORS[i] }} />
            <div className="nm">
              {name}
              {i === leader && showScores ? <span className="chip" style={{ color: "var(--gold)", borderColor: "rgba(255,209,102,.5)" }}>lead</span> : null}
              {state.round > 0 || qualifyPhase ? (
                <span className={`chip ${qualified ? "q" : "nq"}`}>
                  {qualified ? "qualified" : qualifyPhase ? "locked out" : "not yet"}
                </span>
              ) : null}
            </div>
            <div className="col-cash">
              <div className="small muted">cash</div>
              <div className="mono" style={{ fontWeight: 700 }}>{fmt1(state.cash[i])}</div>
              <div className="bar"><i style={{ width: `${barPct}%` }} /></div>
            </div>
            <div className="col-proj">
              <div className="small muted">projects won</div>
              <div className="mono" style={{ fontWeight: 700 }}>{fmt(state.projValue[i])}</div>
            </div>
            {showScores ? (
              <div className="scoreval">
                <div className="small muted">score</div>
                {fmt1(sc[i])}
              </div>
            ) : (
              <div />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function BidDistribution({
  rows,
  recommended,
  cash,
  maxRows = 8,
}: {
  rows: BidBreakdownRow[];
  recommended: number;
  cash: number;
  maxRows?: number;
}) {
  const top = rows.slice(0, maxRows).filter((r) => r.prob > 0.005);
  const maxP = Math.max(...top.map((r) => r.prob), 0.01);
  return (
    <div className="dist">
      {top.map((r) => (
        <div className={`drow ${r.bid === recommended ? "best" : ""}`} key={r.bid}>
          <div className="mono">{fmt(r.bid)}</div>
          <div className="dbar"><i style={{ width: `${(r.prob / maxP) * 100}%` }} /></div>
          <div className="mono muted">{(r.prob * 100).toFixed(0)}%</div>
        </div>
      ))}
      <div className="tiny muted">
        Highlighted = single best bid. The bot mixes across these (a Nash-style mixed
        strategy), so it's unpredictable. Max legal bid: {fmt(maxBidFor(cash))}.
      </div>
    </div>
  );
}
