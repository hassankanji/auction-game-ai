import { useMemo, useState } from "react";
import { PolicyNet } from "../bot/net";
import {
  GameState, newGame, applyRound, eligible, isDone, maxBidFor, scores,
  RETURNS, QUALIFY_ROUND_IDX, N_ROUNDS, N_PLAYERS, TOTAL_RETURNS,
} from "../engine/game";
import { analyzeDecision } from "../bot/bot";
import { SEAT_COLORS, fmt, fmt1, RoundPips, Prize, PlayersTable, BidDistribution } from "./common";

interface Outcome { winner: number; winningBid: number }

// Rebuild the exact public state from the recorded round outcomes. Cash is derivable
// because money is only ever spent by winning, and winning bids are public.
function rebuild(outcomes: Outcome[]): GameState {
  let s = newGame();
  for (const o of outcomes) {
    if (isDone(s)) break;
    s = applyRound(s, o.winner, o.winningBid, { [o.winner]: o.winningBid }, false);
  }
  return s;
}

function rationale(state: GameState, mySeat: number, rec: number): string[] {
  const r = state.round;
  const sc = scores(state);
  const myCash = state.cash[mySeat];
  const notes: string[] = [];
  const bigLeft = RETURNS.slice(Math.max(r, QUALIFY_ROUND_IDX)).reduce((a, b) => a + b, 0);

  if (r < QUALIFY_ROUND_IDX && !state.wonAny[mySeat]) {
    const roundsLeft = QUALIFY_ROUND_IDX - r;
    notes.push(
      `You are NOT yet qualified. You must win one of the next ${roundsLeft} round(s) to unlock rounds 7–10 (worth ${fmt(bigLeft)} in total). A cheap win here is worth paying up for.`
    );
  } else if (r < QUALIFY_ROUND_IDX && state.wonAny[mySeat]) {
    notes.push(`You're already qualified — you don't need this round. Only take it if the price is right; otherwise preserve cash for the big rounds.`);
  }

  // Budget standing among eligible players.
  const rivalsCash = state.cash.filter((_, i) => i !== mySeat && (r < QUALIFY_ROUND_IDX || state.wonAny[i]));
  const maxRival = rivalsCash.length ? Math.max(...rivalsCash) : 0;
  if (myCash > maxRival + 0.5) {
    notes.push(`You hold the largest budget among live bidders (${fmt1(myCash)} vs ${fmt1(maxRival)}). You can credibly win the most valuable remaining projects — don't overpay early.`);
  } else if (myCash >= maxRival - 0.5) {
    notes.push(`Your budget (${fmt1(myCash)}) is level with the top rival (${fmt1(maxRival)}). Whoever preserves cash into the big rounds wins them — don't overpay unless it secures qualification.`);
  } else {
    notes.push(`Your budget (${fmt1(myCash)}) trails the top rival (${fmt1(maxRival)}). Pick your spots; don't get into a bidding war you can't win on the biggest projects.`);
  }

  if (r >= QUALIFY_ROUND_IDX) {
    const nQual = state.wonAny.filter(Boolean).length;
    notes.push(`Big round: ${fmt(RETURNS[r])} project, ${nQual} qualified bidders. Cash you keep is worth only 0.9× at scoring, so it's worth paying close to ${fmt(rec)} to win when you can.`);
  } else if (r === HAIRCUT_HINT_ROUND) {
    notes.push(`Reminder: after round 5 all remaining cash is cut 10%. Cash held from here to the end is worth ~0.81×, so hoarding is expensive.`);
  }

  const myRank = sc.map((v, i) => i).sort((a, b) => sc[b] - sc[a]).indexOf(mySeat) + 1;
  notes.push(`Current standing: #${myRank} of 5 (score ${fmt1(sc[mySeat])}). ${myRank === 1 ? "Protect the lead; deny rivals the big projects." : "You need points — the big rounds are where the game is decided."}`);
  return notes;
}
const HAIRCUT_HINT_ROUND = 4;

const DEFAULT_NAMES = ["You", "Player 2", "Player 3", "Player 4", "Player 5"];

export function Advisor({ net }: { net: PolicyNet }) {
  const [names, setNames] = useState<string[]>(DEFAULT_NAMES);
  const [mySeat, setMySeat] = useState(0);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [winner, setWinner] = useState(0);
  const [winningBid, setWinningBid] = useState(0);

  const state = useMemo(() => rebuild(outcomes), [outcomes]);
  const round = state.round;
  const gameOver = isDone(state);
  const elig = gameOver ? [] : eligible(state);
  const myEligible = elig.includes(mySeat);

  const decision = useMemo(() => {
    if (gameOver || !myEligible) return null;
    return analyzeDecision(net, state, mySeat, "expert");
  }, [state, mySeat, myEligible, gameOver, net]);

  const notes = useMemo(
    () => (decision ? rationale(state, mySeat, decision.recommended) : []),
    [decision, state, mySeat]
  );

  function record() {
    if (gameOver) return;
    const w = winner;
    const wb = Math.max(0, Math.min(maxBidFor(state.cash[w]), Math.round(winningBid)));
    setOutcomes((o) => [...o, { winner: w, winningBid: wb }]);
    setWinningBid(0);
  }
  function undo() { setOutcomes((o) => o.slice(0, -1)); }
  function reset() { setOutcomes([]); }

  const sc = scores(state);
  const order = sc.map((v, i) => i).sort((a, b) => sc[b] - sc[a]);

  return (
    <div className="grid">
      <div className="panel grid">
        <h2 style={{ margin: "2px 0 0" }}>Optimal-move advisor</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          Playing real people? Record each round's <b>winner</b> and <b>winning bid</b> as they're
          announced. The advisor rebuilds the exact game state (everyone's cash is derivable from
          public winning bids) and tells you the equilibrium-optimal bid for <b>your</b> seat.
        </p>
        <div className="row">
          <label className="fld">
            Which player are you?
            <select value={mySeat} onChange={(e) => setMySeat(parseInt(e.target.value))}>
              {names.map((n, i) => <option key={i} value={i}>{n}</option>)}
            </select>
          </label>
          <label className="fld" style={{ flex: 1, minWidth: 200 }}>
            Rename players (comma-separated)
            <input
              type="text"
              value={names.join(", ")}
              onChange={(e) => {
                const parts = e.target.value.split(",").map((s) => s.trim());
                const nn = [...DEFAULT_NAMES];
                for (let i = 0; i < N_PLAYERS; i++) if (parts[i]) nn[i] = parts[i];
                setNames(nn);
              }}
            />
          </label>
          <div className="spacer" />
          <button className="btn secondary" onClick={undo} disabled={outcomes.length === 0}>Undo</button>
          <button className="btn secondary" onClick={reset} disabled={outcomes.length === 0}>Reset</button>
        </div>
      </div>

      <div className="panel grid">
        <div className="round-head">
          <Prize round={Math.min(round, N_ROUNDS - 1)} />
          <div style={{ flex: 1, minWidth: 240 }}>
            <div className="small muted">
              {gameOver ? "Game complete" : `Round ${round + 1} of 10`} · {fmt(TOTAL_RETURNS)} total returns in play
            </div>
            <div style={{ height: 8 }} />
            <RoundPips state={state} />
          </div>
        </div>
        <PlayersTableNamed state={state} names={names} mySeat={mySeat} />
      </div>

      {!gameOver && (
        <div className="panel grid">
          <h3 style={{ margin: 0 }}>Your optimal bid — round {round + 1} ({fmt(RETURNS[round])})</h3>
          {myEligible && decision ? (
            <>
              <div className="kpi">
                <div className="k"><span>Recommended bid</span><b style={{ color: "var(--gold)" }}>{fmt(decision.recommended)}</b></div>
                <div className="k"><span>Your cash</span><b>{fmt1(state.cash[mySeat])}</b></div>
                <div className="k"><span>Max legal bid</span><b>{fmt(maxBidFor(state.cash[mySeat]))}</b></div>
                <div className="k"><span>Win signal</span><b>{(decision.value >= 0 ? "+" : "") + decision.value.toFixed(2)}</b></div>
              </div>
              <div className="grid" style={{ gridTemplateColumns: "1fr", gap: 12 }}>
                <div>
                  <div className="small muted" style={{ marginBottom: 6 }}>Equilibrium bid distribution (mix these to stay unpredictable)</div>
                  <BidDistribution rows={decision.distribution} recommended={decision.recommended} cash={state.cash[mySeat]} />
                </div>
                <div className="grid" style={{ gap: 6 }}>
                  <div className="small muted">Why</div>
                  {notes.map((n, i) => <div className="notice" key={i}>{n}</div>)}
                </div>
              </div>
            </>
          ) : (
            <div className="notice bad">
              You're locked out of this round (you didn't win any of rounds 1–6). Nothing to bid.
            </div>
          )}
        </div>
      )}

      {!gameOver && (
        <div className="panel grid">
          <div className="small muted">Record what actually happened this round</div>
          <div className="row" style={{ alignItems: "flex-end" }}>
            <label className="fld">
              Winner
              <select value={winner} onChange={(e) => setWinner(parseInt(e.target.value))}>
                {elig.map((p) => <option key={p} value={p}>{names[p]}</option>)}
              </select>
            </label>
            <label className="fld" style={{ width: 150 }}>
              Winning bid ($m)
              <input type="number" min={0} value={winningBid} onChange={(e) => setWinningBid(parseInt(e.target.value || "0"))} />
            </label>
            <button className="btn" onClick={record}>Record round →</button>
          </div>
          <div className="tiny muted">
            Tip: if you won, put yourself as the winner and the amount you paid. The advisor will
            update every player's remaining cash automatically.
          </div>
        </div>
      )}

      {gameOver && (
        <div className="panel result-final">
          <h3 style={{ margin: 0 }}>Final standings</h3>
          <table className="tbl">
            <thead><tr><th>#</th><th>Player</th><th className="num">Projects</th><th className="num">Score</th></tr></thead>
            <tbody>
              {order.map((seat, i) => (
                <tr key={seat} className={i === 0 ? "r1" : ""}>
                  <td>{i + 1}</td>
                  <td>{names[seat]}{seat === mySeat ? " (you)" : ""}</td>
                  <td className="num mono">{fmt(state.projValue[seat])}</td>
                  <td className="num mono" style={{ fontWeight: 800 }}>{fmt1(sc[seat])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {outcomes.length > 0 && (
        <div className="panel">
          <div className="small muted" style={{ marginBottom: 8 }}>Recorded rounds</div>
          <div className="log">
            {outcomes.map((o, i) => (
              <div className="logrow" key={i}>
                <div className="mono">R{i + 1}</div>
                <div><b style={{ color: SEAT_COLORS[o.winner] }}>{names[o.winner]}</b> won {fmt(RETURNS[i])} for {fmt(o.winningBid)}</div>
                <div className="muted mono">{fmt(RETURNS[i])}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Players table with custom names (advisor uses arbitrary names).
function PlayersTableNamed({ state, names, mySeat }: { state: GameState; names: string[]; mySeat: number }) {
  const sc = scores(state);
  const leader = sc.indexOf(Math.max(...sc));
  const qualifyPhase = state.round >= QUALIFY_ROUND_IDX;
  return (
    <div className="players">
      {names.map((name, i) => {
        const qualified = state.wonAny[i];
        const barPct = Math.max(2, (state.cash[i] / 500) * 100);
        return (
          <div className={`pl${i === mySeat ? " me" : ""}${i === leader ? " lead" : ""}`} key={i}>
            <div className="dot" style={{ background: SEAT_COLORS[i] }} />
            <div className="nm">
              {name}{i === mySeat ? " (you)" : ""}
              {state.round > 0 || qualifyPhase ? (
                <span className={`chip ${qualified ? "q" : "nq"}`}>{qualified ? "qualified" : qualifyPhase ? "locked out" : "not yet"}</span>
              ) : null}
            </div>
            <div className="col-cash">
              <div className="small muted">cash</div>
              <div className="mono" style={{ fontWeight: 700 }}>{fmt1(state.cash[i])}</div>
              <div className="bar"><i style={{ width: `${barPct}%` }} /></div>
            </div>
            <div className="col-proj">
              <div className="small muted">projects</div>
              <div className="mono" style={{ fontWeight: 700 }}>{fmt(state.projValue[i])}</div>
            </div>
            <div className="scoreval"><div className="small muted">score</div>{fmt1(sc[i])}</div>
          </div>
        );
      })}
    </div>
  );
}
