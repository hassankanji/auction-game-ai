import { useMemo, useState } from "react";
import { PolicyNet } from "../bot/net";
import {
  GameState, newGame, eligible, isDone, applyRound, topOfBids, validateBid,
  maxBidFor, scores, RETURNS, QUALIFY_ROUND_IDX, N_ROUNDS,
} from "../engine/game";
import { Difficulty, botBid, botResubmit, analyzeDecision } from "../bot/bot";
import {
  SEAT_NAMES, SEAT_COLORS, fmt, fmt1, RoundPips, Prize, PlayersTable, BidDistribution,
} from "./common";

type Phase = "setup" | "bidding" | "tiebreak" | "reveal" | "done";

interface Reveal {
  round: number;
  ret: number;
  winner: number;
  winningBid: number;
  originalBids: Record<number, number>;
  tieBroken: boolean;
  humanEligible: boolean;
  humanBid: number;
}

const HUMAN = 0;

export function PlayGame({ net }: { net: PolicyNet }) {
  const [difficulty, setDifficulty] = useState<Difficulty>("expert");
  const [state, setState] = useState<GameState>(() => newGame());
  const [phase, setPhase] = useState<Phase>("setup");
  const [humanBid, setHumanBid] = useState<number>(0);
  const [tied, setTied] = useState<number[]>([]);
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [revealAll, setRevealAll] = useState(false);
  const [log, setLog] = useState<Reveal[]>([]);

  const humanEligible = phase !== "done" && !isDone(state) && eligible(state).includes(HUMAN);
  const cash0 = state.cash[HUMAN];
  const mx0 = maxBidFor(cash0);

  const hint = useMemo(() => {
    if (!showHint || !humanEligible) return null;
    return analyzeDecision(net, state, HUMAN, "expert");
  }, [showHint, humanEligible, state, net]);

  function start() {
    const g = newGame();
    setState(g);
    setLog([]);
    setReveal(null);
    setHumanBid(Math.round(RETURNS[0] * 0.6));
    setPhase("bidding");
  }

  // Resolve a set of bids, auto-running bot-only tie resubmissions. If the human is tied,
  // returns { pause:true, tied } so the UI can ask the human to resubmit.
  function resolveOrPause(
    st: GameState,
    bids: Record<number, number>
  ): { pause: false; winner: number; winningBid: number; tieBroken: boolean } | { pause: true; tied: number[] } {
    let cur = { ...bids };
    let tieBroken = false;
    for (let attempt = 0; attempt < 16; attempt++) {
      const { tied: t, winningBid } = topOfBids(cur);
      if (t.length === 1) return { pause: false, winner: t[0], winningBid, tieBroken };
      tieBroken = true;
      if (t.includes(HUMAN)) return { pause: true, tied: t };
      const nb: Record<number, number> = {};
      for (const p of t) nb[p] = botResubmit(net, st, p, difficulty);
      cur = nb;
    }
    // Deterministic fallback: lowest seat index among final tied set.
    const { tied: t, winningBid } = topOfBids(cur);
    return { pause: false, winner: t[0], winningBid, tieBroken: true };
  }

  function finishRound(
    st: GameState,
    winner: number,
    winningBid: number,
    originalBids: Record<number, number>,
    tieBroken: boolean,
    hElig: boolean,
    hBid: number
  ) {
    const rv: Reveal = {
      round: st.round,
      ret: RETURNS[st.round],
      winner,
      winningBid,
      originalBids,
      tieBroken,
      humanEligible: hElig,
      humanBid: hBid,
    };
    const ns = applyRound(st, winner, winningBid, originalBids, tieBroken);
    setReveal(rv);
    setLog((L) => [...L, rv]);
    setState(ns);
    setPhase(isDone(ns) ? "reveal" : "reveal");
  }

  function submit() {
    const st = state;
    const elig = eligible(st);
    const bids: Record<number, number> = {};
    const hBid = humanEligible ? validateBid(cash0, humanBid) : 0;
    if (humanEligible) bids[HUMAN] = hBid;
    for (const p of elig) if (p !== HUMAN) bids[p] = botBid(net, st, p, difficulty);

    const res = resolveOrPause(st, bids);
    if (res.pause) {
      setTied(res.tied);
      setPhase("tiebreak");
      setHumanBid(Math.min(mx0, hBid + 1));
      return;
    }
    finishRound(st, res.winner, res.winningBid, bids, res.tieBroken, humanEligible, hBid);
  }

  function submitTiebreak() {
    const st = state;
    const hBid = validateBid(cash0, humanBid);
    const nb: Record<number, number> = {};
    for (const p of tied) nb[p] = p === HUMAN ? hBid : botResubmit(net, st, p, difficulty);
    const res = resolveOrPause(st, nb);
    if (res.pause) {
      setTied(res.tied);
      setHumanBid(Math.min(mx0, hBid + 1));
      return;
    }
    // Preserve original sealed bids for the reveal (use last-round humanBid as human's).
    finishRound(st, res.winner, res.winningBid, nb, true, true, hBid);
  }

  function playBotsOnly() {
    // Human not eligible this round (locked out of 7-10): bots contest it.
    const st = state;
    const elig = eligible(st);
    const bids: Record<number, number> = {};
    for (const p of elig) bids[p] = botBid(net, st, p, difficulty);
    const res = resolveOrPause(st, bids);
    if (res.pause) {
      // shouldn't happen (human not eligible), but guard: resolve deterministically
      finishRound(st, res.tied[0], bids[res.tied[0]] ?? 0, bids, true, false, 0);
      return;
    }
    finishRound(st, res.winner, res.winningBid, bids, res.tieBroken, false, 0);
  }

  function next() {
    if (isDone(state)) {
      setPhase("done");
    } else {
      setReveal(null);
      const nextElig = eligible(state).includes(HUMAN);
      setHumanBid(nextElig ? Math.round(RETURNS[state.round] * 0.5) : 0);
      setPhase("bidding");
    }
  }

  // ---- render --------------------------------------------------------------

  if (phase === "setup") {
    return (
      <div className="panel grid">
        <h2 style={{ margin: "2px 0 0" }}>Play the tournament</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          You are <b style={{ color: SEAT_COLORS[0] }}>You</b> against four trained bots. Ten
          sealed-bid rounds, $500m each. Win projects, qualify for the big rounds (7–10), and
          finish with the highest score = projects won + 0.9 × cash. Remember: cash is cut 10%
          after round 5 and another 10% at the end, so hoarding is expensive.
        </p>
        <div className="row">
          <label className="fld">
            Bot difficulty
            <select value={difficulty} onChange={(e) => setDifficulty(e.target.value as Difficulty)}>
              <option value="easy">Easy (loose, beatable)</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
              <option value="expert">Expert (true equilibrium — top difficulty)</option>
            </select>
          </label>
          <div className="spacer" />
          <button className="btn" onClick={start}>Start game →</button>
        </div>
      </div>
    );
  }

  const done = phase === "done";
  const sc = scores(state);
  const order = sc.map((v, i) => i).sort((a, b) => sc[b] - sc[a]);

  return (
    <div className="grid">
      <div className="panel grid">
        <div className="round-head">
          <Prize round={Math.min(state.round, N_ROUNDS - 1)} />
          <div style={{ flex: 1, minWidth: 240 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <div className="small muted">
                Round {Math.min(state.round + (phase === "reveal" ? 0 : 1), N_ROUNDS)} of 10
                {state.round >= QUALIFY_ROUND_IDX ? " · qualification rounds" : ""}
              </div>
              <div className="row" style={{ gap: 6 }}>
                <button className={`ghost ${showHint ? "on" : ""}`} onClick={() => setShowHint((v) => !v)}>
                  {showHint ? "Hint on" : "Show bot hint"}
                </button>
                <button className={`ghost ${revealAll ? "on" : ""}`} onClick={() => setRevealAll((v) => !v)}>
                  {revealAll ? "Revealing bids" : "Reveal all bids"}
                </button>
              </div>
            </div>
            <div style={{ height: 8 }} />
            <RoundPips state={state} />
          </div>
        </div>
      </div>

      <div className="panel">
        <PlayersTable state={state} meSeat={HUMAN} />
      </div>

      {/* Bidding / tiebreak / reveal / done */}
      {phase === "bidding" && !done && (
        <div className="panel grid">
          {humanEligible ? (
            <>
              <div className="row" style={{ alignItems: "flex-end" }}>
                <label className="fld" style={{ flex: 1, minWidth: 220 }}>
                  Your sealed bid for the {fmt(RETURNS[state.round])} project
                  <input
                    type="range" min={0} max={mx0} value={Math.min(humanBid, mx0)}
                    onChange={(e) => setHumanBid(parseInt(e.target.value))}
                  />
                </label>
                <label className="fld" style={{ width: 130 }}>
                  Amount ($m)
                  <input
                    type="number" min={0} max={mx0} value={Math.min(humanBid, mx0)}
                    onChange={(e) => setHumanBid(parseInt(e.target.value || "0"))}
                  />
                </label>
                <button className="btn" onClick={submit}>Submit bid</button>
              </div>
              <div className="tiny muted">
                Legal range $0–{fmt(mx0)}. Bids are sealed — the bots choose at the same time.
              </div>
              {hint && (
                <div className="panel" style={{ background: "var(--bg)" }}>
                  <div className="row" style={{ justifyContent: "space-between" }}>
                    <b className="small">What the trained bot would do in your seat</b>
                    <span className="small muted">
                      est. win-chance signal: {(hint.value >= 0 ? "+" : "") + hint.value.toFixed(2)}
                    </span>
                  </div>
                  <div style={{ height: 8 }} />
                  <BidDistribution rows={hint.distribution} recommended={hint.recommended} cash={cash0} />
                  <div className="row" style={{ marginTop: 8 }}>
                    <button className="btn secondary" onClick={() => setHumanBid(hint.recommended)}>
                      Use bot's pick ({fmt(hint.recommended)})
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="grid">
              <div className="notice bad">
                You didn't win any of rounds 1–6, so you're <b>locked out</b> of the big rounds.
                The bots will contest this one.
              </div>
              <button className="btn" onClick={playBotsOnly}>Play out round →</button>
            </div>
          )}
        </div>
      )}

      {phase === "tiebreak" && (
        <div className="panel grid">
          <div className="notice warn">
            Tie for the top bid between {tied.map((p) => SEAT_NAMES[p]).join(", ")} — only tied
            players resubmit. Enter a new bid.
          </div>
          <div className="row" style={{ alignItems: "flex-end" }}>
            <label className="fld" style={{ flex: 1, minWidth: 220 }}>
              Your resubmitted bid
              <input type="range" min={0} max={mx0} value={Math.min(humanBid, mx0)}
                onChange={(e) => setHumanBid(parseInt(e.target.value))} />
            </label>
            <label className="fld" style={{ width: 130 }}>
              Amount ($m)
              <input type="number" min={0} max={mx0} value={Math.min(humanBid, mx0)}
                onChange={(e) => setHumanBid(parseInt(e.target.value || "0"))} />
            </label>
            <button className="btn" onClick={submitTiebreak}>Resubmit</button>
          </div>
        </div>
      )}

      {phase === "reveal" && reveal && (
        <div className="panel grid">
          <div className="row" style={{ alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <div className="small muted">Round {reveal.round + 1} result</div>
              <div style={{ fontSize: 18, fontWeight: 800 }}>
                <span style={{ color: SEAT_COLORS[reveal.winner] }}>{SEAT_NAMES[reveal.winner]}</span>{" "}
                won the {fmt(reveal.ret)} project for {fmt(reveal.winningBid)}
                {reveal.tieBroken ? " (after a tie-break)" : ""}
              </div>
              {reveal.humanEligible && (
                <div style={{ marginTop: 6 }}>
                  {reveal.winner === HUMAN ? (
                    <span className="badge win">
                      You won it (net {reveal.ret - reveal.winningBid >= 0 ? "+" : "−"}
                      {fmt(Math.abs(reveal.ret - reveal.winningBid))})
                    </span>
                  ) : (
                    <span className="badge lose">You bid {fmt(reveal.humanBid)} — {reveal.humanBid >= reveal.winningBid ? "tie lost" : "outbid"}</span>
                  )}
                </div>
              )}
            </div>
            <button className="btn" onClick={next}>{isDone(state) ? "See final result →" : "Next round →"}</button>
          </div>
          {revealAll && (
            <table className="tbl">
              <thead><tr><th>Player</th><th className="num">Sealed bid</th><th></th></tr></thead>
              <tbody>
                {Object.keys(reveal.originalBids).map(Number).sort((a, b) => reveal.originalBids[b] - reveal.originalBids[a]).map((p) => (
                  <tr key={p}>
                    <td><span className="dot" style={{ background: SEAT_COLORS[p], display: "inline-block", marginRight: 8 }} />{SEAT_NAMES[p]}</td>
                    <td className="num mono">{fmt(reveal.originalBids[p])}</td>
                    <td>{p === reveal.winner ? <span className="badge win">won</span> : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {done && (
        <div className="panel result-final">
          <h2 style={{ margin: 0 }}>Final standings</h2>
          <table className="tbl">
            <thead><tr><th>#</th><th>Player</th><th className="num">Projects</th><th className="num">Cash ×0.9</th><th className="num">Score</th></tr></thead>
            <tbody>
              {order.map((seat, i) => (
                <tr key={seat} className={i === 0 ? "r1" : ""}>
                  <td>{i + 1}</td>
                  <td><span className="dot" style={{ background: SEAT_COLORS[seat], display: "inline-block", marginRight: 8 }} />{SEAT_NAMES[seat]}{seat === HUMAN ? " (you)" : ""}</td>
                  <td className="num mono">{fmt(state.projValue[seat])}</td>
                  <td className="num mono">{fmt1(0.9 * state.cash[seat])}</td>
                  <td className="num mono" style={{ fontWeight: 800 }}>{fmt1(sc[seat])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="notice good">
            {order[0] === HUMAN
              ? "🏆 You won the tournament! You out-played the trained bots."
              : `${SEAT_NAMES[order[0]]} won. You finished #${order.indexOf(HUMAN) + 1}. Try the Advisor tab to study optimal play.`}
          </div>
          <div className="row"><button className="btn" onClick={start}>Play again</button></div>
        </div>
      )}

      {log.length > 0 && (
        <div className="panel">
          <div className="small muted" style={{ marginBottom: 8 }}>Round history</div>
          <div className="log">
            {log.map((r) => (
              <div className="logrow" key={r.round}>
                <div className="mono">R{r.round + 1}</div>
                <div>
                  <b style={{ color: SEAT_COLORS[r.winner] }}>{SEAT_NAMES[r.winner]}</b> won {fmt(r.ret)} for {fmt(r.winningBid)}
                </div>
                <div className="muted mono">{r.humanEligible ? `you: ${fmt(r.humanBid)}` : "locked out"}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
