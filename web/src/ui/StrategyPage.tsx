import { RETURNS } from "../engine/game";
import { fmt } from "./common";

export function StrategyPage() {
  const bigSum = RETURNS.slice(6).reduce((a, b) => a + b, 0);
  const total = RETURNS.reduce((a, b) => a + b, 0);
  return (
    <div className="grid">
      <div className="panel grid">
        <h2 style={{ margin: "2px 0 0" }}>How the strong bots actually win</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          The bots here were trained by self-play (they played millions of games against
          copies of themselves and past versions), so their strategy is an approximate Nash
          equilibrium — robust, not tuned to beat one specific opponent. Here's what that
          strategy looks like in words.
        </p>
      </div>

      <div className="panel grid">
        <h3 style={{ margin: 0 }}>1. Cash is worth ~0.81×, so hoarding loses</h3>
        <p className="muted">
          Remaining cash is cut 10% after round 5 and another 10% at scoring. A dollar you
          carry from the start to the end scores only about <b>$0.81</b>. That means a project
          is worth winning whenever its return exceeds the end-value of the cash you spend —
          and it's why the winning bots deploy their budget aggressively rather than sitting on it.
        </p>
      </div>

      <div className="panel grid">
        <h3 style={{ margin: 0 }}>2. Qualification is the whole game</h3>
        <p className="muted">
          Rounds 7–10 hold {fmt(bigSum)} of the {fmt(total)} total returns — <b>{Math.round((bigSum / total) * 100)}%</b>.
          You can only bid in them if you won at least one of rounds 1–6. So the bots treat a
          cheap early win as near-mandatory: they'll pay a premium (well above the tiny face
          value) for round 1 or 2/3 if they're at risk of being locked out, because the option
          value of qualifying dwarfs the $25–50m sticker price.
        </p>
      </div>

      <div className="panel grid">
        <h3 style={{ margin: 0 }}>3. Everyone's cash is known — exploit it</h3>
        <p className="muted">
          Although the brief says you aren't told opponents' cash, it's fully derivable: money
          is only ever spent by winning, and winning bids are announced. The bots track every
          player's exact budget and bid to beat the strongest bidder who can actually afford to
          contest a project — and refuse to overpay when they hold a decisive budget lead into
          the biggest rounds.
        </p>
      </div>

      <div className="panel grid">
        <h3 style={{ margin: 0 }}>4. Sealed bids ⇒ mix your strategy</h3>
        <p className="muted">
          Because bids are sealed and simultaneous, you can't see and just-outbid — you commit
          blind. The equilibrium answer is a <b>mixed strategy</b>: a distribution over how much
          of your budget to commit, so opponents can't predict and counter you. That's exactly
          what the Advisor shows — a spread of bids, with the single best one highlighted.
        </p>
      </div>

      <div className="panel grid">
        <h3 style={{ margin: 0 }}>5. The endgame is a budget knife-fight</h3>
        <p className="muted">
          By rounds 7–10 the qualified players fight over {fmt(bigSum)} with the budgets they've
          preserved. The bots plan backwards: they enter the big rounds with enough cash to win
          the specific projects where they have a budget edge, and they'd rather win a {fmt(1000)}
          project for a lot than hold cash that scores 0.9×. Whoever best converted the cheap
          early rounds into a preserved budget usually takes the {fmt(900)} and {fmt(1000)} finales.
        </p>
      </div>

      <div className="panel">
        <div className="small muted">
          Want the numbers? The training curves and exploitability analysis live in the repo
          under <code>analysis/</code> and <code>models/train_log.json</code>.
        </div>
      </div>
    </div>
  );
}
