import { useEffect, useState } from "react";
import { loadPolicy, PolicyNet } from "./bot/net";
import { PlayGame } from "./ui/PlayGame";
import { Advisor } from "./ui/Advisor";
import { StrategyPage } from "./ui/StrategyPage";
import { runParityCheck } from "./bot/parity";

type Tab = "play" | "advisor" | "strategy";

export function App() {
  const [tab, setTab] = useState<Tab>("play");
  const [net, setNet] = useState<PolicyNet | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadPolicy()
      .then((n) => {
        setNet(n);
        if (import.meta.env.DEV) runParityCheck(n);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <div className="logo">A</div>
          <div>
            <h1>Auction Game AI</h1>
            <p>Trained self-play bots for the 5-player sealed-bid tournament</p>
          </div>
        </div>
        <nav className="tabs">
          <button className={`tab ${tab === "play" ? "active" : ""}`} onClick={() => setTab("play")}>
            Play vs Bots
          </button>
          <button className={`tab ${tab === "advisor" ? "active" : ""}`} onClick={() => setTab("advisor")}>
            Advisor
          </button>
          <button className={`tab ${tab === "strategy" ? "active" : ""}`} onClick={() => setTab("strategy")}>
            Strategy
          </button>
        </nav>
      </header>

      {err && (
        <div className="panel notice bad">
          Could not load the trained bot ({err}). If you just cloned the repo, the model is
          published to <code>web/public/policy.json</code> by the training pipeline.
        </div>
      )}
      {!net && !err && <div className="panel muted">Loading the trained bot…</div>}

      {net && tab === "play" && <PlayGame net={net} />}
      {net && tab === "advisor" && <Advisor net={net} />}
      {tab === "strategy" && <StrategyPage />}

      <div className="footer">
        Bots trained by self-play (fictitious self-play, PPO). Everything runs in your
        browser. &nbsp;·&nbsp;
        <a href="https://github.com/hassankanji/auction-game-ai" target="_blank" rel="noreferrer">
          source on GitHub
        </a>
      </div>
    </div>
  );
}
