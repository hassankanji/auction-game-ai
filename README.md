# Auction Game AI

Trained, top-difficulty bots for the **5-Player Global Tournament Auction Game**, plus a
browser app to play against them and a real-time **advisor** that tells you your optimal
bid when you play real people.

> **Play it:** once deployed, the web app lives at
> `https://hassankanji.github.io/auction-game-ai/` (GitHub Pages — no install, works
> anywhere). The bots run entirely in your browser.

---

## The game

Five investors start with **$500m** each and bid in **10 sealed-bid auctions**, one per
round. Each round auctions a single renewable-energy project with a **fixed return**; the
highest integer bidder wins it, pays their bid, and keeps the return.

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------:|--:|--:|--:|---:|---:|---:|---:|---:|---:|----:|
| Return ($m) | 25 | 50 | 50 | 100 | 200 | 350 | 500 | 750 | 900 | 1000 |

Rules that make it deep:

- **Sealed & simultaneous.** You submit your bid privately; you only learn *who won* and
  *the winning bid* afterwards.
- **Qualification gate.** You may bid in rounds 7–10 (which hold **80% of all returns**)
  only if you won **at least one** of rounds 1–6.
- **Two 10% cash haircuts.** Remaining cash is cut 10% after round 5, and unspent cash is
  cut another 10% at scoring — so **$1 held to the end is worth only ~$0.81**. Hoarding is
  punished; deploying cash to win projects is rewarded.
- **Final score = value of projects won + 0.9 × remaining cash.** Highest score wins.
- **Ties** for the top bid: only the tied players resubmit, until it's broken.

**The key insight the bots exploit:** cash is only ever spent by winning, and winning bids
are announced — so although the brief says you aren't told opponents' cash, it is fully
*derivable*. Every player's budget is common knowledge; the only hidden information is the
current round's simultaneous bids. That turns the game into a sequence of budget-constrained
first-price auctions with a qualification gate — an ideal target for poker-style self-play
equilibrium training.

## What's here

```
engine/     Exact game engine (rules + agents). Source of truth. Fully tested.
training/   Self-play reinforcement learning + game-theoretic analysis -> trained bots.
analysis/   Strategy study: when to qualify, how much to shade, endgame cash value.
web/        Static site (Vite + TypeScript): play vs bots + the optimal-move advisor.
models/     Trained policy exported to JSON, run in-browser by the web app.
tests/      Engine correctness tests (pytest).
```

## ▶ Play now

**https://hassankanji.github.io/auction-game-ai/** — no install, works on any device. The
trained bot runs entirely in your browser.

- **Play vs Bots** — full 10-round tournament against four trained bots (Easy → Expert).
  Toggle a *bot hint* to see what the trained policy would do in your seat, and *reveal all
  bids* after each round to study play.
- **Advisor** — playing real people? Enter each round's winner and winning bid; it rebuilds
  the exact game state and tells you your equilibrium-optimal bid, with reasoning.
- **Strategy** — a plain-English tour of what strong play looks like.

## What the bots learned (a peek)

Probed at canonical decision points (`python -m analysis.strategy_report`):

| Situation | What the bot does |
|---|---|
| R1 opening, nobody qualified | Over-pays face value (~$30m for the $25m project) to lock in qualification |
| R6, **last** chance to qualify | Bids **$160–190m** for the $350m project — qualification is worth almost anything |
| R6, already qualified | Bids small / mixes — it doesn't need the round, so it preserves cash |
| R10 finale, biggest budget | Bids **$320–360m** to seize the $1,000m project |
| R10 finale, short on cash | Shoves nearly all-in — correct desperation play |
| R7, first big round, even budgets | Shades low, saving budget for the bigger $750/$900/$1,000 rounds |

## Results

The shipped bot (self-play, ~250 iterations, selected by a stable blend of win-share vs the
strategic heuristic and the mixed pool) — win-share over 800 games each, where **0.20 is
chance** in a 5-player game:

| Opponent (×4) | Bot win-share |
|---|---:|
| Strategic heuristic (best hand-crafted) | **0.98** |
| Value heuristic | 0.82 |
| Random | 0.73 |
| Mixed pool (random/value/strategic) | 0.60 |

It dominates every reference opponent. Training curves: [`analysis/training_curves.png`](analysis/training_curves.png).
Exploitability (how much a from-scratch best-response can beat it) is measured by
`python -m analysis.exploitability`; the number is reported in [`analysis/README.md`](analysis/README.md).

## Status

- [x] Game engine + tests (12 passing)
- [x] Heuristic reference bots (random / value / strategic)
- [x] Vectorised self-play simulator
- [x] Self-play RL training pipeline (PPO + fictitious self-play)
- [x] Strategy analysis + writeup ([`analysis/`](analysis/))
- [x] Web app (play vs bots)
- [x] Advisor tool (optimal move vs real opponents)
- [x] GitHub Pages deployment (auto-deploys on push)

## Run the engine / tests locally

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## How the bots are trained

The flagship bot is trained by **self-play** (each bot plays against copies of itself and a
population of past checkpoints — *fictitious self-play*), so its strategy is robust rather
than tuned to beat one specific opponent. Strength is measured by **exploitability**: how
much a best-response opponent can gain against it. A CFR / game-theory abstraction of the
game is used to cross-check the learned strategy and to *explain* what strong play looks
like. See [`analysis/`](analysis/) for the writeup.
