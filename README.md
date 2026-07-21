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

## Status

- [x] Game engine + tests
- [x] Heuristic reference bots (random / value / strategic)
- [ ] Fast vectorised simulator
- [ ] Self-play RL training pipeline
- [ ] Strategy analysis + writeup
- [ ] Web app (play vs bots)
- [ ] Advisor tool (optimal move vs real opponents)
- [ ] GitHub Pages deployment

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
