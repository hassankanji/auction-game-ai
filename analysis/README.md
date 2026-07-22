# Strategy analysis — how the trained bots win

This folder explains what the self-play bots actually learned, and gives you tools to
reproduce it. If you just want to *use* the strategy, the **Advisor** tab in the web app
applies all of this automatically. If you want to *understand* it, read on.

## The game in one paragraph

Five investors, $500m each, bid in 10 sealed first-price auctions. You keep the return of
any project you win. You can only bid in rounds 7–10 (which hold **$3,150m of the $3,925m**
total returns — 80%) if you won one of rounds 1–6. Cash is cut 10% after round 5 and another
10% at scoring, so a dollar held to the end is worth ~$0.81. Final score = projects won +
0.9 × cash; highest wins. The one structural gift: because money is only ever spent by
winning and winning bids are announced, **everyone's cash is public knowledge** — the only
hidden information is the current round's simultaneous bids.

## Five principles the bots converge on

**1. Cash is worth ~0.81×, so deploy it.** The double haircut means hoarding is a slow leak.
A project is worth winning whenever its return beats the *end-value* of the cash you spend.
The bots run their budgets down into projects rather than sitting on them.

**2. Qualification is the pivot of the whole game.** Missing rounds 7–10 forfeits 80% of the
available returns. So the bots treat a cheap early win as close to mandatory and will *overpay
on face value* to get one — e.g. paying ~$30m for the round-1 $25m project. That looks like a
$5–6m loss, but it buys an option on $3,150m. They only relax this once they're safely
qualified.

**3. Everyone's budget is known — bid against the player who can actually afford to fight.**
The bots reconstruct every opponent's exact cash and size their bids to beat the strongest
*solvent* rival for a project, not an imaginary one. With a decisive budget lead into the
finale they pay only what they must; when short, they pick one project to contest rather than
bleeding cash across several they'll lose.

**4. Sealed bids ⇒ mix.** You can't see and just-outbid; you commit blind. The equilibrium is
a *distribution* over how much budget to commit, so opponents can't predict and counter you.
The bot samples from this mix (and the app adds small off-grid jitter so a human can't beat it
by simply bidding "their number + 1").

**5. The endgame is a budget knife-fight.** By rounds 7–10 the qualified players fight over
$3,150m with the cash they preserved. Whoever converted the cheap early rounds into a bigger
preserved budget usually takes the $900m and $1,000m finales — which is why principle 2
(qualify *cheaply*) and principle 1 (don't hoard, but don't waste) are in tension and have to
be balanced. That balance is exactly what self-play tunes.

## How the bots were trained

- **Self-play + fictitious self-play.** All five seats share one policy that plays against
  copies of itself and a rolling pool of frozen past versions. Training against past selves
  (not just the current one) keeps the strategy robust instead of cycling into a gimmick that
  only beats today's opponent.
- **PPO**, with a reward that is primarily the **tournament win-share** (1 for a sole win, 1/k
  for a k-way tie for first, else 0) plus a small rank-shaping term — i.e. it optimises for
  *winning the game*, exactly as the brief scores it, not for raw points.
- **Action space:** a fixed grid of "fraction of my current cash" bids, which matches the
  sealed-bid decision (how much of my budget do I commit?) and keeps the policy well-defined.

## Reproduce / inspect

```bash
# Train (≈1 hour on CPU); exports the best model to models/policy.json
python -m training.train --iters 1800 --games 2048

# Read the strategy out of the trained net at canonical decision points
python -m analysis.strategy_report

# How exploitable is it? Trains a best-response and reports the gap over chance (0.20)
python -m analysis.exploitability --br-iters 300

# Training curves -> analysis/training_curves.png
python -m analysis.plot_training
```

## Results

See [`../models/train_log.json`](../models/train_log.json) for the full training log and
`training_curves.png` for the plots. Headline numbers (win-share, where 0.20 is chance in a
5-player game) and the exploitability gap are printed by the commands above and summarised in
the top-level [README](../README.md).
