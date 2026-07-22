// Dev-only parity check: verifies the TypeScript feature encoding and MLP forward pass
// reproduce the Python reference (dumped by training/dump_parity.py -> public/parity_ref.json)
// to within a small tolerance. Runs on load in dev; logs to the console.

import { PolicyNet } from "./net";
import { features } from "./features";

interface Case {
  cash: number[];
  won: number[];
  proj: number[];
  round: number;
  me: number;
  features: number[];
  logits: number[];
  value: number;
}

function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

export async function runParityCheck(net: PolicyNet): Promise<void> {
  let ref: { cases: Case[] };
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}parity_ref.json`);
    if (!res.ok) throw new Error(String(res.status));
    ref = await res.json();
  } catch {
    console.info("[parity] parity_ref.json not found — skipping (regenerate with training/dump_parity.py)");
    return;
  }
  let maxFeat = 0;
  let maxLogit = 0;
  for (const c of ref.cases) {
    const f = features(c.cash, c.won, c.proj, c.round, c.me);
    maxFeat = Math.max(maxFeat, maxAbsDiff(f, c.features));
    const { logits } = net.evaluate(c.features); // use Python features to isolate net error
    maxLogit = Math.max(maxLogit, maxAbsDiff(logits, c.logits));
  }
  const ok = maxFeat < 1e-4 && maxLogit < 2e-3;
  const tag = ok ? "✅ PASS" : "❌ FAIL";
  console[ok ? "info" : "error"](
    `[parity] ${tag}  features Δ=${maxFeat.toExponential(2)}  logits Δ=${maxLogit.toExponential(2)}  (${ref.cases.length} cases)`
  );
}
