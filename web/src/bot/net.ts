// Minimal MLP forward pass for the exported PolicyValueNet (format "mlp-pv-v1").
// Mirrors PyTorch semantics: Linear (y = x·Wᵀ + b, W shape [out,in]), LayerNorm over the
// last dim with biased variance, and Tanh. See training/policy.py :: export_weights.

export interface LinearLayer {
  type: "linear";
  weight: number[][]; // [out][in]
  bias: number[];
}
export interface LayerNormLayer {
  type: "layernorm";
  weight: number[];
  bias: number[];
  eps: number;
}
export interface TanhLayer {
  type: "tanh";
}
export type Layer = LinearLayer | LayerNormLayer | TanhLayer;

export interface PolicyWeights {
  format: string;
  in_dim: number;
  hidden: number;
  n_actions: number;
  bid_fractions: number[];
  layers: Layer[];
  pi: { weight: number[][]; bias: number[] };
  value: { weight: number[][]; bias: number[] };
}

function linear(x: number[], W: number[][], b: number[]): number[] {
  const out = new Array(W.length);
  for (let o = 0; o < W.length; o++) {
    let s = b[o];
    const row = W[o];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function layernorm(x: number[], w: number[], b: number[], eps: number): number[] {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varr = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    varr += d * d;
  }
  varr /= n; // biased variance, matching PyTorch LayerNorm
  const inv = 1 / Math.sqrt(varr + eps);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

function tanh(x: number[]): number[] {
  return x.map(Math.tanh);
}

export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

export class PolicyNet {
  constructor(public w: PolicyWeights) {}

  private body(feat: number[]): number[] {
    let h = feat;
    for (const layer of this.w.layers) {
      if (layer.type === "linear") h = linear(h, layer.weight, layer.bias);
      else if (layer.type === "layernorm") h = layernorm(h, layer.weight, layer.bias, layer.eps);
      else if (layer.type === "tanh") h = tanh(h);
    }
    return h;
  }

  // Returns policy probabilities over bid fractions, plus the value estimate.
  evaluate(feat: number[]): { probs: number[]; logits: number[]; value: number } {
    const h = this.body(feat);
    const logits = linear(h, this.w.pi.weight, this.w.pi.bias);
    const value = linear(h, this.w.value.weight, this.w.value.bias)[0];
    return { probs: softmax(logits), logits, value };
  }

  get bidFractions(): number[] {
    return this.w.bid_fractions;
  }
}

let cached: PolicyNet | null = null;

// Load the exported weights from the site's base URL (works in dev and on Pages).
export async function loadPolicy(): Promise<PolicyNet> {
  if (cached) return cached;
  const url = `${import.meta.env.BASE_URL}policy.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load policy.json (${res.status})`);
  const w = (await res.json()) as PolicyWeights;
  cached = new PolicyNet(w);
  return cached;
}
