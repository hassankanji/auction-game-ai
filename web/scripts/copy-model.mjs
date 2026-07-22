// Copies the trained model (single source of truth: /models) into web/public so the app
// can fetch it. Runs automatically before `dev` and `build`.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pub = resolve(here, "../public");
mkdirSync(pub, { recursive: true });

for (const name of ["policy.json", "parity_ref.json"]) {
  const src = resolve(here, "../../models", name);
  const dst = resolve(pub, name);
  if (existsSync(src)) {
    copyFileSync(src, dst);
    console.log(`[copy-model] ${name} -> web/public/`);
  } else if (name === "policy.json") {
    console.warn(`[copy-model] WARNING: ${src} not found — train first (python -m training.train)`);
  }
}
