import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves the project at /auction-game-ai/. In dev, base is "/".
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/auction-game-ai/" : "/",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false },
}));
