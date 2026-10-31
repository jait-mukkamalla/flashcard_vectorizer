import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The production build lands inside the Python package, which serves it at /graph/ (end users never need Node).
export default defineConfig({
  base: "/graph/",
  plugins: [react()],
  build: {
    outDir: "../src/flashcard_vectorizer/static/graph",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
  server: {
    // `npm run dev` talks to a running `flashcard-vectorizer serve`. The backend refuses foreign
    // origins, so present the requests as coming from the app itself.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        headers: { origin: "http://127.0.0.1:8765" },
      },
    },
  },
  test: { environment: "node" },
});
