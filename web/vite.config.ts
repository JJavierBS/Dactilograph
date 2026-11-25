import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  test: {
    alias: {
      "onnxruntime-web/wasm": "onnxruntime-web",
    },
  },
});
