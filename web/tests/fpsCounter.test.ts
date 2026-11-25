import { describe, expect, it } from "vitest";
import { FpsCounter } from "../src/fpsCounter";

describe("FpsCounter", () => {
  it("has no rate after the first frame", () => {
    expect(new FpsCounter().tick(0)).toBe(0);
  });

  it("measures a constant frame rate", () => {
    const counter = new FpsCounter();
    let fps = 0;
    for (let frame = 0; frame < 20; frame++) {
      fps = counter.tick(frame * 100);
    }
    expect(fps).toBeCloseTo(10);
  });

  it("smooths a single slow frame", () => {
    const counter = new FpsCounter(0.9);
    for (let frame = 0; frame < 10; frame++) {
      counter.tick(frame * 100);
    }
    expect(counter.tick(900 + 1000)).toBeCloseTo(0.9 * 10 + 0.1 * 1);
  });
});
