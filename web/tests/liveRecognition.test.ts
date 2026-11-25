import { describe, expect, it } from "vitest";
import {
  CONFIRMATIONS,
  COOLDOWN_MS,
  DynamicRecognizer,
  MIN_FRAMES,
  NO_DYNAMIC_LETTER,
  WINDOW_MS,
} from "../src/dynamicRecognizer";
import type { Hand, Point3 } from "../src/hand";
import { LiveRecognizer } from "../src/liveRecognizer";
import type { Prediction } from "../src/staticClassifier";
import { HOLD_MS, MIN_CONFIDENCE, StaticLetterStabilizer } from "../src/staticStabilizer";

const SURE = MIN_CONFIDENCE + 0.1;
const UNSURE = MIN_CONFIDENCE - 0.1;
const FRAME_MS = 33;
const HAND: Hand = {
  landmarks: Array.from({ length: 21 }, (): Point3 => [0, 0, 0]),
  handedness: "Right",
  imageWidth: 640,
  imageHeight: 480,
};

function prediction(letter: string, confidence: number): Prediction {
  return { letter, confidence };
}

class FakePredictor {
  calls = 0;

  constructor(public letter: string) {}

  readonly predict = (): string => {
    this.calls += 1;
    return this.letter;
  };
}

function feed(
  recognizer: DynamicRecognizer,
  startMs: number,
  numFrames: number,
  frameMs: number = FRAME_MS,
): (string | null)[] {
  return Array.from({ length: numFrames }, (_, frame) =>
    recognizer.update(startMs + frame * frameMs, HAND),
  );
}

const framesToFillWindow = Math.floor(WINDOW_MS / FRAME_MS) + 1;

describe("StaticLetterStabilizer", () => {
  it("confirms a letter after it is held", () => {
    const stabilizer = new StaticLetterStabilizer();
    expect(stabilizer.update(0, prediction("A", SURE))).toBeNull();
    expect(stabilizer.update(HOLD_MS / 2, prediction("A", SURE))).toBeNull();
    expect(stabilizer.update(HOLD_MS, prediction("A", SURE))).toBe("A");
  });

  it("confirms a held letter only once", () => {
    const stabilizer = new StaticLetterStabilizer();
    stabilizer.update(0, prediction("A", SURE));
    stabilizer.update(HOLD_MS, prediction("A", SURE));
    expect(stabilizer.update(HOLD_MS * 5, prediction("A", SURE))).toBeNull();
  });

  it("restarts the hold when the letter changes", () => {
    const stabilizer = new StaticLetterStabilizer();
    stabilizer.update(0, prediction("A", SURE));
    expect(stabilizer.update(HOLD_MS, prediction("B", SURE))).toBeNull();
    expect(stabilizer.update(HOLD_MS * 2, prediction("B", SURE))).toBe("B");
  });

  it("restarts the hold on an unsure or missing prediction", () => {
    const stabilizer = new StaticLetterStabilizer();
    stabilizer.update(0, prediction("A", SURE));
    stabilizer.update(HOLD_MS / 2, prediction("A", UNSURE));
    expect(stabilizer.update(HOLD_MS, prediction("A", SURE))).toBeNull();
    stabilizer.update(HOLD_MS * 1.5, null);
    expect(stabilizer.update(HOLD_MS * 2, prediction("A", SURE))).toBeNull();
  });

  it("confirms the same letter again after letting go", () => {
    const stabilizer = new StaticLetterStabilizer();
    stabilizer.update(0, prediction("A", SURE));
    stabilizer.update(HOLD_MS, prediction("A", SURE));
    stabilizer.update(5000, null);
    stabilizer.update(10000, prediction("A", SURE));
    expect(stabilizer.update(10000 + HOLD_MS, prediction("A", SURE))).toBe("A");
  });
});

describe("DynamicRecognizer", () => {
  it("predicts nothing until the window is full", () => {
    const predictor = new FakePredictor("Z");
    feed(new DynamicRecognizer(predictor.predict), 0, framesToFillWindow - 5);
    expect(predictor.calls).toBe(0);
  });

  it("returns a letter after several confirmations", () => {
    const results = feed(
      new DynamicRecognizer(new FakePredictor("Z").predict),
      0,
      framesToFillWindow + CONFIRMATIONS,
    );
    expect(results.filter((result) => result !== null)).toEqual(["Z"]);
    const first = results.indexOf("Z");
    expect(results.slice(first - CONFIRMATIONS + 1, first)).toEqual(
      Array(CONFIRMATIONS - 1).fill(null),
    );
  });

  it("never returns the no-letter label", () => {
    const results = feed(new DynamicRecognizer(new FakePredictor(NO_DYNAMIC_LETTER).predict), 0, 200);
    expect(new Set(results)).toEqual(new Set([null]));
  });

  it("restarts the confirmation when the prediction changes", () => {
    const predictor = new FakePredictor("Z");
    const recognizer = new DynamicRecognizer(predictor.predict);
    const results: (string | null)[] = [];
    while (predictor.calls < CONFIRMATIONS - 1) {
      results.push(...feed(recognizer, results.length * FRAME_MS, 1));
    }
    predictor.letter = "J";
    results.push(...feed(recognizer, results.length * FRAME_MS, CONFIRMATIONS - 1));
    expect(new Set(results)).toEqual(new Set([null]));
    expect(feed(recognizer, results.length * FRAME_MS, 1)).toEqual(["J"]);
  });

  it("does not recognize the same gesture twice", () => {
    const predictor = new FakePredictor("Z");
    const recognizer = new DynamicRecognizer(predictor.predict);
    const first = feed(recognizer, 0, framesToFillWindow + CONFIRMATIONS);
    predictor.calls = 0;
    const results = feed(recognizer, first.length * FRAME_MS, Math.floor(COOLDOWN_MS / FRAME_MS));
    expect(new Set(results)).toEqual(new Set([null]));
    expect(predictor.calls).toBe(0);
  });

  it("does not classify a window with too few hands", () => {
    const predictor = new FakePredictor("Z");
    const recognizer = new DynamicRecognizer(predictor.predict);
    for (let frame = 0; frame < 200; frame++) {
      recognizer.update(frame * FRAME_MS, frame % 5 === 0 ? HAND : null);
    }
    expect(Math.floor(framesToFillWindow / 5)).toBeLessThan(MIN_FRAMES);
    expect(predictor.calls).toBe(0);
  });

  it("classifies a window recorded at a low frame rate", () => {
    const predictor = new FakePredictor("Z");
    const slowFrameMs = 125;
    feed(
      new DynamicRecognizer(predictor.predict),
      0,
      Math.floor(WINDOW_MS / slowFrameMs) + CONFIRMATIONS,
      slowFrameMs,
    );
    expect(predictor.calls).toBeGreaterThan(0);
  });

  it("drops old frames from the window", () => {
    const sizes: number[] = [];
    const recognizer = new DynamicRecognizer((hands, timestampsMs) => {
      sizes.push(hands.length);
      expect(timestampsMs.at(-1)! - timestampsMs[0]!).toBeLessThanOrEqual(WINDOW_MS);
      return NO_DYNAMIC_LETTER;
    });
    feed(recognizer, 0, 200);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(framesToFillWindow);
  });
});

describe("LiveRecognizer", () => {
  it("confirms static and moving letters from the same frames", async () => {
    const live = new LiveRecognizer(
      async () => prediction("A", SURE),
      new FakePredictor("Z").predict,
    );
    const confirmed: string[] = [];
    for (let frame = 0; frame <= framesToFillWindow + CONFIRMATIONS; frame++) {
      confirmed.push(...(await live.read(frame * FRAME_MS, HAND)).confirmed);
    }
    expect(confirmed).toEqual(["A", "Z"]);
  });

  it("has no prediction without a hand", async () => {
    const live = new LiveRecognizer(async () => prediction("A", SURE), () => "Z");
    expect(await live.read(0, null)).toEqual({ hand: null, prediction: null, confirmed: [] });
  });

  it("forgets the held letter after a reset", async () => {
    const live = new LiveRecognizer(async () => prediction("A", SURE), () => NO_DYNAMIC_LETTER);
    await live.read(0, HAND);
    live.reset();
    expect((await live.read(HOLD_MS, HAND)).confirmed).toEqual([]);
  });
});
