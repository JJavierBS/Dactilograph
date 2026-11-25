import type { Hand, Handedness, Point3 } from "../src/hand";
import normalizationCases from "./fixtures/normalization.json";
import sequenceCases from "./fixtures/sequences.json";

export interface HandFixture {
  landmarks: number[][];
  handedness: string;
  width: number;
  height: number;
}

export function toHand(fixture: HandFixture): Hand {
  if (fixture.handedness !== "Left" && fixture.handedness !== "Right") {
    throw new Error(`Unknown handedness ${fixture.handedness}`);
  }
  return {
    landmarks: fixture.landmarks.map((point): Point3 => {
      const [x, y, z] = point;
      if (x === undefined || y === undefined || z === undefined) {
        throw new Error("A landmark needs three coordinates");
      }
      return [x, y, z];
    }),
    handedness: fixture.handedness as Handedness,
    imageWidth: fixture.width,
    imageHeight: fixture.height,
  };
}

export const normalizationFixtures = normalizationCases.map((fixture) => ({
  hand: toHand(fixture),
  normalized: fixture.normalized,
}));

export const sequenceFixtures = sequenceCases.map((fixture) => ({
  hands: fixture.hands.map(toHand),
  timestampsMs: fixture.timestamps_ms,
  features: fixture.features,
  letter: fixture.letter,
}));

export function expectClose(
  actual: readonly number[],
  expected: readonly number[],
  tolerance: number,
): void {
  if (actual.length !== expected.length) {
    throw new Error(`Expected ${expected.length} values, got ${actual.length}`);
  }
  actual.forEach((value, index) => {
    const difference = Math.abs(value - expected[index]!);
    if (!(difference <= tolerance)) {
      throw new Error(`Value ${index}: expected ${expected[index]}, got ${value}`);
    }
  });
}
