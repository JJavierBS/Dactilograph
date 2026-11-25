import { describe, expect, it } from "vitest";
import type { Hand, Point3 } from "../src/hand";
import { normalizeHand, palmSize } from "../src/normalization";
import { expectClose, normalizationFixtures } from "./fixtures";

const TOLERANCE = 1e-9;

function flatHand(handedness: Hand["handedness"] = "Right"): Hand {
  const landmarks = Array.from({ length: 21 }, (_, index): Point3 => [0.5 + index * 0.01, 0.5, 0]);
  return { landmarks, handedness, imageWidth: 640, imageHeight: 480 };
}

describe("normalizeHand", () => {
  it.each(normalizationFixtures.map((fixture, index) => [index, fixture] as const))(
    "matches Python for recorded hand %i",
    (_, { hand, normalized }) => {
      expectClose(normalizeHand(hand), normalized, TOLERANCE);
    },
  );

  it("puts the wrist at the origin and the palm at unit size", () => {
    const normalized = normalizeHand(flatHand());
    expect(normalized.slice(0, 3)).toEqual([0, 0, 0]);
    expect(Math.hypot(...normalized.slice(27, 30))).toBeCloseTo(1, 12);
  });

  it("mirrors the left hand", () => {
    const right = normalizeHand(flatHand("Right"));
    const left = normalizeHand(flatHand("Left"));
    expectClose(
      left,
      right.map((value, index) => (index % 3 === 0 ? -value : value)),
      TOLERANCE,
    );
  });

  it("rejects a hand without size", () => {
    expect(() => palmSize(Array.from({ length: 21 }, () => [1, 1, 1]))).toThrow();
  });
});
