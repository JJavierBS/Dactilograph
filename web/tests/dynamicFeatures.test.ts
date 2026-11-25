import { describe, expect, it } from "vitest";
import { sequenceFeatures } from "../src/dynamicFeatures";
import { interpolate, linspace, median, resample } from "../src/trajectory";
import { expectClose, sequenceFixtures } from "./fixtures";

const TOLERANCE = 1e-9;

describe("sequenceFeatures", () => {
  it.each(sequenceFixtures.map((fixture) => [fixture.letter, fixture] as const))(
    "matches Python for a recorded %s",
    (_, { hands, timestampsMs, features }) => {
      expectClose(sequenceFeatures(hands, timestampsMs), features, TOLERANCE);
    },
  );
});

describe("resample", () => {
  it("interpolates at evenly spaced instants", () => {
    expect(resample([0, 10, 30], [[0], [10], [30]], 4)).toEqual([[0], [10], [20], [30]]);
  });

  it("rejects timestamps that do not increase", () => {
    expect(() => resample([0, 0], [[1], [2]])).toThrow();
  });

  it("needs at least two frames and one timestamp per frame", () => {
    expect(() => resample([0], [[1]])).toThrow();
    expect(() => resample([0, 1], [[1]])).toThrow();
  });
});

describe("helpers", () => {
  it("takes the middle value or the mean of the two middle values", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });

  it("ends linspace exactly at the stop value", () => {
    expect(linspace(0, 1, 3)).toEqual([0, 0.5, 1]);
  });

  it("clamps interpolation outside the known range", () => {
    expect(interpolate(-1, [0, 1], [5, 6])).toBe(5);
    expect(interpolate(2, [0, 1], [5, 6])).toBe(6);
  });
});
