import { describe, expect, it } from "vitest";
import labelsFile from "../public/models/letter_classifier.json";
import dynamicModel from "../public/models/dynamic_classifier.json";
import { DynamicClassifier } from "../src/dynamicClassifier";
import { sequenceFeatures } from "../src/dynamicFeatures";
import { bestPrediction, StaticClassifier } from "../src/staticClassifier";
import staticCases from "./fixtures/static_classifier.json";
import { expectClose, sequenceFixtures } from "./fixtures";

const PROBABILITY_TOLERANCE = 1e-5;

describe("StaticClassifier", () => {
  it("gives the same probabilities and letter as Python", async () => {
    expect(labelsFile.labels).toEqual(staticCases.labels);
    const classifier = await StaticClassifier.load(
      "public/models/letter_classifier.onnx",
      labelsFile.labels,
    );
    for (const { input, probabilities, letter } of staticCases.cases) {
      const actual = Array.from(await classifier.probabilities(input));
      expectClose(actual, probabilities, PROBABILITY_TOLERANCE);
      expect((await classifier.predict(input)).letter).toBe(letter);
    }
  });

  it("picks the most likely label", () => {
    const prediction = bestPrediction(["A", "B", "C"], [0.2, 0.5, 0.3]);
    expect(prediction).toEqual({ letter: "B", confidence: 0.5 });
  });

  it("rejects labels that do not match the probabilities", () => {
    expect(() => bestPrediction(["A"], [0.5, 0.5])).toThrow();
  });
});

describe("DynamicClassifier", () => {
  const classifier = new DynamicClassifier(dynamicModel);

  it.each(sequenceFixtures.map((fixture) => [fixture.letter, fixture] as const))(
    "recognizes a recorded %s like Python",
    (_, { hands, timestampsMs, letter }) => {
      expect(classifier.predict(sequenceFeatures(hands, timestampsMs))).toBe(letter);
    },
  );

  it("only accepts the 1-nearest-neighbor model", () => {
    expect(() => new DynamicClassifier({ ...dynamicModel, k: 3 })).toThrow();
  });
});
