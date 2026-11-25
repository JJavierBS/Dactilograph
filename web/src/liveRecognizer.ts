import { DynamicRecognizer, type SequencePredictor } from "./dynamicRecognizer";
import type { Hand } from "./hand";
import type { Prediction } from "./staticClassifier";
import { StaticLetterStabilizer } from "./staticStabilizer";

export type StaticPredictor = (hand: Hand) => Promise<Prediction>;

export interface Reading {
  readonly hand: Hand | null;
  readonly prediction: Prediction | null;
  readonly confirmed: readonly string[];
}

export class LiveRecognizer {
  private stabilizer = new StaticLetterStabilizer();
  private dynamic: DynamicRecognizer;

  constructor(
    private readonly predictStatic: StaticPredictor,
    private readonly predictSequence: SequencePredictor,
  ) {
    this.dynamic = new DynamicRecognizer(predictSequence);
  }

  reset(): void {
    this.stabilizer = new StaticLetterStabilizer();
    this.dynamic = new DynamicRecognizer(this.predictSequence);
  }

  async read(nowMs: number, hand: Hand | null): Promise<Reading> {
    const prediction = hand === null ? null : await this.predictStatic(hand);
    const made = [this.stabilizer.update(nowMs, prediction), this.dynamic.update(nowMs, hand)];
    const confirmed = made.filter((letter): letter is string => letter !== null);
    return { hand, prediction, confirmed };
  }
}
