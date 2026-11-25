import type { Prediction } from "./staticClassifier";

export const HOLD_MS = 400;
export const MIN_CONFIDENCE = 0.6;

export class StaticLetterStabilizer {
  private candidate: string | null = null;
  private sinceMs = 0;
  private emitted = false;

  constructor(
    private readonly holdMs: number = HOLD_MS,
    private readonly minConfidence: number = MIN_CONFIDENCE,
  ) {}

  update(nowMs: number, prediction: Prediction | null): string | null {
    if (prediction === null || prediction.confidence < this.minConfidence) {
      this.candidate = null;
      return null;
    }
    if (prediction.letter !== this.candidate) {
      this.candidate = prediction.letter;
      this.sinceMs = nowMs;
      this.emitted = false;
      return null;
    }
    if (this.emitted || nowMs - this.sinceMs < this.holdMs) {
      return null;
    }
    this.emitted = true;
    return this.candidate;
  }
}
