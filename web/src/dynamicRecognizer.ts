import type { Hand } from "./hand";

export const NO_DYNAMIC_LETTER = "-";
export const WINDOW_MS = 1500;
export const MIN_WINDOW_SPAN = 0.9;
export const MIN_FRAMES = 12;
export const CONFIRMATIONS = 3;
export const COOLDOWN_MS = 1000;

export type SequencePredictor = (hands: readonly Hand[], timestampsMs: readonly number[]) => string;

interface TimedHand {
  readonly timestampMs: number;
  readonly hand: Hand;
}

export class DynamicRecognizer {
  private frames: TimedHand[] = [];
  private candidate: string | null = null;
  private streak = 0;
  private cooldownUntilMs = -Infinity;

  constructor(private readonly predict: SequencePredictor) {}

  update(timestampMs: number, hand: Hand | null): string | null {
    if (hand !== null) {
      this.frames.push({ timestampMs, hand });
    }
    const oldest = timestampMs - WINDOW_MS;
    while (this.frames.length > 0 && this.frames[0]!.timestampMs < oldest) {
      this.frames.shift();
    }

    if (timestampMs < this.cooldownUntilMs || !this.windowIsComplete()) {
      this.resetStreak();
      return null;
    }

    const letter = this.predict(
      this.frames.map((frame) => frame.hand),
      this.frames.map((frame) => frame.timestampMs),
    );
    if (letter === NO_DYNAMIC_LETTER) {
      this.resetStreak();
      return null;
    }
    this.streak = letter === this.candidate ? this.streak + 1 : 1;
    this.candidate = letter;
    if (this.streak < CONFIRMATIONS) {
      return null;
    }

    this.frames = [];
    this.resetStreak();
    this.cooldownUntilMs = timestampMs + COOLDOWN_MS;
    return letter;
  }

  private windowIsComplete(): boolean {
    if (this.frames.length < MIN_FRAMES) {
      return false;
    }
    const spanMs = this.frames.at(-1)!.timestampMs - this.frames[0]!.timestampMs;
    return spanMs >= WINDOW_MS * MIN_WINDOW_SPAN;
  }

  private resetStreak(): void {
    this.candidate = null;
    this.streak = 0;
  }
}
