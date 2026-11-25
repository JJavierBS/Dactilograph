export class LearnSession {
  readonly letters: readonly string[];
  index: number;
  successMs: number | null = null;

  constructor(letters: readonly string[], start: string | null = null) {
    if (letters.length === 0) {
      throw new Error("The session needs at least one letter");
    }
    this.letters = letters;
    this.index = start === null ? 0 : Math.max(0, letters.indexOf(start));
  }

  get target(): string {
    return this.letters[this.index]!;
  }

  get succeeded(): boolean {
    return this.successMs !== null;
  }

  select(letter: string): void {
    const index = this.letters.indexOf(letter);
    if (index === -1) {
      throw new Error(`Unknown letter ${letter}`);
    }
    this.index = index;
    this.successMs = null;
  }

  next(): void {
    this.index = (this.index + 1) % this.letters.length;
    this.successMs = null;
  }

  previous(): void {
    this.index = (this.index - 1 + this.letters.length) % this.letters.length;
    this.successMs = null;
  }

  check(confirmed: readonly string[], nowMs: number): boolean {
    if (this.succeeded || !confirmed.includes(this.target)) {
      return false;
    }
    this.successMs = nowMs;
    return true;
  }
}
