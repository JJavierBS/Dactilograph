export const START_LIVES = 3;
export const POPS_PER_LEVEL = 5;
export const START_SPEED_PER_MS = 0.08 / 1000;
export const SPEED_GROWTH = 1.15;
export const START_SPAWN_INTERVAL_MS = 4000;
export const SPAWN_INTERVAL_DECAY = 0.9;
export const MIN_SPAWN_INTERVAL_MS = 1500;
export const MAX_BUBBLES = 4;
export const BUBBLE_RADIUS = 0.07;
export const MIN_X = 0.12;
export const MAX_X = 0.88;
export const HELP_AFTER_MS = 6000;
export const POP_DURATION_MS = 400;
export const MISS_DURATION_MS = 1200;

export type Random = () => number;

export interface Bubble {
  readonly letter: string;
  readonly x: number;
  y: number;
  readonly spawnedMs: number;
}

export interface Pop {
  readonly letter: string;
  readonly x: number;
  readonly y: number;
  readonly poppedMs: number;
}

export class BubbleGame {
  bubbles: Bubble[] = [];
  pops: Pop[] = [];
  misses: Pop[] = [];
  score = 0;
  lives = START_LIVES;
  private lastUpdateMs: number;
  private nextSpawnMs: number;

  constructor(
    private readonly letters: readonly string[],
    private readonly random: Random,
    nowMs: number,
  ) {
    if (letters.length === 0) {
      throw new Error("The game needs at least one letter");
    }
    this.lastUpdateMs = nowMs;
    this.nextSpawnMs = nowMs;
  }

  get level(): number {
    return 1 + Math.floor(this.score / POPS_PER_LEVEL);
  }

  get speedPerMs(): number {
    return START_SPEED_PER_MS * SPEED_GROWTH ** (this.level - 1);
  }

  get spawnIntervalMs(): number {
    const interval = START_SPAWN_INTERVAL_MS * SPAWN_INTERVAL_DECAY ** (this.level - 1);
    return Math.max(interval, MIN_SPAWN_INTERVAL_MS);
  }

  get isOver(): boolean {
    return this.lives <= 0;
  }

  update(nowMs: number): void {
    const elapsedMs = nowMs - this.lastUpdateMs;
    this.lastUpdateMs = nowMs;
    this.pops = this.pops.filter((pop) => nowMs - pop.poppedMs < POP_DURATION_MS);
    this.misses = this.misses.filter((miss) => nowMs - miss.poppedMs < MISS_DURATION_MS);
    if (this.isOver) {
      return;
    }

    for (const bubble of this.bubbles) {
      bubble.y -= this.speedPerMs * elapsedMs;
    }
    const escaped = this.bubbles.filter((bubble) => bubble.y < -BUBBLE_RADIUS);
    this.bubbles = this.bubbles.filter((bubble) => bubble.y >= -BUBBLE_RADIUS);
    this.lives = Math.max(this.lives - escaped.length, 0);
    this.misses.push(
      ...escaped.map((bubble) => ({ letter: bubble.letter, x: bubble.x, y: 0, poppedMs: nowMs })),
    );
    if (this.isOver) {
      this.bubbles = [];
      return;
    }

    if (nowMs >= this.nextSpawnMs) {
      this.spawn(nowMs);
      this.nextSpawnMs = nowMs + this.spawnIntervalMs;
    }
  }

  tryPop(letter: string, nowMs: number): boolean {
    if (this.isOver) {
      return false;
    }
    const target = highest(this.bubbles.filter((bubble) => bubble.letter === letter));
    if (target === undefined) {
      return false;
    }
    this.bubbles = this.bubbles.filter((bubble) => bubble !== target);
    this.pops.push({ letter: target.letter, x: target.x, y: target.y, poppedMs: nowMs });
    this.score += 1;
    return true;
  }

  helpLetter(nowMs: number): string | null {
    const waiting = this.bubbles.filter((bubble) => nowMs - bubble.spawnedMs >= HELP_AFTER_MS);
    return highest(waiting)?.letter ?? null;
  }

  private spawn(nowMs: number): void {
    if (this.bubbles.length >= MAX_BUBBLES) {
      return;
    }
    const onScreen = new Set(this.bubbles.map((bubble) => bubble.letter));
    const available = this.letters.filter((letter) => !onScreen.has(letter));
    if (available.length === 0) {
      return;
    }
    const letter = available[Math.floor(this.random() * available.length)]!;
    const x = MIN_X + this.random() * (MAX_X - MIN_X);
    this.bubbles.push({ letter, x, y: 1 + BUBBLE_RADIUS, spawnedMs: nowMs });
  }
}

function highest(bubbles: readonly Bubble[]): Bubble | undefined {
  return bubbles.reduce<Bubble | undefined>(
    (best, bubble) => (best === undefined || bubble.y < best.y ? bubble : best),
    undefined,
  );
}
