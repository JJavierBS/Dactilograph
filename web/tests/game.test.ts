import { describe, expect, it } from "vitest";
import {
  BUBBLE_RADIUS,
  type Bubble,
  BubbleGame,
  HELP_AFTER_MS,
  MAX_BUBBLES,
  MAX_X,
  MIN_X,
  MISS_DURATION_MS,
  POP_DURATION_MS,
  POPS_PER_LEVEL,
  START_LIVES,
  START_SPAWN_INTERVAL_MS,
} from "../src/game";

const LETTERS = ["A", "B", "C", "D", "E", "F"];

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

function newGame(letters: readonly string[] = LETTERS): BubbleGame {
  return new BubbleGame(letters, seededRandom(1), 0);
}

function bubble(letter: string, x: number, y: number, spawnedMs = 0): Bubble {
  return { letter, x, y, spawnedMs };
}

describe("BubbleGame", () => {
  it("shows the first bubble at the bottom on the first update", () => {
    const game = newGame();
    game.update(0);
    expect(game.bubbles).toHaveLength(1);
    const [first] = game.bubbles;
    expect(LETTERS).toContain(first!.letter);
    expect(first!.y).toBeCloseTo(1 + BUBBLE_RADIUS);
    expect(first!.x).toBeGreaterThanOrEqual(MIN_X);
    expect(first!.x).toBeLessThanOrEqual(MAX_X);
  });

  it("makes bubbles rise over time", () => {
    const game = newGame();
    game.update(0);
    const start = game.bubbles[0]!.y;
    game.update(1000);
    expect(game.bubbles[0]!.y).toBeCloseTo(start - game.speedPerMs * 1000);
  });

  it("adds bubbles at the spawn interval", () => {
    const game = newGame();
    game.update(0);
    game.update(START_SPAWN_INTERVAL_MS - 100);
    expect(game.bubbles).toHaveLength(1);
    game.update(START_SPAWN_INTERVAL_MS);
    expect(game.bubbles).toHaveLength(2);
  });

  it("never repeats a letter on screen", () => {
    const game = newGame(["A", "B"]);
    for (let ms = 0; ms < 40000; ms += 4000) {
      game.update(ms);
      const letters = game.bubbles.map((item) => item.letter);
      expect(new Set(letters).size).toBe(letters.length);
    }
  });

  it("never has more than the maximum bubbles", () => {
    const game = newGame(Array.from({ length: 20 }, (_, index) => String.fromCharCode(65 + index)));
    game.lives = 100;
    for (let step = 0; step < 200; step++) {
      game.update(step * 500);
      expect(game.bubbles.length).toBeLessThanOrEqual(MAX_BUBBLES);
    }
  });

  it("pops the bubble of the letter made", () => {
    const game = newGame();
    game.bubbles = [bubble("A", 0.5, 0.5), bubble("B", 0.3, 0.6)];
    expect(game.tryPop("A", 1000)).toBe(true);
    expect(game.bubbles.map((item) => item.letter)).toEqual(["B"]);
    expect(game.score).toBe(1);
    expect(game.pops.map((pop) => pop.letter)).toEqual(["A"]);
  });

  it("ignores a letter without a bubble", () => {
    const game = newGame();
    game.bubbles = [bubble("A", 0.5, 0.5)];
    expect(game.tryPop("Z", 1000)).toBe(false);
    expect(game.score).toBe(0);
    expect(game.bubbles).toHaveLength(1);
  });

  it("removes the pop animation after its duration", () => {
    const game = newGame();
    game.bubbles = [bubble("A", 0.5, 0.5)];
    game.tryPop("A", 0);
    game.update(POP_DURATION_MS / 2);
    expect(game.pops).toHaveLength(1);
    game.update(POP_DURATION_MS);
    expect(game.pops).toEqual([]);
  });

  it("takes a life for an escaped bubble and keeps it as a miss for a while", () => {
    const game = newGame();
    game.bubbles = [bubble("A", 0.3, -BUBBLE_RADIUS + 0.001)];
    game.update(1000);
    expect(game.lives).toBe(START_LIVES - 1);
    expect(game.misses.map((miss) => [miss.letter, miss.x])).toEqual([["A", 0.3]]);
    game.update(1000 + MISS_DURATION_MS);
    expect(game.misses).toEqual([]);
  });

  it("ends without lives and then ignores letters", () => {
    const game = newGame();
    game.lives = 1;
    game.bubbles = [bubble("A", 0.5, -BUBBLE_RADIUS + 0.001), bubble("B", 0.5, 0.5)];
    game.update(1000);
    expect(game.isOver).toBe(true);
    expect(game.bubbles).toEqual([]);
    expect(game.tryPop("B", 1000)).toBe(false);
    game.update(10000);
    expect(game.bubbles).toEqual([]);
  });

  it("gets faster with each level", () => {
    const game = newGame();
    const speed = game.speedPerMs;
    const interval = game.spawnIntervalMs;
    game.score = POPS_PER_LEVEL;
    expect(game.level).toBe(2);
    expect(game.speedPerMs).toBeGreaterThan(speed);
    expect(game.spawnIntervalMs).toBeLessThan(interval);
  });

  it("helps with the highest bubble that has waited too long", () => {
    const game = newGame();
    game.bubbles = [
      bubble("A", 0.5, 0.4),
      bubble("B", 0.5, 0.2),
      bubble("C", 0.5, 0.1, HELP_AFTER_MS),
    ];
    expect(game.helpLetter(HELP_AFTER_MS - 100)).toBeNull();
    expect(game.helpLetter(HELP_AFTER_MS)).toBe("B");
  });

  it("needs letters", () => {
    expect(() => newGame([])).toThrow();
  });
});
