import { describe, expect, it } from "vitest";
import referenceHands from "../public/models/reference_hands.json";
import { ALPHABET, DYNAMIC_LETTERS } from "../src/alphabet";
import { fitHand } from "../src/chalk";
import { LearnSession } from "../src/learning";
import { PROGRESS_KEY, Progress, type KeyValueStore } from "../src/progress";

class MemoryStore implements KeyValueStore {
  readonly items = new Map<string, string>();

  getItem(key: string): string | null {
    return this.items.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.items.set(key, value);
  }
}

describe("ALPHABET", () => {
  it("has the 30 letters with a reference hand each", () => {
    expect(ALPHABET).toHaveLength(30);
    expect(new Set(ALPHABET)).toEqual(new Set(Object.keys(referenceHands)));
    expect([...DYNAMIC_LETTERS].every((letter) => ALPHABET.includes(letter))).toBe(true);
  });
});

describe("LearnSession", () => {
  it("starts at the given letter", () => {
    expect(new LearnSession(["A", "B", "C"], "B").target).toBe("B");
    expect(new LearnSession(["A", "B", "C"], "Z").target).toBe("A");
  });

  it("wraps around when moving", () => {
    const session = new LearnSession(["A", "B", "C"]);
    session.previous();
    expect(session.target).toBe("C");
    session.next();
    expect(session.target).toBe("A");
  });

  it("succeeds once when the target is confirmed", () => {
    const session = new LearnSession(["A", "B"]);
    expect(session.check(["B"], 0)).toBe(false);
    expect(session.check(["A"], 100)).toBe(true);
    expect(session.check(["A"], 200)).toBe(false);
    expect(session.successMs).toBe(100);
  });

  it("forgets the success when changing letter", () => {
    const session = new LearnSession(["A", "B"]);
    session.check(["A"], 0);
    session.next();
    expect(session.succeeded).toBe(false);
    session.select("A");
    expect(session.succeeded).toBe(false);
  });
});

describe("Progress", () => {
  it("saves and loads learned letters and the high score", () => {
    const store = new MemoryStore();
    const progress = new Progress();
    expect(progress.learn("B")).toBe(true);
    expect(progress.learn("B")).toBe(false);
    expect(progress.recordScore(7)).toBe(true);
    expect(progress.recordScore(3)).toBe(false);
    progress.save(store);
    const loaded = Progress.load(store);
    expect(loaded.learned).toEqual(new Set(["B"]));
    expect(loaded.highScore).toBe(7);
  });

  it("starts empty when nothing or garbage was saved", () => {
    const store = new MemoryStore();
    expect(Progress.load(store).learned.size).toBe(0);
    store.setItem(PROGRESS_KEY, "not json");
    expect(Progress.load(store).highScore).toBe(0);
    expect(Progress.load(null).learned.size).toBe(0);
  });

  it("uses the same format as the desktop app", () => {
    const store = new MemoryStore();
    new Progress(new Set(["C", "A"]), 4).save(store);
    expect(JSON.parse(store.getItem(PROGRESS_KEY)!)).toEqual({ aprendidas: ["A", "C"], record: 4 });
  });
});

describe("fitHand", () => {
  it("fits the hand inside the box keeping its proportions", () => {
    const points = fitHand(
      [
        [0, 0],
        [2, 1],
      ],
      [0, 0],
      100,
      0.1,
    );
    expect(points).toEqual([
      [10, 30],
      [90, 70],
    ]);
  });
});
