export const PROGRESS_KEY = "dactilograph-progreso";

export type KeyValueStore = Pick<Storage, "getItem" | "setItem">;

interface SavedProgress {
  readonly aprendidas?: unknown;
  readonly record?: unknown;
}

export class Progress {
  constructor(
    readonly learned: Set<string> = new Set(),
    public highScore: number = 0,
  ) {}

  static load(store: KeyValueStore | null): Progress {
    try {
      const saved = JSON.parse(store?.getItem(PROGRESS_KEY) ?? "{}") as SavedProgress;
      const learned = Array.isArray(saved.aprendidas)
        ? saved.aprendidas.filter((letter): letter is string => typeof letter === "string")
        : [];
      const record = Number(saved.record ?? 0);
      return new Progress(new Set(learned), Number.isFinite(record) ? record : 0);
    } catch {
      return new Progress();
    }
  }

  save(store: KeyValueStore | null): void {
    const data = { aprendidas: [...this.learned].sort(), record: this.highScore };
    try {
      store?.setItem(PROGRESS_KEY, JSON.stringify(data));
    } catch {
      return;
    }
  }

  learn(letter: string): boolean {
    if (this.learned.has(letter)) {
      return false;
    }
    this.learned.add(letter);
    return true;
  }

  recordScore(score: number): boolean {
    if (score <= this.highScore) {
      return false;
    }
    this.highScore = score;
    return true;
  }
}

export function browserStore(): KeyValueStore | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
