import type { Reading } from "./liveRecognizer";

export interface Screen {
  readonly name: string;
  readonly element: HTMLElement;
  readonly usesCamera: boolean;
  enter(nowMs: number): void;
  leave(): void;
  update(reading: Reading, nowMs: number): void;
  handleKey(event: KeyboardEvent): boolean;
}

export function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`Missing element #${id}`);
  }
  return found as T;
}

export function formatConfidence(confidence: number): string {
  return confidence.toFixed(2).replace(".", ",");
}
