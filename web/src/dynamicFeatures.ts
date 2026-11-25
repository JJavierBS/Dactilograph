import type { Hand } from "./hand";
import { normalizeHand } from "./normalization";
import { trajectoryFeatures } from "./trajectory";

export function meanHandshape(hands: readonly Hand[]): number[] {
  if (hands.length === 0) {
    throw new Error("The sequence has no hands");
  }
  const sums: number[] = [];
  for (const shape of hands.map(normalizeHand)) {
    shape.forEach((value, index) => {
      sums[index] = (sums[index] ?? 0) + value;
    });
  }
  return sums.map((value) => value / hands.length);
}

export function sequenceFeatures(
  hands: readonly Hand[],
  timestampsMs: readonly number[],
): number[] {
  return [...trajectoryFeatures(hands, timestampsMs), ...meanHandshape(hands)];
}
