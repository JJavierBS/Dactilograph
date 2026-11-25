import { INDEX_FINGER_TIP, WRIST, type Hand, type Handedness } from "./hand";
import { landmark, palmSize, toSquarePixels } from "./normalization";

export const TRACKED_POINTS = [WRIST, INDEX_FINGER_TIP] as const;
export const DEFAULT_NUM_FRAMES = 20;

export function median(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error("The median needs at least one value");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? landmark(sorted, middle)
    : (landmark(sorted, middle - 1) + landmark(sorted, middle)) / 2;
}

export function mostCommonHandedness(hands: readonly Hand[]): Handedness {
  const counts = new Map<Handedness, number>();
  for (const hand of hands) {
    counts.set(hand.handedness, (counts.get(hand.handedness) ?? 0) + 1);
  }
  let best: Handedness = "Right";
  let bestCount = 0;
  for (const [handedness, count] of counts) {
    if (count > bestCount) {
      best = handedness;
      bestCount = count;
    }
  }
  return best;
}

export function handTrajectory(hands: readonly Hand[]): number[][] {
  if (hands.length === 0) {
    throw new Error("The sequence has no hands");
  }
  const pixels = hands.map((hand) => toSquarePixels(hand).map(([x, y]) => [x, y] as const));
  const scale = median(pixels.map((points) => palmSize(points)));
  const [originX, originY] = landmark(landmark(pixels, 0), WRIST);
  const mirror = mostCommonHandedness(hands) === "Left" ? -1 : 1;
  return pixels.map((points) =>
    TRACKED_POINTS.flatMap((index) => {
      const [x, y] = landmark(points, index);
      return [((x - originX) / scale) * mirror, (y - originY) / scale];
    }),
  );
}

export function linspace(start: number, stop: number, count: number): number[] {
  if (count === 1) {
    return [start];
  }
  const step = (stop - start) / (count - 1);
  const values = Array.from({ length: count }, (_, index) => index * step + start);
  values[count - 1] = stop;
  return values;
}

export function interpolate(x: number, xs: readonly number[], ys: readonly number[]): number {
  const last = xs.length - 1;
  if (x <= landmark(xs, 0)) {
    return landmark(ys, 0);
  }
  if (x >= landmark(xs, last)) {
    return landmark(ys, last);
  }
  let low = 0;
  let high = last;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (landmark(xs, middle) <= x) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const x0 = landmark(xs, low);
  const y0 = landmark(ys, low);
  const slope = (landmark(ys, high) - y0) / (landmark(xs, high) - x0);
  return slope * (x - x0) + y0;
}

export function resample(
  timestampsMs: readonly number[],
  frames: readonly (readonly number[])[],
  numFrames: number = DEFAULT_NUM_FRAMES,
): number[][] {
  if (timestampsMs.length !== frames.length) {
    throw new Error(`Got ${timestampsMs.length} timestamps for ${frames.length} frames`);
  }
  if (timestampsMs.length < 2) {
    throw new Error("At least 2 frames are needed to resample");
  }
  for (let index = 1; index < timestampsMs.length; index++) {
    if (landmark(timestampsMs, index) <= landmark(timestampsMs, index - 1)) {
      throw new Error("Timestamps must be strictly increasing");
    }
  }
  const width = landmark(frames, 0).length;
  const columns = Array.from({ length: width }, (_, column) =>
    frames.map((frame) => landmark(frame, column)),
  );
  const first = landmark(timestampsMs, 0);
  const last = landmark(timestampsMs, timestampsMs.length - 1);
  const times = linspace(first, last, numFrames);
  return times.map((time) => columns.map((values) => interpolate(time, timestampsMs, values)));
}

export function trajectoryFeatures(
  hands: readonly Hand[],
  timestampsMs: readonly number[],
  numFrames: number = DEFAULT_NUM_FRAMES,
): number[] {
  return resample(timestampsMs, handTrajectory(hands), numFrames).flat();
}
