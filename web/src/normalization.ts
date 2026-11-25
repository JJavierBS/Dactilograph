import { MIDDLE_FINGER_MCP, WRIST, type Hand, type Point3 } from "./hand";

const MIN_PALM_SIZE = 1e-6;

export function landmark<T>(points: readonly T[], index: number): T {
  const point = points[index];
  if (point === undefined) {
    throw new Error(`The hand has no landmark ${index}`);
  }
  return point;
}

export function toSquarePixels(hand: Hand): Point3[] {
  const { imageWidth: width, imageHeight: height } = hand;
  return hand.landmarks.map(([x, y, z]) => [x * width, y * height, z * width]);
}

export function palmSize(points: readonly (readonly number[])[]): number {
  const wrist = landmark(points, WRIST);
  const middle = landmark(points, MIDDLE_FINGER_MCP);
  let squares = 0;
  middle.forEach((value, axis) => {
    squares += (value - landmark(wrist, axis)) ** 2;
  });
  const size = Math.sqrt(squares);
  if (size < MIN_PALM_SIZE) {
    throw new Error("The hand has zero size and cannot be normalized");
  }
  return size;
}

export function normalizeHand(hand: Hand): number[] {
  const pixels = toSquarePixels(hand);
  const [wristX, wristY, wristZ] = landmark(pixels, WRIST);
  const centered = pixels.map(([x, y, z]): Point3 => [x - wristX, y - wristY, z - wristZ]);
  const size = palmSize(centered);
  const mirror = hand.handedness === "Left" ? -1 : 1;
  return centered.flatMap(([x, y, z]) => [(x / size) * mirror, y / size, z / size]);
}
