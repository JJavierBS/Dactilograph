export type Point3 = readonly [number, number, number];

export type Handedness = "Left" | "Right";

export interface Hand {
  readonly landmarks: readonly Point3[];
  readonly handedness: Handedness;
  readonly imageWidth: number;
  readonly imageHeight: number;
}

export const NUM_LANDMARKS = 21;
export const WRIST = 0;
export const INDEX_FINGER_TIP = 8;
export const MIDDLE_FINGER_MCP = 9;
