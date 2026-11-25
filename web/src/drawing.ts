import { HandLandmarker } from "@mediapipe/tasks-vision";
import type { Hand } from "./hand";

export type Point2 = readonly [number, number];

export const FIT_MARGIN = 0.08;

export function drawSkeleton(
  context: CanvasRenderingContext2D,
  points: readonly Point2[],
  color: string,
  lineWidth: number,
): void {
  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = lineWidth;
  context.lineCap = "round";
  context.beginPath();
  for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
    const from = points[start];
    const to = points[end];
    if (from !== undefined && to !== undefined) {
      context.moveTo(from[0], from[1]);
      context.lineTo(to[0], to[1]);
    }
  }
  context.stroke();
  for (const [x, y] of points) {
    context.beginPath();
    context.arc(x, y, lineWidth * 1.4, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

export function drawHand(
  context: CanvasRenderingContext2D,
  hand: Hand,
  color: string,
  lineWidth: number,
): void {
  const { width, height } = context.canvas;
  const points = hand.landmarks.map(([x, y]): Point2 => [x * width, y * height]);
  drawSkeleton(context, points, color, lineWidth);
}

export function fitHand(
  points: readonly (readonly number[])[],
  size: number,
  margin: number = FIT_MARGIN,
): Point2[] {
  const xs = points.map((point) => point[0] ?? 0);
  const ys = points.map((point) => point[1] ?? 0);
  const low = [Math.min(...xs), Math.min(...ys)] as const;
  const range = [Math.max(...xs) - low[0], Math.max(...ys) - low[1]] as const;
  const usable = size * (1 - 2 * margin);
  const scale = usable / Math.max(range[0], range[1], 1e-6);
  const offset = [
    size * margin + (usable - range[0] * scale) / 2,
    size * margin + (usable - range[1] * scale) / 2,
  ] as const;
  return points.map(
    (point): Point2 => [
      ((point[0] ?? 0) - low[0]) * scale + offset[0],
      ((point[1] ?? 0) - low[1]) * scale + offset[1],
    ],
  );
}
