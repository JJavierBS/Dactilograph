import { HandLandmarker } from "@mediapipe/tasks-vision";
import type { Hand } from "./hand";

export function drawHand(
  context: CanvasRenderingContext2D,
  hand: Hand,
  color: string,
  lineWidth: number,
): void {
  const { width, height } = context.canvas;
  const points = hand.landmarks.map(([x, y]) => [x * width, y * height] as const);
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
