import { ALPHABET } from "./alphabet";
import {
  BLUE,
  BOARD_BOTTOM,
  type Chalk,
  type Color,
  DIM,
  HEIGHT,
  handInBox,
  PINK,
  type Point2,
  putText,
  rgb,
  TAPE,
  textSize,
  WHITE,
  YELLOW,
} from "./chalk";
import type { Reading } from "./liveRecognizer";
import type { Progress } from "./progress";

export type ReferenceHands = Readonly<Record<string, readonly (readonly number[])[]>>;

export interface Context {
  readonly nowMs: number;
  readonly frame: CanvasImageSource | null;
  readonly reading: Reading | null;
  readonly mouse: Point2 | null;
  readonly references: ReferenceHands;
  readonly progress: Progress;
}

export interface CanvasScreen {
  readonly trayActive: string;
  readonly usesCamera: boolean;
  enter(context: Context): void;
  keyAction(key: string): string | null;
  handle(action: string, context: Context): string | null;
  update(context: Context): void;
  draw(canvas: CanvasRenderingContext2D, chalk: Chalk, context: Context): Button[];
}

export class Button {
  constructor(
    readonly x: number,
    readonly y: number,
    readonly width: number,
    readonly height: number,
    readonly action: string,
  ) {}

  contains(point: Point2 | null): boolean {
    if (point === null) {
      return false;
    }
    const [px, py] = point;
    return px >= this.x && px < this.x + this.width && py >= this.y && py < this.y + this.height;
  }
}

export const TRAY_ITEMS: readonly (readonly [string, string, string])[] = [
  ["Menú", "Esc", "menu"],
  ["Aprender", "1", "aprender"],
  ["Reconocer", "2", "reconocer"],
  ["Jugar", "3", "jugar"],
];
const TRAY_LEFT = 60;
const TRAY_GAP = 52;
const STICK_WIDTH = 66;
const STICK_HEIGHT = 14;
const STICK_LIFT = 5;
const LABEL_COLOR: Color = [240, 238, 235];
const LETTER_COLORS: readonly Color[] = [WHITE, YELLOW, PINK, BLUE];
const RING_DRAW_MS = 350;

export function drawTray(
  canvas: CanvasRenderingContext2D,
  active: string,
  mouse: Point2 | null,
  items: readonly (readonly [string, string, string])[] = TRAY_ITEMS,
): Button[] {
  const buttons: Button[] = [];
  let x = TRAY_LEFT;
  for (const [label, key, action] of items) {
    const [labelWidth] = textSize(canvas, label, 0.75, 2);
    const button = new Button(x - 10, BOARD_BOTTOM + 4, STICK_WIDTH + labelWidth + 36, 62, action);
    const lifted = button.contains(mouse);
    const stickTop = HEIGHT - 50 - (lifted ? STICK_LIFT : 0);
    const stickColor = action === active ? YELLOW : lifted ? WHITE : TAPE;
    canvas.fillStyle = rgb(stickColor);
    canvas.fillRect(x, stickTop, STICK_WIDTH, STICK_HEIGHT);
    canvas.beginPath();
    canvas.arc(x + STICK_WIDTH, stickTop + STICK_HEIGHT / 2, STICK_HEIGHT / 2, 0, Math.PI * 2);
    canvas.fill();
    putText(canvas, label, [x + STICK_WIDTH + 18, HEIGHT - 31], 0.75, LABEL_COLOR, 2);
    putText(canvas, key, [x + STICK_WIDTH + 18, HEIGHT - 9], 0.6, LABEL_COLOR, 1);
    buttons.push(button);
    x += STICK_WIDTH + 18 + labelWidth + TRAY_GAP;
  }
  return buttons;
}

export function letterColor(letter: string): Color {
  return LETTER_COLORS[Math.max(0, ALPHABET.indexOf(letter)) % LETTER_COLORS.length]!;
}

export function ringProgress(sinceMs: number | null | undefined, nowMs: number): number {
  if (sinceMs === null || sinceMs === undefined) {
    return 0;
  }
  return Math.min((nowMs - sinceMs) / RING_DRAW_MS, 1);
}

export function letterScale(letter: string, singleScale: number): number {
  return letter.length === 1 ? singleScale : singleScale * 0.62;
}

export function formatConfidence(confidence: number): string {
  return confidence.toFixed(2).replace(".", ",");
}

export function chalkUserHand(
  chalk: Chalk,
  reading: Reading | null,
  topLeft: Point2,
  size: Point2,
): void {
  if (reading?.hand) {
    chalk.hand(handInBox(reading.hand.landmarks, topLeft, size), YELLOW, 3);
  }
}

export function noHandNote(chalk: Chalk, reading: Reading | null, center: Point2): void {
  if (reading !== null && reading.hand === null) {
    chalk.text("no veo tu mano", center, 1.0, DIM, 2, "script", "center");
  }
}

export function chalkWordButton(
  chalk: Chalk,
  label: string,
  key: string,
  center: Point2,
  color: Color,
  action: string,
  context: Context,
  hoverSince: Map<string, number>,
  emphasized = false,
): Button {
  const [width, height] = chalk.size(label, 1.15, 2, "script");
  const [x, y] = center;
  const button = new Button(x - width / 2 - 24, y - height - 24, width + 48, height + 64, action);
  const hovered = button.contains(context.mouse);
  if (hovered) {
    if (!hoverSince.has(action)) {
      hoverSince.set(action, context.nowMs);
    }
  } else {
    hoverSince.delete(action);
  }
  chalk.text(label, [x, y], 1.15, emphasized ? YELLOW : color, 2, "script", "center");
  chalk.text(key, [x, y + 30], 0.6, DIM, 1, "sans", "center");
  if (hovered || emphasized) {
    const progress = hovered ? ringProgress(hoverSince.get(action), context.nowMs) : 1;
    chalk.ring([x, y - height / 2 + 12], [width / 2 + 26, height / 2 + 34], YELLOW, progress, 2);
  }
  return button;
}
