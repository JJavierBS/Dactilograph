import { HandLandmarker } from "@mediapipe/tasks-vision";

export const WIDTH = 1280;
export const HEIGHT = 720;
export const FRAME = 22;
export const TRAY_HEIGHT = 70;
export const BOARD_BOTTOM = HEIGHT - TRAY_HEIGHT;

export type Color = readonly [number, number, number];
export type Point2 = readonly [number, number];

export const BOARD: Color = [33, 53, 45];
export const WHITE: Color = [236, 235, 226];
export const YELLOW: Color = [242, 211, 91];
export const PINK: Color = [238, 142, 164];
export const BLUE: Color = [140, 195, 230];
export const DIM: Color = [160, 170, 165];
export const FAINT: Color = [105, 120, 110];
export const WOOD: Color = [138, 90, 50];
export const WOOD_DARK: Color = [94, 59, 31];
export const TAPE: Color = [220, 215, 205];

export type Font = "script" | "sans";
export type Align = "left" | "center" | "right";
export type Random = () => number;

const FONT_PX: Record<Font, number> = { script: 26, sans: 28.3 };
const ASCENT = 0.935;
const SMEARS = 9;
const SMEAR_BLUR = 35;
const SMEAR_STRENGTH = 14;
const BOARD_NOISE = 3;
const GRAIN_MIN = 0.45;
const SHADOW_OFFSET: Point2 = [6, 8];
const SHADOW_BLUR = 16;
const SHADOW_STRENGTH = 0.55;
const TAPE_SIZE: Point2 = [84, 26];
const BONES = HandLandmarker.HAND_CONNECTIONS.map(({ start, end }) => [start, end] as const);

export function rgb(color: Color, alpha = 1): string {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

export function chalkStyle(color: Color): string {
  const peak = Math.max(color[0], color[1], color[2], 1);
  const boost = 255 / peak;
  return rgb([color[0] * boost, color[1] * boost, color[2] * boost], peak / 255);
}

export function fontFor(scale: number, thickness: number, font: Font): string {
  const style = font === "script" ? "italic" : "normal";
  const weight = thickness <= 1 ? 400 : 550;
  return `${style} ${weight} ${(FONT_PX[font] * scale).toFixed(2)}px Rubik, sans-serif`;
}

export function textSize(
  context: CanvasRenderingContext2D,
  text: string,
  scale: number,
  thickness = 2,
  font: Font = "sans",
): Point2 {
  context.save();
  context.font = fontFor(scale, thickness, font);
  const width = context.measureText(text).width;
  context.restore();
  return [Math.round(width), Math.round(FONT_PX[font] * scale * ASCENT)];
}

export function putText(
  context: CanvasRenderingContext2D,
  text: string,
  origin: Point2,
  scale: number,
  color: Color,
  thickness = 2,
  font: Font = "sans",
  align: Align = "left",
  style: string = rgb(color),
): Point2 {
  const size = textSize(context, text, scale, thickness, font);
  let x = origin[0];
  if (align === "center") {
    x -= size[0] / 2;
  } else if (align === "right") {
    x -= size[0];
  }
  context.save();
  context.font = fontFor(scale, thickness, font);
  context.fillStyle = style;
  context.textBaseline = "alphabetic";
  context.fillText(text, x, origin[1]);
  context.restore();
  return size;
}

function gaussian(random: Random): number {
  const u = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

function boxBlur(values: Float32Array, width: number, height: number, radius: number): void {
  const line = new Float32Array(Math.max(width, height));
  const pass = (length: number, count: number, index: (i: number, j: number) => number) => {
    for (let j = 0; j < count; j++) {
      let sum = 0;
      for (let i = -radius; i <= radius; i++) {
        sum += values[index(Math.min(Math.max(i, 0), length - 1), j)]!;
      }
      for (let i = 0; i < length; i++) {
        line[i] = sum / (2 * radius + 1);
        const add = Math.min(i + radius + 1, length - 1);
        const remove = Math.max(i - radius, 0);
        sum += values[index(add, j)]! - values[index(remove, j)]!;
      }
      for (let i = 0; i < length; i++) {
        values[index(i, j)] = line[i]!;
      }
    }
  };
  pass(width, height, (i, j) => j * width + i);
  pass(height, width, (i, j) => i * width + j);
}

export function gaussianBlur(
  values: Float32Array,
  width: number,
  height: number,
  sigma: number,
): void {
  const radius = Math.max(1, Math.round(Math.sqrt((12 * sigma * sigma) / 3 + 1) / 2));
  for (let pass = 0; pass < 3; pass++) {
    boxBlur(values, width, height, radius);
  }
}

export function newCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("The browser cannot draw on a canvas");
  }
  return context;
}

export function boardBackground(random: Random): HTMLCanvasElement {
  const smears = newCanvas(WIDTH, HEIGHT);
  const smearContext = context2d(smears);
  smearContext.fillStyle = "#fff";
  for (let index = 0; index < SMEARS; index++) {
    const x = 100 + random() * (WIDTH - 200);
    const y = 80 + random() * (BOARD_BOTTOM - 160);
    const rx = 120 + random() * 200;
    const ry = 30 + random() * 40;
    const angle = ((random() * 30 - 15) * Math.PI) / 180;
    smearContext.beginPath();
    smearContext.ellipse(x, y, rx, ry, angle, 0, Math.PI * 2);
    smearContext.fill();
  }
  const smearPixels = smearContext.getImageData(0, 0, WIDTH, HEIGHT).data;
  const smear = new Float32Array(WIDTH * HEIGHT);
  for (let index = 0; index < smear.length; index++) {
    smear[index] = smearPixels[index * 4]! / 255;
  }
  gaussianBlur(smear, WIDTH, HEIGHT, SMEAR_BLUR);

  const board = newCanvas(WIDTH, HEIGHT);
  const boardContext = context2d(board);
  const image = boardContext.createImageData(WIDTH, HEIGHT);
  for (let index = 0; index < smear.length; index++) {
    const light = smear[index]! * SMEAR_STRENGTH;
    for (let channel = 0; channel < 3; channel++) {
      const value = BOARD[channel]! + light + gaussian(random) * BOARD_NOISE;
      image.data[index * 4 + channel] = value;
    }
    image.data[index * 4 + 3] = 255;
  }
  boardContext.putImageData(image, 0, 0);
  drawWood(boardContext, random);
  return board;
}

export function chalkGrain(random: Random): HTMLCanvasElement {
  const grain = new Float32Array(WIDTH * HEIGHT);
  for (let index = 0; index < grain.length; index++) {
    grain[index] = GRAIN_MIN + random() * (1 - GRAIN_MIN);
  }
  gaussianBlur(grain, WIDTH, HEIGHT, 0.6);
  const canvas = newCanvas(WIDTH, HEIGHT);
  const context = context2d(canvas);
  const image = context.createImageData(WIDTH, HEIGHT);
  for (let index = 0; index < grain.length; index++) {
    image.data[index * 4 + 3] = grain[index]! * 255;
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

function woodGrain(
  context: CanvasRenderingContext2D,
  random: Random,
  box: readonly [number, number, number, number],
  horizontal: boolean,
): void {
  const [x0, y0, width, height] = box;
  const [length, across] = horizontal ? [width, height] : [height, width];
  context.save();
  context.beginPath();
  context.rect(x0, y0, width, height);
  context.clip();
  context.lineWidth = 1;
  let position = random() * 3;
  while (position < across) {
    const shade = 0.82 + random() * 0.26;
    const phase = random() * 2 * Math.PI;
    const amplitude = 0.4 + random() * 1.2;
    const period = 60 + random() * 100;
    context.strokeStyle = rgb([WOOD[0] * shade, WOOD[1] * shade, WOOD[2] * shade]);
    context.beginPath();
    for (let along = 0; along <= length + 8; along += 8) {
      const offset = position + amplitude * Math.sin(along / period + phase);
      const [x, y] = horizontal ? [x0 + along, y0 + offset] : [x0 + offset, y0 + along];
      if (along === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
    context.stroke();
    position += 2.5 + random() * 3.5;
  }
  context.restore();
}

function drawWood(context: CanvasRenderingContext2D, random: Random): void {
  const inner = FRAME / 2;
  context.fillStyle = rgb(WOOD);
  context.fillRect(0, 0, WIDTH, inner);
  context.fillRect(0, 0, inner, HEIGHT);
  context.fillRect(WIDTH - inner, 0, inner, HEIGHT);
  context.fillRect(0, BOARD_BOTTOM, WIDTH, HEIGHT - BOARD_BOTTOM);
  woodGrain(context, random, [0, 0, WIDTH, inner], true);
  woodGrain(context, random, [0, 0, inner, BOARD_BOTTOM], false);
  woodGrain(context, random, [WIDTH - inner, 0, inner, BOARD_BOTTOM], false);
  woodGrain(context, random, [0, BOARD_BOTTOM + 4, WIDTH, HEIGHT - BOARD_BOTTOM - 4], true);
  context.strokeStyle = rgb(WOOD_DARK);
  context.lineWidth = 2;
  context.strokeRect(inner, inner, WIDTH - 2 * inner, BOARD_BOTTOM - inner);
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(0, BOARD_BOTTOM);
  context.lineTo(WIDTH, BOARD_BOTTOM);
  context.stroke();
}

export function pinPhoto(
  context: CanvasRenderingContext2D,
  photo: CanvasImageSource,
  topLeft: Point2,
  size: Point2,
): void {
  const [x, y] = topLeft;
  const [width, height] = size;
  context.save();
  context.shadowColor = `rgba(0, 0, 0, ${SHADOW_STRENGTH})`;
  context.shadowBlur = SHADOW_BLUR;
  context.shadowOffsetX = SHADOW_OFFSET[0];
  context.shadowOffsetY = SHADOW_OFFSET[1];
  context.fillStyle = "#000";
  context.fillRect(x, y, width, height);
  context.restore();
  context.drawImage(photo, x, y, width, height);
  for (const [cx, angle] of [
    [x + 18, -22],
    [x + width - 18, 20],
  ] as const) {
    context.save();
    context.translate(cx, y + 2);
    context.rotate((angle * Math.PI) / 180);
    context.fillStyle = rgb(TAPE, 0.7);
    context.fillRect(-TAPE_SIZE[0] / 2, -TAPE_SIZE[1] / 2, TAPE_SIZE[0], TAPE_SIZE[1]);
    context.restore();
  }
}

export class Chalk {
  readonly layer: HTMLCanvasElement = newCanvas(WIDTH, HEIGHT);
  readonly context: CanvasRenderingContext2D = context2d(this.layer);

  constructor(private readonly grain: HTMLCanvasElement) {}

  clear(): void {
    this.context.globalCompositeOperation = "source-over";
    this.context.clearRect(0, 0, WIDTH, HEIGHT);
  }

  apply(target: CanvasRenderingContext2D): void {
    this.context.globalCompositeOperation = "destination-in";
    this.context.drawImage(this.grain, 0, 0);
    this.context.globalCompositeOperation = "source-over";
    target.drawImage(this.layer, 0, 0);
  }

  text(
    text: string,
    origin: Point2,
    scale: number,
    color: Color,
    thickness = 2,
    font: Font = "sans",
    align: Align = "left",
  ): Point2 {
    return putText(
      this.context,
      text,
      origin,
      scale,
      color,
      thickness,
      font,
      align,
      chalkStyle(color),
    );
  }

  size(text: string, scale: number, thickness = 2, font: Font = "sans"): Point2 {
    return textSize(this.context, text, scale, thickness, font);
  }

  line(start: Point2, end: Point2, color: Color, thickness = 2): void {
    const context = this.context;
    context.strokeStyle = chalkStyle(color);
    context.lineWidth = thickness;
    context.lineCap = "round";
    context.beginPath();
    context.moveTo(start[0], start[1]);
    context.lineTo(end[0], end[1]);
    context.stroke();
  }

  circle(center: Point2, radius: number, color: Color, thickness = 2): void {
    const context = this.context;
    context.beginPath();
    context.arc(center[0], center[1], radius, 0, Math.PI * 2);
    if (thickness < 0) {
      context.fillStyle = chalkStyle(color);
      context.fill();
    } else {
      context.strokeStyle = chalkStyle(color);
      context.lineWidth = thickness;
      context.stroke();
    }
  }

  ring(center: Point2, axes: Point2, color: Color, progress = 1, thickness = 3, tilt = -4): void {
    const sweep = 345 * Math.min(Math.max(progress, 0), 1);
    if (sweep <= 0) {
      return;
    }
    const context = this.context;
    const start = (-100 * Math.PI) / 180;
    context.strokeStyle = chalkStyle(color);
    context.lineWidth = thickness;
    context.lineCap = "round";
    context.beginPath();
    context.ellipse(
      center[0],
      center[1],
      axes[0],
      axes[1],
      (tilt * Math.PI) / 180,
      start,
      start + (sweep * Math.PI) / 180,
    );
    context.stroke();
  }

  cross(center: Point2, size: number, color: Color, thickness = 3): void {
    const [x, y] = center;
    this.line([x - size, y - size], [x + size, y + size], color, thickness);
    this.line([x + size, y - size], [x - size, y + size], color, thickness);
  }

  hand(points: readonly Point2[], color: Color, thickness = 3): void {
    const context = this.context;
    context.strokeStyle = chalkStyle(color);
    context.fillStyle = chalkStyle(color);
    context.lineWidth = thickness;
    context.lineCap = "round";
    context.beginPath();
    for (const [start, end] of BONES) {
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
      context.arc(x, y, thickness + 1, 0, Math.PI * 2);
      context.fill();
    }
  }
}

export function fitHand(
  points: readonly (readonly number[])[],
  topLeft: Point2,
  size: number,
  margin = 0.08,
): Point2[] {
  const xs = points.map((point) => point[0] ?? 0);
  const ys = points.map((point) => point[1] ?? 0);
  const low = [Math.min(...xs), Math.min(...ys)] as const;
  const range = [Math.max(...xs) - low[0], Math.max(...ys) - low[1]] as const;
  const usable = size * (1 - 2 * margin);
  const scale = usable / Math.max(range[0], range[1], 1e-6);
  return points.map((point): Point2 => [
    Math.round(
      ((point[0] ?? 0) - low[0]) * scale +
        topLeft[0] +
        size * margin +
        (usable - range[0] * scale) / 2,
    ),
    Math.round(
      ((point[1] ?? 0) - low[1]) * scale +
        topLeft[1] +
        size * margin +
        (usable - range[1] * scale) / 2,
    ),
  ]);
}

export function handInBox(
  landmarks: readonly (readonly number[])[],
  topLeft: Point2,
  size: Point2,
): Point2[] {
  return landmarks.map((point): Point2 => [
    Math.round((point[0] ?? 0) * size[0] + topLeft[0]),
    Math.round((point[1] ?? 0) * size[1] + topLeft[1]),
  ]);
}
