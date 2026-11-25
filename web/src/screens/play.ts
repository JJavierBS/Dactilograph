import { ALPHABET, isDynamic } from "../alphabet";
import {
  BOARD,
  type Chalk,
  type Color,
  context2d,
  DIM,
  FAINT,
  fitHand,
  gaussianBlur,
  newCanvas,
  PINK,
  type Point2,
  WHITE,
  YELLOW,
} from "../chalk";
import { BUBBLE_RADIUS, BubbleGame, MISS_DURATION_MS, POP_DURATION_MS, START_LIVES } from "../game";
import { browserStore } from "../progress";
import {
  type Button,
  type CanvasScreen,
  type Context,
  chalkUserHand,
  chalkWordButton,
  letterColor,
  letterScale,
  ringProgress,
} from "../ui";

const ARENA_TOP_LEFT = [240, 36] as const;
const ARENA_SIZE = [800, 600] as const;
const CAMERA_WEIGHT = 0.6;
const ARENA_FEATHER = 40;
const GAME_OVER_FADE = 0.65;
const LEFT_X = 60;
const RIGHT_X = 1148;
const HELP_SIZE = 170;
const HELP_X = 135;
const BURST_RAYS = 8;
const RESTART_KEY = "r";
const MAX_STEP_MS = 100;

function featherMask(weight: number, color: Color): HTMLCanvasElement {
  const [width, height] = ARENA_SIZE;
  const alpha = new Float32Array(width * height);
  for (let y = ARENA_FEATHER; y < height - ARENA_FEATHER; y++) {
    alpha.fill(1, y * width + ARENA_FEATHER, y * width + width - ARENA_FEATHER);
  }
  gaussianBlur(alpha, width, height, ARENA_FEATHER / 2);
  const canvas = newCanvas(width, height);
  const context = context2d(canvas);
  const image = context.createImageData(width, height);
  for (let index = 0; index < alpha.length; index++) {
    image.data.set([color[0], color[1], color[2], alpha[index]! * weight * 255], index * 4);
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

function faded(color: Color, fade: number): Color {
  return [color[0] * fade, color[1] * fade, color[2] * fade];
}

export class PlayScreen implements CanvasScreen {
  readonly trayActive = "jugar";
  readonly usesCamera = true;
  private readonly cameraMask = featherMask(CAMERA_WEIGHT, [0, 0, 0]);
  private readonly overMask = featherMask(GAME_OVER_FADE, BOARD);
  private readonly cameraLayer = newCanvas(ARENA_SIZE[0], ARENA_SIZE[1]);
  private readonly hoverSince = new Map<string, number>();
  private game: BubbleGame | null = null;
  private clockMs = 0;
  private lastNowMs: number | null = null;
  private newRecord = false;
  private previousRecord = 0;
  private overSince: number | null = null;

  enter(): void {
    this.game = null;
    this.lastNowMs = null;
  }

  keyAction(key: string): string | null {
    return key === RESTART_KEY ? "otra" : null;
  }

  handle(action: string, context: Context): string | null {
    if (action === "otra") {
      this.start(context);
      return null;
    }
    return action;
  }

  update(context: Context): void {
    const stepMs =
      this.lastNowMs === null ? 0 : Math.min(context.nowMs - this.lastNowMs, MAX_STEP_MS);
    this.lastNowMs = context.nowMs;
    this.clockMs += stepMs;
    if (this.game === null) {
      this.start(context);
    }
    const game = this.game!;
    for (const letter of context.reading?.confirmed ?? []) {
      game.tryPop(letter, this.clockMs);
    }
    game.update(this.clockMs);
    if (game.isOver && this.overSince === null) {
      this.overSince = context.nowMs;
      this.newRecord = context.progress.recordScore(game.score);
      if (this.newRecord) {
        context.progress.save(browserStore());
      }
    }
  }

  draw(canvas: CanvasRenderingContext2D, chalk: Chalk, context: Context): Button[] {
    const game = this.game;
    if (game === null) {
      return [];
    }
    this.drawArena(canvas, context, game);
    chalkUserHand(chalk, context.reading, ARENA_TOP_LEFT, ARENA_SIZE);
    this.drawBubbles(chalk, game);
    this.drawScores(chalk, game, context);
    this.drawSide(chalk, game, context);
    return game.isOver ? this.drawGameOver(chalk, game, context) : [];
  }

  private start(context: Context): void {
    this.game = new BubbleGame(ALPHABET, Math.random, this.clockMs);
    this.previousRecord = context.progress.highScore;
    this.newRecord = false;
    this.overSince = null;
  }

  private drawArena(canvas: CanvasRenderingContext2D, context: Context, game: BubbleGame): void {
    const [x, y] = ARENA_TOP_LEFT;
    const [width, height] = ARENA_SIZE;
    if (context.frame !== null) {
      const layer = context2d(this.cameraLayer);
      layer.globalCompositeOperation = "source-over";
      layer.clearRect(0, 0, width, height);
      layer.drawImage(context.frame, 0, 0, width, height);
      layer.globalCompositeOperation = "destination-in";
      layer.drawImage(this.cameraMask, 0, 0);
      canvas.drawImage(this.cameraLayer, x, y);
    }
    if (game.isOver) {
      canvas.drawImage(this.overMask, x, y);
    }
  }

  private toArena(x: number, y: number): Point2 {
    return [
      Math.round(ARENA_TOP_LEFT[0] + x * ARENA_SIZE[0]),
      Math.round(ARENA_TOP_LEFT[1] + y * ARENA_SIZE[1]),
    ];
  }

  private drawBubbles(chalk: Chalk, game: BubbleGame): void {
    const radius = Math.round(BUBBLE_RADIUS * ARENA_SIZE[1]);
    for (const bubble of game.bubbles) {
      const [cx, cy] = this.toArena(bubble.x, bubble.y);
      const color = letterColor(bubble.letter);
      chalk.circle([cx, cy], radius, color, 3);
      chalk.ring([cx - 16, cy - 18], [14, 9], color, 0.3, 2, -40);
      const scale = letterScale(bubble.letter, 1.7);
      chalk.text(bubble.letter, [cx, cy + 20], scale, color, 3, "script", "center");
    }
    for (const pop of game.pops) {
      const [cx, cy] = this.toArena(pop.x, pop.y);
      const progress = (this.clockMs - pop.poppedMs) / POP_DURATION_MS;
      const inner = radius * (0.6 + progress);
      const outer = radius * (1 + 1.2 * progress);
      const color = letterColor(pop.letter);
      for (let ray = 0; ray < BURST_RAYS; ray++) {
        const angle = (2 * Math.PI * ray) / BURST_RAYS;
        chalk.line(
          [cx + inner * Math.cos(angle), cy + inner * Math.sin(angle)],
          [cx + outer * Math.cos(angle), cy + outer * Math.sin(angle)],
          color,
          3,
        );
      }
    }
    for (const miss of game.misses) {
      const [cx, top] = this.toArena(miss.x, 0);
      const center = [cx, top + radius + 6] as const;
      const fade = Math.max(0, 1 - (this.clockMs - miss.poppedMs) / MISS_DURATION_MS);
      chalk.circle(center, radius, faded(FAINT, fade), 2);
      chalk.cross(center, Math.round(radius * 0.75), faded(PINK, fade), 3);
    }
  }

  private drawScores(chalk: Chalk, game: BubbleGame, context: Context): void {
    chalk.text(`Puntos ${game.score}`, [LEFT_X, 100], 1.3, WHITE, 2, "script");
    chalk.text(`Nivel ${game.level}`, [LEFT_X, 150], 1.3, WHITE, 2, "script");
    if (this.newRecord) {
      const [labelWidth] = chalk.text("récord", [LEFT_X, 192], 0.65, DIM, 1);
      const oldX = LEFT_X + labelWidth + 10;
      const [width, height] = chalk.text(String(this.previousRecord), [oldX, 192], 0.65, DIM, 1);
      const middle = 192 - height / 2;
      chalk.line([oldX - 4, middle + 4], [oldX + width + 4, middle - 4], PINK, 2);
      const newX = oldX + width + 14;
      chalk.text(String(context.progress.highScore), [newX, 194], 0.9, YELLOW, 2, "script");
    } else {
      chalk.text(`récord ${context.progress.highScore}`, [LEFT_X, 192], 0.65, DIM, 1);
    }
    this.drawHelp(chalk, game, context);
  }

  private drawSide(chalk: Chalk, game: BubbleGame, context: Context): void {
    chalk.text("vidas", [RIGHT_X, 80], 0.6, DIM, 1, "sans", "center");
    for (let life = 0; life < START_LIVES; life++) {
      chalk.circle([RIGHT_X - 44 + life * 44, 112], 13, PINK, life < game.lives ? -1 : 2);
    }
    if (game.isOver) {
      return;
    }
    const prediction = context.reading?.prediction ?? null;
    chalk.text("haces", [RIGHT_X, 190], 0.6, DIM, 1, "sans", "center");
    if (prediction === null) {
      chalk.text("-", [RIGHT_X, 270], 2.6, DIM, 4, "script", "center");
    } else {
      const scale = letterScale(prediction.letter, 2.6);
      chalk.text(prediction.letter, [RIGHT_X, 270], scale, YELLOW, 4, "script", "center");
    }
  }

  private drawHelp(chalk: Chalk, game: BubbleGame, context: Context): void {
    if (game.isOver) {
      return;
    }
    const letter = game.helpLetter(this.clockMs);
    const reference = letter === null ? undefined : context.references[letter];
    if (letter === null || reference === undefined) {
      return;
    }
    chalk.text("así se hace la", [HELP_X, 300], 0.9, WHITE, 2, "script", "center");
    chalk.text(letter, [HELP_X, 360], letterScale(letter, 1.8), YELLOW, 3, "script", "center");
    if (isDynamic(letter)) {
      chalk.text("con movimiento", [HELP_X, 388], 0.55, DIM, 1, "sans", "center");
    }
    const topLeft = [HELP_X - HELP_SIZE / 2, 395] as const;
    chalk.hand(fitHand(reference, topLeft, HELP_SIZE), WHITE, 2);
  }

  private drawGameOver(chalk: Chalk, game: BubbleGame, context: Context): Button[] {
    const centerX = ARENA_TOP_LEFT[0] + ARENA_SIZE[0] / 2;
    chalk.text("Fin de la partida", [centerX, 210], 2.6, WHITE, 4, "script", "center");
    const points = game.score === 1 ? "1 punto" : `${game.score} puntos`;
    chalk.text(points, [centerX, 300], 1.6, YELLOW, 3, "script", "center");
    if (this.newRecord) {
      chalk.text("¡nuevo récord!", [centerX, 360], 1.2, PINK, 2, "script", "center");
      const progress = ringProgress(this.overSince, context.nowMs);
      chalk.ring([centerX, 350], [130, 36], PINK, progress, 2);
    }
    return [
      chalkWordButton(
        chalk,
        "otra partida",
        "r",
        [centerX - 150, 480],
        WHITE,
        "otra",
        context,
        this.hoverSince,
        true,
      ),
      chalkWordButton(
        chalk,
        "menú",
        "Esc",
        [centerX + 170, 480],
        WHITE,
        "menu",
        context,
        this.hoverSince,
      ),
    ];
  }
}
