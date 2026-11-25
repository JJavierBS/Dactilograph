import { type Chalk, DIM, pinPhoto, WHITE, YELLOW } from "../chalk";
import {
  type Button,
  type CanvasScreen,
  type Context,
  chalkUserHand,
  formatConfidence,
  letterScale,
  noHandNote,
  ringProgress,
} from "../ui";

const CAMERA_TOP_LEFT = [60, 120] as const;
const CAMERA_SIZE = [680, 510] as const;
const COLUMN_X = 1000;
const HISTORY_LENGTH = 6;
const HISTORY_STEP = 72;

export class RecognizeScreen implements CanvasScreen {
  readonly trayActive = "reconocer";
  readonly usesCamera = true;
  private readonly history: (readonly [string, number])[] = [];

  enter(): void {}

  keyAction(): string | null {
    return null;
  }

  handle(action: string): string | null {
    return action;
  }

  update(context: Context): void {
    for (const letter of context.reading?.confirmed ?? []) {
      this.history.push([letter, context.nowMs]);
    }
    this.history.splice(0, Math.max(0, this.history.length - HISTORY_LENGTH));
  }

  draw(canvas: CanvasRenderingContext2D, chalk: Chalk, context: Context): Button[] {
    chalk.text("Haz cualquier letra delante de la cámara", [60, 78], 1.2, WHITE, 2, "script");
    if (context.frame !== null) {
      pinPhoto(canvas, context.frame, CAMERA_TOP_LEFT, CAMERA_SIZE);
    }
    chalkUserHand(chalk, context.reading, CAMERA_TOP_LEFT, CAMERA_SIZE);
    noHandNote(chalk, context.reading, [CAMERA_TOP_LEFT[0] + 340, CAMERA_TOP_LEFT[1] + 265]);

    chalk.text("veo", [COLUMN_X, 150], 0.6, DIM, 1, "sans", "center");
    const prediction = context.reading?.prediction ?? null;
    if (prediction === null) {
      chalk.text("-", [COLUMN_X, 330], 6.0, DIM, 8, "script", "center");
    } else {
      const scale = letterScale(prediction.letter, 6.0);
      chalk.text(prediction.letter, [COLUMN_X, 330], scale, WHITE, 10, "script", "center");
      chalk.text(
        formatConfidence(prediction.confidence),
        [COLUMN_X, 395],
        1.0,
        WHITE,
        2,
        "sans",
        "center",
      );
      chalk.text("seguridad", [COLUMN_X, 422], 0.5, DIM, 1, "sans", "center");
    }

    chalk.text("lo último que has hecho", [COLUMN_X, 500], 0.9, DIM, 1, "script", "center");
    const startX = COLUMN_X - ((HISTORY_LENGTH - 1) * HISTORY_STEP) / 2;
    this.history.forEach(([letter, madeMs], index) => {
      const x = startX + index * HISTORY_STEP;
      const newest = index === this.history.length - 1;
      const color = newest ? YELLOW : WHITE;
      chalk.text(letter, [x, 585], letterScale(letter, 1.6), color, 3, "script", "center");
      if (newest) {
        chalk.ring([x, 568], [30, 34], YELLOW, ringProgress(madeMs, context.nowMs), 2, 8);
      }
    });
    return [];
  }
}
