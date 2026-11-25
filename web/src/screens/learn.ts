import { ALPHABET, isDynamic } from "../alphabet";
import { BLUE, type Chalk, DIM, FAINT, fitHand, pinPhoto, WHITE, YELLOW } from "../chalk";
import { LearnSession } from "../learning";
import { browserStore } from "../progress";
import {
  Button,
  type CanvasScreen,
  type Context,
  chalkUserHand,
  chalkWordButton,
  formatConfidence,
  letterScale,
  noHandNote,
  ringProgress,
} from "../ui";

const GUIDE_LEFT = 50;
const GUIDE_Y = 64;
const GUIDE_GAP = 15;
const CAMERA_TOP_LEFT = [420, 122] as const;
const CAMERA_SIZE = [640, 480] as const;
const COLUMN_X = 1160;
const LEFT_X = 215;
const REFERENCE_SIZE = 200;
const NEXT_KEYS = new Set([" ", "ArrowRight"]);
const PREVIOUS_KEYS = new Set(["Backspace", "ArrowLeft"]);
const LETTER_ACTION = "letra:";

export class LearnScreen implements CanvasScreen {
  readonly trayActive = "aprender";
  readonly usesCamera = true;
  private readonly session = new LearnSession(ALPHABET);
  private readonly hoverSince = new Map<string, number>();

  enter(context: Context): void {
    const pending = ALPHABET.find((letter) => !context.progress.learned.has(letter));
    if (pending !== undefined && context.progress.learned.has(this.session.target)) {
      this.session.select(pending);
    }
  }

  keyAction(key: string): string | null {
    if (NEXT_KEYS.has(key)) {
      return "siguiente";
    }
    if (PREVIOUS_KEYS.has(key)) {
      return "anterior";
    }
    return null;
  }

  handle(action: string): string | null {
    if (action === "siguiente") {
      this.session.next();
    } else if (action === "anterior") {
      this.session.previous();
    } else if (action.startsWith(LETTER_ACTION)) {
      this.session.select(action.slice(LETTER_ACTION.length));
    } else {
      return action;
    }
    return null;
  }

  update(context: Context): void {
    if (context.reading === null) {
      return;
    }
    if (this.session.check(context.reading.confirmed, context.nowMs)) {
      context.progress.learn(this.session.target);
      context.progress.save(browserStore());
    }
  }

  draw(canvas: CanvasRenderingContext2D, chalk: Chalk, context: Context): Button[] {
    const buttons = this.drawGuide(chalk, context);
    this.drawTarget(chalk, context);
    if (context.frame !== null) {
      pinPhoto(canvas, context.frame, CAMERA_TOP_LEFT, CAMERA_SIZE);
    }
    chalkUserHand(chalk, context.reading, CAMERA_TOP_LEFT, CAMERA_SIZE);
    noHandNote(chalk, context.reading, [CAMERA_TOP_LEFT[0] + 320, CAMERA_TOP_LEFT[1] + 250]);
    this.drawVerdict(chalk, context);
    const succeeded = this.session.succeeded;
    buttons.push(
      chalkWordButton(
        chalk,
        "siguiente",
        "espacio",
        [COLUMN_X, 560],
        WHITE,
        "siguiente",
        context,
        this.hoverSince,
        succeeded,
      ),
      chalkWordButton(
        chalk,
        "anterior",
        "retroceso",
        [COLUMN_X, 470],
        DIM,
        "anterior",
        context,
        this.hoverSince,
      ),
    );
    return buttons;
  }

  private drawGuide(chalk: Chalk, context: Context): Button[] {
    const buttons: Button[] = [];
    let x = GUIDE_LEFT;
    for (const letter of ALPHABET) {
      const learned = context.progress.learned.has(letter);
      const isTarget = letter === this.session.target;
      const [width, height] = chalk.size(letter, 0.7, 2);
      const button = new Button(
        x - 6,
        GUIDE_Y - height - 10,
        width + 12,
        height + 30,
        `${LETTER_ACTION}${letter}`,
      );
      const color = isTarget ? YELLOW : learned || button.contains(context.mouse) ? WHITE : DIM;
      chalk.text(letter, [x, GUIDE_Y], 0.7, color, 2);
      const dot = [x + width / 2, GUIDE_Y + 14] as const;
      chalk.circle(dot, 3, learned ? color : FAINT, learned ? -1 : 1);
      if (isTarget) {
        chalk.ring([x + width / 2, GUIDE_Y - height / 2 + 2], [width / 2 + 12, 22], YELLOW, 1, 2);
      }
      buttons.push(button);
      x += width + GUIDE_GAP;
    }
    const learnedCount = ALPHABET.filter((letter) => context.progress.learned.has(letter)).length;
    const position = `${this.session.index + 1} de ${ALPHABET.length}  ·  ${learnedCount} aprendidas`;
    chalk.text(position, [1240, GUIDE_Y], 0.55, DIM, 1, "sans", "right");
    chalk.line([40, GUIDE_Y + 28], [1240, GUIDE_Y + 32], FAINT, 1);
    return buttons;
  }

  private drawTarget(chalk: Chalk, context: Context): void {
    const target = this.session.target;
    chalk.text(target, [LEFT_X, 330], letterScale(target, 8.5), WHITE, 14, "script", "center");
    const dynamic = isDynamic(target);
    chalk.text(
      dynamic ? "con movimiento" : "estática",
      [LEFT_X, 385],
      1.3,
      BLUE,
      2,
      "script",
      "center",
    );
    const reference = context.references[target];
    if (reference !== undefined) {
      const topLeft = [LEFT_X - REFERENCE_SIZE / 2, 395] as const;
      chalk.hand(fitHand(reference, topLeft, REFERENCE_SIZE), WHITE, 3);
    }
    const note = dynamic ? "empieza así y haz el gesto" : "así se hace";
    chalk.text(note, [LEFT_X, 625], 0.9, DIM, 1, "script", "center");
  }

  private drawVerdict(chalk: Chalk, context: Context): void {
    const target = this.session.target;
    if (this.session.succeeded) {
      chalk.text(target, [COLUMN_X, 250], letterScale(target, 3.2), YELLOW, 5, "script", "center");
      const progress = ringProgress(this.session.successMs, context.nowMs);
      chalk.ring([COLUMN_X, 220], [62, 66], YELLOW, progress, 3, 8);
      chalk.text("¡Bien!", [COLUMN_X, 340], 1.5, YELLOW, 2, "script", "center");
      return;
    }
    chalk.text("veo", [COLUMN_X, 150], 0.6, DIM, 1, "sans", "center");
    const prediction = context.reading?.prediction ?? null;
    if (prediction === null) {
      chalk.text("-", [COLUMN_X, 250], 3.2, DIM, 4, "script", "center");
      return;
    }
    const color = prediction.letter === target ? WHITE : DIM;
    const scale = letterScale(prediction.letter, 3.2);
    chalk.text(prediction.letter, [COLUMN_X, 250], scale, color, 5, "script", "center");
    const confidence = formatConfidence(prediction.confidence);
    chalk.text(confidence, [COLUMN_X, 315], 0.9, WHITE, 2, "sans", "center");
    chalk.text("seguridad", [COLUMN_X, 340], 0.5, DIM, 1, "sans", "center");
  }
}
