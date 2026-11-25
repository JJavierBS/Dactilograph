import { ALPHABET, isDynamic } from "../alphabet";
import { BLUE, type Chalk, type Color, DIM, fitHand, PINK, WHITE, YELLOW } from "../chalk";
import { Button, type CanvasScreen, type Context, letterScale, ringProgress } from "../ui";

const OPTIONS: readonly (readonly [string, string, string, Color, string])[] = [
  ["1", "Aprender", "las 30 letras, una a una", BLUE, "aprender"],
  ["2", "Reconocer", "haz letras y mira cuál ve", WHITE, "reconocer"],
  ["3", "Jugar", "explota burbujas signando", PINK, "jugar"],
];
const OPTION_KEYS: Readonly<Record<string, string>> = {
  "1": "aprender",
  "2": "reconocer",
  "3": "jugar",
};
const FIRST_OPTION_Y = 365;
const OPTION_STEP = 95;
const SHOWCASE_MS = 1600;
const SHOWCASE_TOP_LEFT = [850, 110] as const;
const SHOWCASE_SIZE = 360;

export class MenuScreen implements CanvasScreen {
  readonly trayActive = "menu";
  readonly usesCamera = false;
  private focus = 0;
  private focusSince: number | null = null;

  enter(context: Context): void {
    this.focusSince = context.nowMs;
  }

  keyAction(key: string): string | null {
    return OPTION_KEYS[key] ?? null;
  }

  handle(action: string): string | null {
    return action;
  }

  update(): void {}

  draw(_canvas: CanvasRenderingContext2D, chalk: Chalk, context: Context): Button[] {
    chalk.text("Dactilograph", [90, 190], 4.0, WHITE, 6, "script");
    chalk.line([95, 215], [720, 222], YELLOW, 3);
    chalk.text("el alfabeto dactilológico de la LSE, con tu mano", [100, 268], 0.8, DIM, 1);

    const buttons = OPTIONS.map(
      (option, index) =>
        new Button(70, FIRST_OPTION_Y + index * OPTION_STEP - 62, 690, 84, option[4]),
    );
    const hovered = buttons.findIndex((button) => button.contains(context.mouse));
    if (hovered !== -1 && hovered !== this.focus) {
      this.focus = hovered;
      this.focusSince = context.nowMs;
    }

    OPTIONS.forEach(([key, name, note, color], index) => {
      const y = FIRST_OPTION_Y + index * OPTION_STEP;
      chalk.text(key, [95, y], 1.2, DIM, 2);
      const [width] = chalk.text(name, [140, y], 2.0, color, 3, "script");
      chalk.text(note, [470, y - 8], 0.7, DIM, 1);
      if (index === this.focus) {
        const progress = ringProgress(this.focusSince, context.nowMs);
        chalk.ring([140 + width / 2 - 20, y - 20], [width / 2 + 70, 48], YELLOW, progress);
      }
    });

    this.drawShowcase(chalk, context);
    const learned = ALPHABET.filter((letter) => context.progress.learned.has(letter)).length;
    chalk.text(
      `aprendidas: ${learned} de ${ALPHABET.length}`,
      [1230, 575],
      0.7,
      WHITE,
      1,
      "sans",
      "right",
    );
    chalk.text(
      `récord: ${context.progress.highScore}`,
      [1230, 610],
      0.7,
      WHITE,
      1,
      "sans",
      "right",
    );
    return buttons;
  }

  private drawShowcase(chalk: Chalk, context: Context): void {
    const shown = ALPHABET.filter((letter) => letter in context.references);
    if (shown.length === 0) {
      return;
    }
    const letter = shown[Math.floor(context.nowMs / SHOWCASE_MS) % shown.length]!;
    chalk.hand(fitHand(context.references[letter]!, SHOWCASE_TOP_LEFT, SHOWCASE_SIZE), WHITE, 3);
    const x = SHOWCASE_TOP_LEFT[0] + SHOWCASE_SIZE;
    const y = SHOWCASE_TOP_LEFT[1] + SHOWCASE_SIZE + 30;
    chalk.text(letter, [x, y], letterScale(letter, 3.0), YELLOW, 4, "script", "right");
    if (isDynamic(letter)) {
      chalk.text("con movimiento", [x, y + 32], 0.6, BLUE, 1, "sans", "right");
    }
  }
}
