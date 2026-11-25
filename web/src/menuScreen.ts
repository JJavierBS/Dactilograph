import { ALPHABET } from "./alphabet";
import { drawSkeleton, fitHand } from "./drawing";
import type { Progress } from "./progress";
import { element, type Screen } from "./screen";

export type ReferenceHands = Readonly<Record<string, readonly (readonly number[])[]>>;

const SHOWCASE_MS = 1600;
const CHALK = "#ecebe2";

export class MenuScreen implements Screen {
  readonly name = "menu";
  readonly element = element<HTMLElement>("screen-menu");
  readonly usesCamera = false;
  private readonly canvas = element<HTMLCanvasElement>("showcase");
  private readonly letterText = element<HTMLElement>("showcase-letter");
  private readonly learnedText = element<HTMLElement>("learned-count");
  private timer: number | undefined;
  private shown = 0;

  constructor(
    private readonly references: ReferenceHands,
    private readonly progress: Progress,
  ) {}

  enter(): void {
    const learned = ALPHABET.filter((letter) => this.progress.learned.has(letter)).length;
    this.learnedText.textContent = `aprendidas: ${learned} de ${ALPHABET.length}`;
    this.showNext();
    this.timer = window.setInterval(() => this.showNext(), SHOWCASE_MS);
  }

  leave(): void {
    window.clearInterval(this.timer);
  }

  update(): void {}

  handleKey(): boolean {
    return false;
  }

  private showNext(): void {
    const letter = ALPHABET[this.shown % ALPHABET.length]!;
    this.shown += 1;
    this.letterText.textContent = letter;
    const context = this.canvas.getContext("2d");
    const points = this.references[letter];
    if (context === null || points === undefined) {
      return;
    }
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    drawSkeleton(context, fitHand(points, this.canvas.width), CHALK, 4);
  }
}
