import { ALPHABET, isDynamic } from "./alphabet";
import { drawSkeleton, fitHand } from "./drawing";
import { LearnSession } from "./learning";
import type { Reading } from "./liveRecognizer";
import type { ReferenceHands } from "./menuScreen";
import { browserStore, type Progress } from "./progress";
import { element, formatConfidence, type Screen } from "./screen";

const NEXT_KEYS = new Set([" ", "ArrowRight"]);
const PREVIOUS_KEYS = new Set(["Backspace", "ArrowLeft"]);
const CHALK = "#ecebe2";

export class LearnScreen implements Screen {
  readonly name = "aprender";
  readonly element = element<HTMLElement>("screen-aprender");
  readonly usesCamera = true;
  private readonly session = new LearnSession(ALPHABET);
  private readonly guide = element<HTMLOListElement>("guide");
  private readonly guideCount = element<HTMLElement>("guide-count");
  private readonly targetLetter = element<HTMLElement>("target-letter");
  private readonly targetKind = element<HTMLElement>("target-kind");
  private readonly targetNote = element<HTMLElement>("target-note");
  private readonly reference = element<HTMLCanvasElement>("reference");
  private readonly readingBlock = element<HTMLElement>("learn-reading");
  private readonly readingLetter = element<HTMLElement>("learn-letter");
  private readonly readingConfidence = element<HTMLElement>("learn-confidence");
  private readonly successText = element<HTMLElement>("learn-success");
  private readonly nextButton = element<HTMLButtonElement>("next");
  private readonly guideButtons = new Map<string, HTMLButtonElement>();

  constructor(
    private readonly references: ReferenceHands,
    private readonly progress: Progress,
  ) {
    this.guide.replaceChildren(
      ...ALPHABET.map((letter) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = letter;
        button.addEventListener("click", () => this.change(() => this.session.select(letter)));
        this.guideButtons.set(letter, button);
        const item = document.createElement("li");
        item.append(button);
        return item;
      }),
    );
    this.nextButton.addEventListener("click", () => this.change(() => this.session.next()));
    element<HTMLButtonElement>("previous").addEventListener("click", () =>
      this.change(() => this.session.previous()),
    );
  }

  enter(): void {
    const pending = ALPHABET.find((letter) => !this.progress.learned.has(letter));
    if (pending !== undefined && this.progress.learned.has(this.session.target)) {
      this.session.select(pending);
    }
    this.render();
  }

  leave(): void {}

  handleKey(event: KeyboardEvent): boolean {
    if (NEXT_KEYS.has(event.key)) {
      this.change(() => this.session.next());
      return true;
    }
    if (PREVIOUS_KEYS.has(event.key)) {
      this.change(() => this.session.previous());
      return true;
    }
    return false;
  }

  update(reading: Reading, nowMs: number): void {
    if (this.session.check(reading.confirmed, nowMs)) {
      this.progress.learn(this.session.target);
      this.progress.save(browserStore());
      this.render();
      return;
    }
    if (this.session.succeeded) {
      return;
    }
    const { prediction } = reading;
    this.readingLetter.textContent = prediction?.letter ?? "–";
    this.readingLetter.classList.toggle(
      "success-letter",
      prediction?.letter === this.session.target,
    );
    this.readingConfidence.textContent =
      prediction === null
        ? "no veo tu mano"
        : `seguridad ${formatConfidence(prediction.confidence)}`;
  }

  private change(action: () => void): void {
    action();
    this.render();
  }

  private render(): void {
    const target = this.session.target;
    for (const [letter, button] of this.guideButtons) {
      button.classList.toggle("learned", this.progress.learned.has(letter));
      button.classList.toggle("current", letter === target);
    }
    const learned = ALPHABET.filter((letter) => this.progress.learned.has(letter)).length;
    this.guideCount.textContent = `${this.session.index + 1} de ${ALPHABET.length} · ${learned} aprendidas`;

    const dynamic = isDynamic(target);
    this.targetLetter.textContent = target;
    this.targetKind.textContent = dynamic ? "con movimiento" : "estática";
    this.targetNote.textContent = dynamic ? "empieza así y haz el gesto" : "así se hace";
    this.drawReference(target);

    const succeeded = this.session.succeeded;
    this.successText.hidden = !succeeded;
    this.readingBlock.hidden = succeeded;
    this.nextButton.classList.toggle("emphasized", succeeded);
    if (!succeeded) {
      this.readingLetter.textContent = "–";
      this.readingLetter.classList.remove("success-letter");
      this.readingConfidence.textContent = "";
    }
  }

  private drawReference(letter: string): void {
    const context = this.reference.getContext("2d");
    if (context === null) {
      return;
    }
    context.clearRect(0, 0, this.reference.width, this.reference.height);
    const points = this.references[letter];
    if (points !== undefined) {
      drawSkeleton(context, fitHand(points, this.reference.width), CHALK, 3);
    }
  }
}
