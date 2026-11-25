import type { Reading } from "./liveRecognizer";
import { element, formatConfidence, type Screen } from "./screen";

const HISTORY_LENGTH = 6;

export class RecognizeScreen implements Screen {
  readonly name = "reconocer";
  readonly element = element<HTMLElement>("screen-reconocer");
  readonly usesCamera = true;
  private readonly letterText = element<HTMLElement>("letter");
  private readonly confidenceText = element<HTMLElement>("confidence");
  private readonly historyList = element<HTMLOListElement>("history");
  private readonly history: string[] = [];

  enter(): void {}

  leave(): void {}

  handleKey(): boolean {
    return false;
  }

  update(reading: Reading): void {
    const { prediction } = reading;
    this.letterText.textContent = prediction?.letter ?? "–";
    this.confidenceText.textContent =
      prediction === null ? "no veo tu mano" : formatConfidence(prediction.confidence);
    if (reading.confirmed.length > 0) {
      this.history.push(...reading.confirmed);
      this.history.splice(0, Math.max(0, this.history.length - HISTORY_LENGTH));
      this.showHistory();
    }
  }

  private showHistory(): void {
    this.historyList.replaceChildren(
      ...this.history.map((letter, index) => {
        const item = document.createElement("li");
        item.textContent = letter;
        item.classList.toggle("newest", index === this.history.length - 1);
        return item;
      }),
    );
  }
}
