import { ALPHABET, isDynamic } from "./alphabet";
import { drawSkeleton, fitHand } from "./drawing";
import { BUBBLE_RADIUS, BubbleGame, MISS_DURATION_MS, POP_DURATION_MS, START_LIVES } from "./game";
import type { Reading } from "./liveRecognizer";
import type { ReferenceHands } from "./menuScreen";
import { browserStore, type Progress } from "./progress";
import { element, type Screen } from "./screen";

const LETTER_COLORS = ["#ecebe2", "#f2d35b", "#ee8ea4", "#8cc3e6"];
const MISS_COLOR = "#ee8ea4";
const CHALK = "#ecebe2";
const MAX_STEP_MS = 100;
const BURST_RAYS = 8;
const RESTART_KEY = "r";

function letterColor(letter: string): string {
  return LETTER_COLORS[Math.max(0, ALPHABET.indexOf(letter)) % LETTER_COLORS.length]!;
}

export class PlayScreen implements Screen {
  readonly name = "jugar";
  readonly element = element<HTMLElement>("screen-jugar");
  readonly usesCamera = true;
  private readonly canvas = element<HTMLCanvasElement>("bubbles");
  private readonly scoreText = element<HTMLElement>("score");
  private readonly levelText = element<HTMLElement>("level");
  private readonly recordText = element<HTMLElement>("record");
  private readonly livesText = element<HTMLElement>("lives");
  private readonly doingText = element<HTMLElement>("doing");
  private readonly help = element<HTMLElement>("help");
  private readonly helpLetter = element<HTMLElement>("help-letter");
  private readonly helpKind = element<HTMLElement>("help-kind");
  private readonly helpHand = element<HTMLCanvasElement>("help-hand");
  private readonly gameOver = element<HTMLElement>("game-over");
  private readonly finalScore = element<HTMLElement>("final-score");
  private readonly newRecord = element<HTMLElement>("new-record");
  private game: BubbleGame | null = null;
  private clockMs = 0;
  private lastFrameMs: number | null = null;
  private frame: number | undefined;
  private shownHelp: string | null = null;
  private finished = false;

  constructor(
    private readonly references: ReferenceHands,
    private readonly progress: Progress,
  ) {
    element<HTMLButtonElement>("again").addEventListener("click", () => this.restart());
  }

  enter(): void {
    this.game = null;
    this.finished = false;
    this.lastFrameMs = null;
    this.gameOver.hidden = true;
    this.doingText.textContent = "–";
    this.frame = requestAnimationFrame((now) => this.tick(now));
  }

  leave(): void {
    if (this.frame !== undefined) {
      cancelAnimationFrame(this.frame);
    }
  }

  handleKey(event: KeyboardEvent): boolean {
    if (event.key !== RESTART_KEY) {
      return false;
    }
    this.restart();
    return true;
  }

  update(reading: Reading): void {
    this.game ??= new BubbleGame(ALPHABET, Math.random, this.clockMs);
    for (const letter of reading.confirmed) {
      this.game.tryPop(letter, this.clockMs);
    }
    this.doingText.textContent = reading.prediction?.letter ?? "–";
  }

  private restart(): void {
    this.game = new BubbleGame(ALPHABET, Math.random, this.clockMs);
    this.finished = false;
    this.gameOver.hidden = true;
  }

  private tick(nowMs: number): void {
    const stepMs = this.lastFrameMs === null ? 0 : Math.min(nowMs - this.lastFrameMs, MAX_STEP_MS);
    this.lastFrameMs = nowMs;
    this.clockMs += stepMs;
    const game = this.game;
    if (game !== null) {
      game.update(this.clockMs);
      if (game.isOver && !this.finished) {
        this.finish(game);
      }
    }
    this.renderPanel(game);
    this.draw(game);
    this.frame = requestAnimationFrame((now) => this.tick(now));
  }

  private finish(game: BubbleGame): void {
    this.finished = true;
    const isRecord = this.progress.recordScore(game.score);
    if (isRecord) {
      this.progress.save(browserStore());
    }
    this.finalScore.textContent = game.score === 1 ? "1 punto" : `${game.score} puntos`;
    this.newRecord.hidden = !isRecord;
    this.gameOver.hidden = false;
  }

  private renderPanel(game: BubbleGame | null): void {
    this.scoreText.textContent = String(game?.score ?? 0);
    this.levelText.textContent = String(game?.level ?? 1);
    this.recordText.textContent = `récord ${this.progress.highScore}`;
    const lives = game?.lives ?? START_LIVES;
    this.livesText.textContent = "●".repeat(lives) + "○".repeat(START_LIVES - lives);
    this.showHelp(game === null || game.isOver ? null : game.helpLetter(this.clockMs));
  }

  private showHelp(letter: string | null): void {
    if (letter === this.shownHelp) {
      return;
    }
    this.shownHelp = letter;
    this.help.hidden = letter === null;
    const points = letter === null ? undefined : this.references[letter];
    if (letter === null || points === undefined) {
      return;
    }
    this.helpLetter.textContent = letter;
    this.helpKind.textContent = isDynamic(letter) ? "con movimiento" : "";
    const context = this.helpHand.getContext("2d");
    if (context !== null) {
      context.clearRect(0, 0, this.helpHand.width, this.helpHand.height);
      drawSkeleton(context, fitHand(points, this.helpHand.width), CHALK, 3);
    }
  }

  private draw(game: BubbleGame | null): void {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(this.canvas.clientWidth * ratio);
    const height = Math.round(this.canvas.clientHeight * ratio);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const context = this.canvas.getContext("2d");
    if (context === null) {
      return;
    }
    context.clearRect(0, 0, width, height);
    if (game === null) {
      return;
    }
    const radius = BUBBLE_RADIUS * height;
    const lineWidth = Math.max(2, radius / 14);
    context.lineWidth = lineWidth;
    context.textAlign = "center";
    context.textBaseline = "middle";

    for (const bubble of game.bubbles) {
      const x = bubble.x * width;
      const y = bubble.y * height;
      const color = letterColor(bubble.letter);
      context.strokeStyle = color;
      context.fillStyle = color;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.stroke();
      context.beginPath();
      context.arc(x, y, radius * 0.72, Math.PI * 1.1, Math.PI * 1.4);
      context.stroke();
      const size = bubble.letter.length === 1 ? radius * 1.1 : radius * 0.75;
      context.font = `italic 700 ${size}px system-ui, sans-serif`;
      context.fillText(bubble.letter, x, y + radius * 0.05);
    }

    for (const pop of game.pops) {
      const progress = (this.clockMs - pop.poppedMs) / POP_DURATION_MS;
      const inner = radius * (0.6 + progress);
      const outer = radius * (1 + 1.2 * progress);
      context.strokeStyle = letterColor(pop.letter);
      context.beginPath();
      for (let ray = 0; ray < BURST_RAYS; ray++) {
        const angle = (2 * Math.PI * ray) / BURST_RAYS;
        context.moveTo(
          pop.x * width + inner * Math.cos(angle),
          pop.y * height + inner * Math.sin(angle),
        );
        context.lineTo(
          pop.x * width + outer * Math.cos(angle),
          pop.y * height + outer * Math.sin(angle),
        );
      }
      context.stroke();
    }

    for (const miss of game.misses) {
      const fade = 1 - (this.clockMs - miss.poppedMs) / MISS_DURATION_MS;
      const x = miss.x * width;
      const y = radius + lineWidth * 3;
      const arm = radius * 0.6;
      context.globalAlpha = Math.max(0, fade);
      context.strokeStyle = MISS_COLOR;
      context.beginPath();
      context.moveTo(x - arm, y - arm);
      context.lineTo(x + arm, y + arm);
      context.moveTo(x + arm, y - arm);
      context.lineTo(x - arm, y + arm);
      context.stroke();
      context.globalAlpha = 1;
    }
  }
}
