import "./style.css";
import {
  boardBackground,
  Chalk,
  chalkGrain,
  DIM,
  FRAME,
  HEIGHT,
  type Point2,
  WIDTH,
} from "./chalk";
import { bindCollaborationForm, COLLABORATE_ACTION, scrollToCollaboration } from "./collaborate";
import { DynamicClassifier, type NearestNeighborsModel } from "./dynamicClassifier";
import { sequenceFeatures } from "./dynamicFeatures";
import { FpsCounter } from "./fpsCounter";
import { HandTracker } from "./handTracker";
import { LiveRecognizer, type Reading } from "./liveRecognizer";
import { normalizeHand } from "./normalization";
import { browserStore, Progress } from "./progress";
import { CameraScreen } from "./screens/camera";
import { LearnScreen } from "./screens/learn";
import { MenuScreen } from "./screens/menu";
import { PlayScreen } from "./screens/play";
import { RecognizeScreen } from "./screens/recognize";
import { configureRuntime, StaticClassifier } from "./staticClassifier";
import { type Button, type CanvasScreen, type Context, drawTray, type ReferenceHands } from "./ui";

const BASE_URL = import.meta.env.BASE_URL;
const MODELS_URL = `${BASE_URL}models/`;
const SCREEN_KEYS: Readonly<Record<string, string>> = {
  Escape: "menu",
  "1": "aprender",
  "2": "reconocer",
  "3": "jugar",
};
const FPS_KEY = "f";
const EDITABLE = "input, textarea, select";
const FPS_ORIGIN: Point2 = [WIDTH - FRAME - 24, FRAME + 34];

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`Missing element #${id}`);
  }
  return found as T;
}

async function fetchJson<T>(name: string): Promise<T> {
  const response = await fetch(`${MODELS_URL}${name}`);
  if (!response.ok) {
    throw new Error(`${name}: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function loadFonts(): Promise<void> {
  const faces = [
    new FontFace("Rubik", `url(${BASE_URL}fonts/Rubik.woff2)`, { weight: "300 900" }),
    new FontFace("Rubik", `url(${BASE_URL}fonts/Rubik-Italic.woff2)`, {
      weight: "300 900",
      style: "italic",
    }),
  ];
  for (const face of await Promise.all(faces.map((face) => face.load()))) {
    document.fonts.add(face);
  }
}

async function loadModels(): Promise<[HandTracker, LiveRecognizer]> {
  configureRuntime();
  const [{ labels }, dynamicModel] = await Promise.all([
    fetchJson<{ labels: string[] }>("letter_classifier.json"),
    fetchJson<NearestNeighborsModel>("dynamic_classifier.json"),
  ]);
  const [tracker, staticClassifier] = await Promise.all([
    HandTracker.create(),
    StaticClassifier.load(`${MODELS_URL}letter_classifier.onnx`, labels),
  ]);
  const dynamicClassifier = new DynamicClassifier(dynamicModel);
  const live = new LiveRecognizer(
    (hand) => staticClassifier.predict(normalizeHand(hand)),
    (hands, timestampsMs) => dynamicClassifier.predict(sequenceFeatures(hands, timestampsMs)),
  );
  return [tracker, live];
}

class App {
  private readonly context: CanvasRenderingContext2D;
  private readonly background = boardBackground(Math.random);
  private readonly chalk = new Chalk(chalkGrain(Math.random));
  private readonly cameraScreen = new CameraScreen();
  private readonly fps = new FpsCounter();
  private current: CanvasScreen;
  private mouse: Point2 | null = null;
  private buttons: Button[] = [];
  private showFps = false;
  private cameraState: "off" | "starting" | "on" = "off";
  private tracker: HandTracker | null = null;
  private live: LiveRecognizer | null = null;
  private latest: Reading | null = null;
  private pending: string[] = [];
  private reading = false;
  private lastFrameTime = -1;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly video: HTMLVideoElement,
    private readonly screens: ReadonlyMap<string, CanvasScreen>,
    private readonly models: Promise<[HandTracker, LiveRecognizer]>,
    private readonly references: ReferenceHands,
    private readonly progress: Progress,
  ) {
    const context = canvas.getContext("2d");
    if (context === null) {
      throw new Error("The browser cannot draw on a canvas");
    }
    this.context = context;
    this.current = screens.get("menu")!;
    models.then(
      () => this.setStatus("Listo: activa la cámara para empezar."),
      (error: unknown) => this.setStatus(`No he podido cargar los modelos: ${String(error)}`, true),
    );
    canvas.addEventListener("mousemove", (event) => {
      this.mouse = this.toBoard(event);
    });
    canvas.addEventListener("mouseleave", () => {
      this.mouse = null;
    });
    canvas.addEventListener("click", (event) => {
      const point = this.toBoard(event);
      const hit = [...this.buttons].reverse().find((button) => button.contains(point));
      if (hit !== undefined) {
        this.act(hit.action);
      }
    });
    document.addEventListener("keydown", (event) => this.onKey(event));
    window.addEventListener("hashchange", () => this.route());
  }

  start(): void {
    this.route();
    requestAnimationFrame((now) => this.tick(now));
  }

  private get shown(): CanvasScreen {
    return this.current.usesCamera && this.cameraState !== "on" ? this.cameraScreen : this.current;
  }

  private setStatus(message: string, failed = false): void {
    this.cameraScreen.status = message;
    this.cameraScreen.failed = failed;
  }

  private toBoard(event: MouseEvent): Point2 {
    const bounds = this.canvas.getBoundingClientRect();
    return [
      ((event.clientX - bounds.left) * WIDTH) / bounds.width,
      ((event.clientY - bounds.top) * HEIGHT) / bounds.height,
    ];
  }

  private route(): void {
    const name = window.location.hash.replace("#", "") || "menu";
    const next = this.screens.get(name) ?? this.screens.get("menu")!;
    this.live?.reset();
    this.latest = null;
    this.pending = [];
    this.current = next;
    next.enter(this.makeContext(performance.now()));
  }

  private go(name: string): void {
    window.location.hash = name === "menu" ? "" : name;
    if (name === "menu" && window.location.hash === "") {
      this.route();
    }
  }

  private act(action: string): void {
    if (action === "activar") {
      void this.startCamera();
      return;
    }
    if (action === COLLABORATE_ACTION) {
      scrollToCollaboration();
      return;
    }
    const target = this.shown.handle(action, this.makeContext(performance.now()));
    if (target !== null && this.screens.has(target)) {
      this.go(target);
    }
  }

  private onKey(event: KeyboardEvent): void {
    const typing = event.target instanceof Element && event.target.closest(EDITABLE) !== null;
    if (typing || event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }
    const action = this.shown.keyAction(event.key) ?? SCREEN_KEYS[event.key] ?? null;
    if (action !== null) {
      event.preventDefault();
      this.act(action);
    } else if (event.key === FPS_KEY) {
      this.showFps = !this.showFps;
    }
  }

  private async startCamera(): Promise<void> {
    if (this.cameraState !== "off") {
      return;
    }
    this.cameraState = "starting";
    try {
      this.setStatus("Pidiendo permiso para usar la cámara…");
      this.video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      await this.video.play();
      this.setStatus("Cargando los modelos…");
      [this.tracker, this.live] = await this.models;
      this.cameraState = "on";
    } catch (error) {
      this.cameraState = "off";
      this.setStatus(
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Sin permiso para usar la cámara. Actívalo en el navegador y vuelve a intentarlo."
          : `No he podido abrir la cámara: ${String(error)}`,
        true,
      );
    }
  }

  private makeContext(nowMs: number): Context {
    const cameraOn = this.cameraState === "on" && this.tracker !== null;
    const reading =
      this.latest === null ? null : { ...this.latest, confirmed: this.pending.splice(0) };
    return {
      nowMs,
      frame: cameraOn ? this.tracker!.mirroredFrame : null,
      reading,
      mouse: this.mouse,
      references: this.references,
      progress: this.progress,
    };
  }

  private readFrame(nowMs: number): void {
    const tracker = this.tracker;
    const live = this.live;
    if (tracker === null || live === null || this.reading) {
      return;
    }
    if (this.video.currentTime === this.lastFrameTime) {
      return;
    }
    this.lastFrameTime = this.video.currentTime;
    const hand = tracker.detect(this.video, nowMs);
    this.fps.tick(nowMs);
    this.reading = true;
    const screen = this.current;
    live
      .read(nowMs, hand)
      .then((result) => {
        if (screen === this.current) {
          this.latest = result;
          this.pending.push(...result.confirmed);
        }
      })
      .catch((error: unknown) => this.setStatus(`Error al reconocer: ${String(error)}`, true))
      .finally(() => {
        this.reading = false;
      });
  }

  private tick(nowMs: number): void {
    if (this.current.usesCamera && this.cameraState === "on") {
      this.readFrame(nowMs);
    }
    const screen = this.shown;
    const context = this.makeContext(nowMs);
    screen.update(context);
    this.context.drawImage(this.background, 0, 0);
    this.chalk.clear();
    const buttons = screen.draw(this.context, this.chalk, context);
    if (this.showFps && context.frame !== null) {
      this.chalk.text(`${Math.round(this.fps.fps)} fps`, FPS_ORIGIN, 0.6, DIM, 1, "sans", "right");
    }
    this.chalk.apply(this.context);
    buttons.push(...drawTray(this.context, this.current.trayActive, this.mouse));
    this.buttons = buttons;
    requestAnimationFrame((now) => this.tick(now));
  }
}

async function main(): Promise<void> {
  bindCollaborationForm(element<HTMLFormElement>("colabora-form"));
  const models = loadModels();
  const [references] = await Promise.all([
    fetchJson<ReferenceHands>("reference_hands.json"),
    loadFonts(),
  ]);
  const progress = Progress.load(browserStore());
  const screens = new Map<string, CanvasScreen>([
    ["menu", new MenuScreen()],
    ["aprender", new LearnScreen()],
    ["reconocer", new RecognizeScreen()],
    ["jugar", new PlayScreen()],
  ]);
  new App(
    element<HTMLCanvasElement>("board"),
    element<HTMLVideoElement>("video"),
    screens,
    models,
    references,
    progress,
  ).start();
}

main().catch((error: unknown) => {
  document.body.textContent = `No he podido arrancar Dactilograph: ${String(error)}`;
});
