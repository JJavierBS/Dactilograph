import "./style.css";
import { drawHand } from "./drawing";
import { DynamicClassifier, type NearestNeighborsModel } from "./dynamicClassifier";
import { sequenceFeatures } from "./dynamicFeatures";
import { FpsCounter } from "./fpsCounter";
import type { Hand } from "./hand";
import { HandTracker } from "./handTracker";
import { LearnScreen } from "./learnScreen";
import { LiveRecognizer } from "./liveRecognizer";
import { MenuScreen, type ReferenceHands } from "./menuScreen";
import { normalizeHand } from "./normalization";
import { browserStore, Progress } from "./progress";
import { RecognizeScreen } from "./recognizeScreen";
import { element, type Screen } from "./screen";
import { configureRuntime, StaticClassifier } from "./staticClassifier";

const MODELS_URL = `${import.meta.env.BASE_URL}models/`;
const HAND_COLOR = "#f2d35b";
const FPS_KEY = "f";
const SCREEN_KEYS: Readonly<Record<string, string>> = {
  "1": "#aprender",
  "2": "#reconocer",
  Escape: "#",
};

const camera = element<HTMLDivElement>("camera");
const cameraGate = element<HTMLDivElement>("camera-gate");
const startButton = element<HTMLButtonElement>("start");
const statusText = element<HTMLParagraphElement>("status");
const video = element<HTMLVideoElement>("video");
const canvas = element<HTMLCanvasElement>("view");
const fpsText = element<HTMLParagraphElement>("fps");
const trayLinks = document.querySelectorAll<HTMLAnchorElement>(".tray a");

function showStatus(message: string): void {
  statusText.textContent = message;
}

async function fetchJson<T>(name: string): Promise<T> {
  const response = await fetch(`${MODELS_URL}${name}`);
  if (!response.ok) {
    throw new Error(`${name}: ${response.status}`);
  }
  return (await response.json()) as T;
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

async function openCamera(): Promise<void> {
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  await video.play();
}

class App {
  private current: Screen;
  private live: LiveRecognizer | null = null;

  constructor(private readonly screens: ReadonlyMap<string, Screen>) {
    this.current = screens.get("menu")!;
  }

  route(): void {
    const name = window.location.hash.replace("#", "") || "menu";
    const next = this.screens.get(name) ?? this.screens.get("menu")!;
    this.current.leave();
    for (const screen of this.screens.values()) {
      screen.element.hidden = screen !== next;
    }
    for (const link of trayLinks) {
      link.classList.toggle("active", link.dataset["screen"] === next.name);
    }
    const slot = next.element.querySelector(".camera-slot");
    camera.hidden = slot === null;
    slot?.append(camera);
    this.live?.reset();
    this.current = next;
    next.enter(performance.now());
  }

  handleKey(event: KeyboardEvent): void {
    if (event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }
    if (this.current.handleKey(event)) {
      event.preventDefault();
      return;
    }
    const target = SCREEN_KEYS[event.key];
    if (target !== undefined) {
      window.location.hash = target;
    } else if (event.key === FPS_KEY) {
      fpsText.hidden = !fpsText.hidden;
    }
  }

  run(tracker: HandTracker, live: LiveRecognizer): void {
    this.live = live;
    const context = canvas.getContext("2d");
    if (context === null) {
      throw new Error("The browser cannot draw on a canvas");
    }
    const fps = new FpsCounter();
    let lastFrameTime = -1;
    let reading = false;
    let hand: Hand | null = null;

    const tick = (now: number): void => {
      if (this.current.usesCamera && !reading && video.currentTime !== lastFrameTime) {
        lastFrameTime = video.currentTime;
        hand = tracker.detect(video, now);
        fpsText.textContent = `${Math.round(fps.tick(now))} fps`;
        reading = true;
        const screen = this.current;
        live
          .read(now, hand)
          .then((result) => {
            if (screen === this.current) {
              screen.update(result, now);
            }
          })
          .catch((error: unknown) => showStatus(`Error al reconocer: ${String(error)}`))
          .finally(() => {
            reading = false;
          });
      }
      const frame = tracker.mirroredFrame;
      if (canvas.width !== frame.width || canvas.height !== frame.height) {
        canvas.width = frame.width;
        canvas.height = frame.height;
      }
      context.drawImage(frame, 0, 0);
      if (hand !== null) {
        drawHand(context, hand, HAND_COLOR, Math.max(2, canvas.width / 200));
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

async function main(): Promise<void> {
  const models = loadModels();
  models.then(
    () => showStatus("Listo. Activa la cámara para empezar."),
    (error: unknown) => showStatus(`No he podido cargar los modelos: ${String(error)}`),
  );
  const progress = Progress.load(browserStore());
  const references = await fetchJson<ReferenceHands>("reference_hands.json");
  const screens: Screen[] = [
    new MenuScreen(references, progress),
    new LearnScreen(references, progress),
    new RecognizeScreen(),
  ];
  const app = new App(new Map(screens.map((screen) => [screen.name, screen])));
  window.addEventListener("hashchange", () => app.route());
  document.addEventListener("keydown", (event) => app.handleKey(event));
  app.route();

  startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    try {
      showStatus("Pidiendo permiso para usar la cámara…");
      await openCamera();
      showStatus("Cargando los modelos…");
      const [tracker, live] = await models;
      cameraGate.hidden = true;
      app.run(tracker, live);
    } catch (error) {
      startButton.disabled = false;
      showStatus(
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Sin permiso para usar la cámara. Actívalo en el navegador y vuelve a intentarlo."
          : `No he podido abrir la cámara: ${String(error)}`,
      );
    }
  });
}

main().catch((error: unknown) => showStatus(`No he podido arrancar: ${String(error)}`));
