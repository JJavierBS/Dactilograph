import "./style.css";
import { drawHand } from "./drawing";
import { DynamicClassifier, type NearestNeighborsModel } from "./dynamicClassifier";
import { sequenceFeatures } from "./dynamicFeatures";
import { FpsCounter } from "./fpsCounter";
import type { Hand } from "./hand";
import { HandTracker } from "./handTracker";
import { LiveRecognizer, type Reading } from "./liveRecognizer";
import { normalizeHand } from "./normalization";
import { configureRuntime, type Prediction, StaticClassifier } from "./staticClassifier";

const MODELS_URL = `${import.meta.env.BASE_URL}models/`;
const HAND_COLOR = "#f2d35b";
const HISTORY_LENGTH = 6;
const FPS_KEY = "f";

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`Missing element #${id}`);
  }
  return found as T;
}

const startButton = element<HTMLButtonElement>("start");
const statusText = element<HTMLParagraphElement>("status");
const video = element<HTMLVideoElement>("video");
const canvas = element<HTMLCanvasElement>("view");
const letterText = element<HTMLParagraphElement>("letter");
const confidenceText = element<HTMLParagraphElement>("confidence");
const historyList = element<HTMLOListElement>("history");
const fpsText = element<HTMLParagraphElement>("fps");

const history: string[] = [];

function showStatus(message: string): void {
  statusText.textContent = message;
}

function showPrediction(prediction: Prediction | null): void {
  letterText.textContent = prediction?.letter ?? "–";
  confidenceText.textContent =
    prediction === null ? "no veo tu mano" : prediction.confidence.toFixed(2).replace(".", ",");
}

function addToHistory(letters: readonly string[]): void {
  if (letters.length === 0) {
    return;
  }
  history.push(...letters);
  history.splice(0, Math.max(0, history.length - HISTORY_LENGTH));
  historyList.replaceChildren(
    ...history.map((letter, index) => {
      const item = document.createElement("li");
      item.textContent = letter;
      item.classList.toggle("newest", index === history.length - 1);
      return item;
    }),
  );
}

function showReading(reading: Reading): void {
  showPrediction(reading.prediction);
  addToHistory(reading.confirmed);
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

function run(tracker: HandTracker, live: LiveRecognizer): void {
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("The browser cannot draw on a canvas");
  }
  let lastFrameTime = -1;
  let reading = false;
  let hand: Hand | null = null;
  const fps = new FpsCounter();
  document.addEventListener("keydown", (event) => {
    if (event.key === FPS_KEY) {
      fpsText.hidden = !fpsText.hidden;
    }
  });

  const tick = (now: number): void => {
    if (!reading && video.currentTime !== lastFrameTime) {
      lastFrameTime = video.currentTime;
      hand = tracker.detect(video, now);
      fpsText.textContent = `${Math.round(fps.tick(now))} fps`;
      reading = true;
      live
        .read(now, hand)
        .then(showReading)
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

const models = loadModels();
models.then(
  () => showStatus("Listo. Activa la cámara para empezar."),
  (error: unknown) => showStatus(`No he podido cargar los modelos: ${String(error)}`),
);

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  try {
    showStatus("Pidiendo permiso para usar la cámara…");
    await openCamera();
    showStatus("Cargando los modelos…");
    const [tracker, live] = await models;
    showStatus("Haz una letra delante de la cámara. Las de movimiento, de principio a fin.");
    startButton.hidden = true;
    run(tracker, live);
  } catch (error) {
    startButton.disabled = false;
    showStatus(
      error instanceof DOMException && error.name === "NotAllowedError"
        ? "Sin permiso para usar la cámara. Actívalo en el navegador y vuelve a intentarlo."
        : `No he podido abrir la cámara: ${String(error)}`,
    );
  }
});
