import "./style.css";
import { drawHand } from "./drawing";
import type { Hand } from "./hand";
import { HandTracker } from "./handTracker";
import { normalizeHand } from "./normalization";
import { configureRuntime, type Prediction, StaticClassifier } from "./staticClassifier";

const MODELS_URL = `${import.meta.env.BASE_URL}models/`;
const HAND_COLOR = "#f2d35b";

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

function showStatus(message: string): void {
  statusText.textContent = message;
}

function showPrediction(prediction: Prediction | null): void {
  letterText.textContent = prediction?.letter ?? "–";
  confidenceText.textContent =
    prediction === null ? "no veo tu mano" : prediction.confidence.toFixed(2).replace(".", ",");
}

async function loadModels(): Promise<[HandTracker, StaticClassifier]> {
  configureRuntime();
  const labelsResponse = await fetch(`${MODELS_URL}letter_classifier.json`);
  const { labels } = (await labelsResponse.json()) as { labels: string[] };
  return Promise.all([
    HandTracker.create(),
    StaticClassifier.load(`${MODELS_URL}letter_classifier.onnx`, labels),
  ]);
}

async function openCamera(): Promise<void> {
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  await video.play();
}

function run(tracker: HandTracker, classifier: StaticClassifier): void {
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("The browser cannot draw on a canvas");
  }
  let lastFrameTime = -1;
  let classifying = false;
  let hand: Hand | null = null;

  const tick = (now: number): void => {
    if (video.currentTime !== lastFrameTime) {
      lastFrameTime = video.currentTime;
      hand = tracker.detect(video, now);
      if (hand === null) {
        showPrediction(null);
      } else if (!classifying) {
        classifying = true;
        classifier
          .predict(normalizeHand(hand))
          .then(showPrediction)
          .finally(() => {
            classifying = false;
          });
      }
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
    const [tracker, classifier] = await models;
    showStatus("Haz una letra delante de la cámara.");
    startButton.hidden = true;
    run(tracker, classifier);
  } catch (error) {
    startButton.disabled = false;
    showStatus(
      error instanceof DOMException && error.name === "NotAllowedError"
        ? "Sin permiso para usar la cámara. Actívalo en el navegador y vuelve a intentarlo."
        : `No he podido abrir la cámara: ${String(error)}`,
    );
  }
});
