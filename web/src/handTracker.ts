import { HandLandmarker } from "@mediapipe/tasks-vision";
import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";
import type { Hand, Handedness, Point3 } from "./hand";

export const HAND_MODEL_URL = `${import.meta.env.BASE_URL}models/hand_landmarker.task`;

function createLandmarker(delegate: "GPU" | "CPU"): Promise<HandLandmarker> {
  return HandLandmarker.createFromOptions(
    { wasmLoaderPath: wasmLoaderUrl, wasmBinaryPath: wasmBinaryUrl },
    {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 1,
    },
  );
}

const MIRRORED_HANDEDNESS: Record<string, Handedness> = { Left: "Right", Right: "Left" };

export class HandTracker {
  private readonly mirror: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private lastTimestampMs = -1;

  private constructor(private readonly landmarker: HandLandmarker) {
    this.mirror = document.createElement("canvas");
    const context = this.mirror.getContext("2d", { willReadFrequently: false });
    if (context === null) {
      throw new Error("The browser cannot draw on a canvas");
    }
    this.context = context;
  }

  static async create(): Promise<HandTracker> {
    try {
      return new HandTracker(await createLandmarker("GPU"));
    } catch {
      return new HandTracker(await createLandmarker("CPU"));
    }
  }

  get mirroredFrame(): HTMLCanvasElement {
    return this.mirror;
  }

  detect(video: HTMLVideoElement, nowMs: number): Hand | null {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width === 0 || height === 0) {
      return null;
    }
    if (this.mirror.width !== width || this.mirror.height !== height) {
      this.mirror.width = width;
      this.mirror.height = height;
    }
    this.context.setTransform(-1, 0, 0, 1, width, 0);
    this.context.drawImage(video, 0, 0, width, height);
    const timestampMs = Math.max(Math.round(nowMs), this.lastTimestampMs + 1);
    this.lastTimestampMs = timestampMs;

    const result = this.landmarker.detectForVideo(this.mirror, timestampMs);
    const landmarks = result.landmarks[0];
    const category = result.handedness[0]?.[0];
    if (landmarks === undefined || category === undefined) {
      return null;
    }
    const handedness = MIRRORED_HANDEDNESS[category.categoryName];
    if (handedness === undefined) {
      return null;
    }
    return {
      landmarks: landmarks.map((point): Point3 => [point.x, point.y, point.z]),
      handedness,
      imageWidth: width,
      imageHeight: height,
    };
  }

  close(): void {
    this.landmarker.close();
  }
}
