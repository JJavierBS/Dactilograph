import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/" +
  "float16/1/hand_landmarker.task";
const MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1";
const MODEL_PATH = new URL("../public/models/hand_landmarker.task", import.meta.url);

function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

if (existsSync(MODEL_PATH) && sha256(readFileSync(MODEL_PATH)) === MODEL_SHA256) {
  process.exit(0);
}

const response = await fetch(MODEL_URL);
if (!response.ok) {
  throw new Error(`Could not download the hand model: ${response.status}`);
}
const data = Buffer.from(await response.arrayBuffer());
if (sha256(data) !== MODEL_SHA256) {
  throw new Error("The downloaded hand model does not match the expected version");
}
writeFileSync(MODEL_PATH, data);
console.log("Hand model downloaded to public/models/hand_landmarker.task");
