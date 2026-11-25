import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

export interface Prediction {
  readonly letter: string;
  readonly confidence: number;
}

const INPUT_NAME = "hand";
const OUTPUT_NAME = "probabilities";

export function configureRuntime(): void {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: wasmUrl };
}

export function bestPrediction(
  labels: readonly string[],
  probabilities: ArrayLike<number>,
): Prediction {
  let best = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index]! > probabilities[best]!) {
      best = index;
    }
  }
  const letter = labels[best];
  if (letter === undefined || probabilities.length !== labels.length) {
    throw new Error("The model and its labels do not match");
  }
  return { letter, confidence: probabilities[best]! };
}

export class StaticClassifier {
  private constructor(
    private readonly session: ort.InferenceSession,
    readonly labels: readonly string[],
  ) {}

  static async load(model: string, labels: readonly string[]): Promise<StaticClassifier> {
    return new StaticClassifier(await ort.InferenceSession.create(model), labels);
  }

  async probabilities(normalized: readonly number[]): Promise<Float32Array> {
    const input = new ort.Tensor("float32", Float32Array.from(normalized), [1, normalized.length]);
    const outputs = await this.session.run({ [INPUT_NAME]: input });
    const output = outputs[OUTPUT_NAME];
    if (output === undefined) {
      throw new Error("The model has no probabilities output");
    }
    return output.data as Float32Array;
  }

  async predict(normalized: readonly number[]): Promise<Prediction> {
    return bestPrediction(this.labels, await this.probabilities(normalized));
  }
}
