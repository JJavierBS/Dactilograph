export interface NearestNeighborsModel {
  readonly k: number;
  readonly labels: readonly string[];
  readonly features: readonly (readonly number[])[];
}

export class DynamicClassifier {
  constructor(private readonly model: NearestNeighborsModel) {
    if (model.k !== 1) {
      throw new Error("Only the 1-nearest-neighbor model is supported");
    }
    if (model.labels.length !== model.features.length || model.features.length === 0) {
      throw new Error("The model needs one label per example");
    }
  }

  predict(features: readonly number[]): string {
    let bestIndex = 0;
    let bestDistance = Infinity;
    this.model.features.forEach((example, index) => {
      let distance = 0;
      for (let column = 0; column < example.length; column++) {
        distance += (example[column]! - features[column]!) ** 2;
      }
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    return this.model.labels[bestIndex]!;
  }
}
