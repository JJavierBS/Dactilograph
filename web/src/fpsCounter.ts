export class FpsCounter {
  private lastMs: number | null = null;
  fps = 0;

  constructor(private readonly smoothing: number = 0.9) {}

  tick(nowMs: number): number {
    if (this.lastMs !== null && nowMs > this.lastMs) {
      const instant = 1000 / (nowMs - this.lastMs);
      this.fps =
        this.fps === 0 ? instant : this.smoothing * this.fps + (1 - this.smoothing) * instant;
    }
    this.lastMs = nowMs;
    return this.fps;
  }
}
