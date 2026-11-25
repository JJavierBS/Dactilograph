import { type Chalk, DIM, PINK, WHITE } from "../chalk";
import { type Button, type CanvasScreen, type Context, chalkWordButton } from "../ui";

const START_KEYS = new Set([" ", "Enter"]);

export class CameraScreen implements CanvasScreen {
  readonly trayActive = "";
  readonly usesCamera = false;
  status = "Cargando los modelos…";
  failed = false;
  private readonly hoverSince = new Map<string, number>();

  enter(): void {}

  keyAction(key: string): string | null {
    return START_KEYS.has(key) ? "activar" : null;
  }

  handle(action: string): string | null {
    return action;
  }

  update(): void {}

  draw(_canvas: CanvasRenderingContext2D, chalk: Chalk, context: Context): Button[] {
    chalk.text("Necesito ver tu mano", [640, 230], 3.0, WHITE, 4, "script", "center");
    chalk.text(
      "La cámara no sale de tu ordenador: todo se calcula en el navegador.",
      [640, 300],
      0.75,
      DIM,
      1,
      "sans",
      "center",
    );
    chalk.text(this.status, [640, 560], 0.75, this.failed ? PINK : DIM, 1, "sans", "center");
    return [
      chalkWordButton(
        chalk,
        "activar la cámara",
        "espacio",
        [640, 440],
        WHITE,
        "activar",
        context,
        this.hoverSince,
        true,
      ),
    ];
  }
}
