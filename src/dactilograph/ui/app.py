import time

import cv2
import numpy as np

from dactilograph.camera import Camera
from dactilograph.classifier import DynamicLetterClassifier, LetterClassifier
from dactilograph.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SEQUENCES_PATH,
    iter_sequences,
    load_dataset,
    load_sequences,
)
from dactilograph.hand_detector import HandDetector
from dactilograph.live_recognition import LiveRecognizer, Reading
from dactilograph.progress import Progress
from dactilograph.reference_hands import reference_hands
from dactilograph.ui.base import Context, Screen
from dactilograph.ui.chalk import HEIGHT, WIDTH, Chalk, board_background, chalk_grain, draw_wood
from dactilograph.ui.learn import LearnScreen
from dactilograph.ui.menu import MenuScreen
from dactilograph.ui.play import PlayScreen
from dactilograph.ui.preparing import NoCameraScreen, Preparation, PreparingScreen, missing_steps
from dactilograph.ui.recognize import RecognizeScreen
from dactilograph.ui.widgets import QUIT_ONLY, TRAY_ITEMS, Button, draw_tray

WINDOW_NAME = "Dactilograph"
ESCAPE = 27
GLOBAL_KEYS = {
    ESCAPE: "menu",
    ord("1"): "aprender",
    ord("2"): "reconocer",
    ord("3"): "jugar",
    ord("q"): "salir",
}
QUIT_KEYS = {ESCAPE: "salir", ord("q"): "salir"}


class App:
    def __init__(self) -> None:
        rng = np.random.default_rng()
        self._rng = rng
        self._background = board_background(rng)
        draw_wood(self._background, rng)
        self._chalk = Chalk(chalk_grain(rng))
        self._progress = Progress.load()
        self._references = reference_hands(
            load_dataset(DEFAULT_DATASET_PATH),
            iter_sequences(load_sequences(DEFAULT_SEQUENCES_PATH)),
        )
        self._mouse: tuple[int, int] | None = None
        self._clicks: list[tuple[int, int]] = []
        self._camera: Camera | None = None
        self._detector: HandDetector | None = None
        self._live: LiveRecognizer | None = None
        self._preparation: Preparation | None = None
        self._screens: dict[str, Screen] = {}
        self._screen: Screen = MenuScreen()

    def run(self) -> None:
        cv2.namedWindow(
            WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
        )
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._start()
        try:
            self._loop()
        finally:
            if self._camera is not None:
                self._camera.close()
            if self._detector is not None:
                self._detector.close()
            cv2.destroyAllWindows()

    def _start(self) -> None:
        try:
            self._camera = Camera()
        except RuntimeError:
            self._screen = NoCameraScreen()
            return
        steps = missing_steps()
        if steps:
            self._preparation = Preparation(steps)
            self._screen = PreparingScreen(self._preparation)
        else:
            self._load_models()

    def _load_models(self) -> None:
        self._detector = HandDetector()
        self._live = LiveRecognizer(LetterClassifier().predict, DynamicLetterClassifier().predict)
        self._screens = {
            "menu": MenuScreen(),
            "aprender": LearnScreen(),
            "reconocer": RecognizeScreen(),
            "jugar": PlayScreen(self._rng),
        }
        self._go("menu", self._context(time.monotonic(), None, None))

    def _loop(self) -> None:
        while True:
            now = time.monotonic()
            if self._preparation is not None and self._preparation.done:
                self._preparation = None
                self._load_models()

            frame = None
            reading = None
            if self._screen.uses_camera and self._camera is not None:
                frame = self._camera.read()
            if frame is not None and self._detector is not None and self._live is not None:
                reading = self._live.read(now, self._detector.detect(frame))

            context = self._context(now, frame, reading)
            self._screen.update(context)
            canvas = self._background.copy()
            self._chalk.clear()
            buttons = self._screen.draw(canvas, self._chalk, context)
            self._chalk.apply(canvas)
            ready = bool(self._screens)
            items = TRAY_ITEMS if ready else self._screen.tray_items or QUIT_ONLY
            buttons += draw_tray(canvas, self._screen.tray_active, self._mouse, items)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKeyEx(1)
            actions = self._clicked_actions(buttons)
            if key != -1:
                screen_action = self._screen.key_action(key)
                global_action = (GLOBAL_KEYS if ready else QUIT_KEYS).get(key)
                if screen_action or global_action:
                    actions.append(screen_action or global_action)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
            for action in actions:
                target = self._screen.handle(action, context)
                if target == "salir":
                    return
                if target == "reintentar":
                    self._start()
                if target is not None and target in self._screens:
                    self._go(target, context)

    def _context(self, now: float, frame: np.ndarray | None, reading: Reading | None) -> Context:
        return Context(now, frame, reading, self._mouse, self._references, self._progress)

    def _go(self, name: str, context: Context) -> None:
        screen = self._screens[name]
        if self._live is not None:
            self._live.reset()
        screen.enter(context)
        self._screen = screen

    def _clicked_actions(self, buttons: list[Button]) -> list[str]:
        actions = []
        for click in self._clicks:
            hit = next((b for b in reversed(buttons) if b.contains(click)), None)
            if hit is not None:
                actions.append(hit.action)
        self._clicks.clear()
        return actions

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self._mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicks.append((x, y))


def main() -> None:
    App().run()
