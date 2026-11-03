from collections.abc import Iterable, Sequence


class LearnSession:
    def __init__(self, letters: Sequence[str], start: str | None = None) -> None:
        if not letters:
            raise ValueError("The session needs at least one letter")
        self.letters = list(letters)
        self.index = self.letters.index(start) if start in self.letters else 0
        self.success_s: float | None = None

    @property
    def target(self) -> str:
        return self.letters[self.index]

    @property
    def succeeded(self) -> bool:
        return self.success_s is not None

    def select(self, letter: str) -> None:
        self.index = self.letters.index(letter)
        self.success_s = None

    def next(self) -> None:
        self.index = (self.index + 1) % len(self.letters)
        self.success_s = None

    def previous(self) -> None:
        self.index = (self.index - 1) % len(self.letters)
        self.success_s = None

    def check(self, confirmed: Iterable[str], now_s: float) -> bool:
        if self.succeeded or self.target not in confirmed:
            return False
        self.success_s = now_s
        return True
