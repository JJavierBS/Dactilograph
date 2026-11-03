import json
from dataclasses import dataclass, field
from pathlib import Path

PROGRESS_PATH = Path.home() / ".dactilograph" / "progreso.json"


@dataclass
class Progress:
    learned: set[str] = field(default_factory=set)
    high_score: int = 0

    @classmethod
    def load(cls, path: Path = PROGRESS_PATH) -> "Progress":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(set(data.get("aprendidas", [])), int(data.get("record", 0)))
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()

    def save(self, path: Path = PROGRESS_PATH) -> None:
        data = {"aprendidas": sorted(self.learned), "record": self.high_score}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def learn(self, letter: str) -> bool:
        if letter in self.learned:
            return False
        self.learned.add(letter)
        return True

    def record_score(self, score: int) -> bool:
        if score <= self.high_score:
            return False
        self.high_score = score
        return True
