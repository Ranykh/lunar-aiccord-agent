# token_meter.py
from pathlib import Path
from datetime import datetime
from threading import Lock

TOKENS_DIR = Path("/Users/ranykhirbawi/Desktop/LunarAIccord/tokens_count")
TOKENS_DIR.mkdir(exist_ok=True, parents=True)
TOTAL_FILE = TOKENS_DIR / "total_tokens.txt"
LOG_FILE   = TOKENS_DIR / "runs.log"

_lock = Lock()

class TokenMeter:
    """
    Minimal token meter:
      - call .add(prompt=..., completion=...) for chat calls
      - call .add_embedding(prompt=...) for embedding calls
    Writes an itemized line to runs.log and maintains a running total in total_tokens.txt
    """
    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.embedding = 0

    def add(self, prompt: int = 0, completion: int = 0):
        with _lock:
            self.prompt += int(prompt or 0)
            self.completion += int(completion or 0)

    def add_embedding(self, tokens: int = 0):
        with _lock:
            self.embedding += int(tokens or 0)

    def _read_total(self) -> int:
        if TOTAL_FILE.exists():
            try:
                return int(TOTAL_FILE.read_text().strip() or "0")
            except Exception:
                return 0
        return 0

    def _write_total(self, total: int):
        TOTAL_FILE.write_text(str(total))

    def flush(self, tag: str = "run"):
        with _lock:
            used = self.prompt + self.completion + self.embedding
            prev = self._read_total()
            new_total = prev + used
            # log line
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else "") +
                                f"{ts}\t{tag}\tprompt={self.prompt}\tcompletion={self.completion}\tembed={self.embedding}\ttotal_added={used}\n")
            self._write_total(new_total)
            # return a summary for printing
            return {"added": used, "prompt": self.prompt, "completion": self.completion,
                    "embedding": self.embedding, "new_total": new_total}
