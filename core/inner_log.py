from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("nyx.log")


class InnerLog:
    """
    Nyx's private diary. Never shown to users directly.
    Entries become context for future LLM calls, giving Nyx a sense of continuity.
    """

    MAX_ENTRIES = 100

    def __init__(self, config, persist_path: str = "./data/inner_log.json"):
        self.config = config
        self.path = Path(persist_path)
        self.entries: list[dict] = []
        self._load()

    def add_entry(self, content: str, entry_type: str = "reflection"):
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "content": content,
        })
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries = self.entries[-self.MAX_ENTRIES :]
        self._save()
        logger.debug("Inner log entry added (%s)", entry_type)

    def get_recent_context(self, n: int | None = None) -> str:
        n = n or self.config.max_log_context_entries
        recent = self.entries[-n:] if self.entries else []
        if not recent:
            return ""
        return "\n".join(
            f"[{e['timestamp'][:16]}] {e['content']}" for e in recent
        )

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.entries = json.load(f)
