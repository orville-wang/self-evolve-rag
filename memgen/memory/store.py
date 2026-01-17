import json
import os
from typing import Any, Dict, List, Optional


class ExperienceStore:
    """Simple JSONL-backed experience store with Jaccard retrieval."""

    def __init__(
        self,
        store_path: str,
        index_type: str = "simple",
        topk: int = 4,
        min_score: float = 0.25,
    ) -> None:
        self.store_path = store_path
        self.index_type = index_type
        self.topk = topk
        self.min_score = min_score
        self.entries: List[Dict[str, Any]] = []

        if self.store_path:
            self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not os.path.exists(self.store_path):
            return
        with open(self.store_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def save(self) -> None:
        if not self.store_path:
            return
        dirpath = os.path.dirname(self.store_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add(self, entry: Dict[str, Any]) -> None:
        self.entries.append(entry)

    def search(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.index_type != "simple":
            raise ValueError(f"Unsupported index_type: {self.index_type}")
        if not query:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[Dict[str, Any]] = []
        for entry in self.entries:
            entry_query = entry.get("query", "")
            entry_tokens = self._tokenize(entry_query)
            if not entry_tokens:
                continue
            score = self._jaccard(query_tokens, entry_tokens)
            if score < self.min_score:
                continue
            scored.append(
                {
                    "text": self._entry_text(entry),
                    "score": score,
                    "entry": entry,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: (topk or self.topk)]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def _entry_text(entry: Dict[str, Any]) -> str:
        return (
            entry.get("memory_text")
            or entry.get("summary")
            or entry.get("answer")
            or entry.get("query")
            or ""
        )
