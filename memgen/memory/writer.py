import time
from typing import Dict, List, Optional


class MemoryWriter:
    """Create and gate experience entries for writeback."""

    def __init__(self, min_reward: float = 0.8, require_grounding: bool = True) -> None:
        self.min_reward = min_reward
        self.require_grounding = require_grounding

    def should_write(self, reward: float, has_grounding: bool = True) -> bool:
        if reward < self.min_reward:
            return False
        if self.require_grounding and not has_grounding:
            return False
        return True

    def create_entry(
        self,
        query: str,
        answer: str,
        reward: float,
        task_type: str = "open_qa",
        context_snippets: Optional[List[str]] = None,
    ) -> Dict:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry_id = f"exp_{int(time.time())}_{abs(hash(query)) % 10000}"
        return {
            "id": entry_id,
            "task_type": task_type,
            "query": query,
            "answer": answer,
            "context_snippets": context_snippets or [],
            "reward": reward,
            "timestamp": timestamp,
        }
