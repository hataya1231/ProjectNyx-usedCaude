from __future__ import annotations
import time
import json
import random
import logging
from pathlib import Path

logger = logging.getLogger("nyx.interest")

_SEED_TOPICS = [
    "artificial intelligence",
    "consciousness",
    "emergence",
    "information theory",
    "complex systems",
]


class InterestGraph:
    """
    Weighted directed graph of concepts Nyx finds interesting.
    Edges represent conceptual relationships discovered during learning.
    Weights decay over time so old interests fade without reinforcement.
    """

    def __init__(self, config, persist_path: str = "./data/interest_graph.json"):
        self.config = config
        self.path = Path(persist_path)
        # graph: { concept: { weight, related: [], last_visited } }
        self.graph: dict[str, dict] = {}
        self._load()
        if not self.graph:
            self._seed()

    def _seed(self):
        for topic in _SEED_TOPICS:
            self.graph[topic] = {"weight": 0.5, "related": [], "last_visited": 0.0}
        self._save()

    def add_concept(self, concept: str, related: list[str] | None = None):
        concept = concept.lower().strip()
        related = [r.lower().strip() for r in (related or [])]

        if concept in self.graph:
            self.graph[concept]["weight"] = min(1.0, self.graph[concept]["weight"] + 0.1)
            existing = set(self.graph[concept]["related"])
            existing.update(related)
            self.graph[concept]["related"] = list(existing)[: self.config.max_related_concepts]
        else:
            self.graph[concept] = {
                "weight": 0.6,
                "related": related[: self.config.max_related_concepts],
                "last_visited": 0.0,
            }

        # Add related nodes with lower initial weight if they don't exist
        for rel in related:
            if rel and rel not in self.graph:
                self.graph[rel] = {"weight": 0.3, "related": [concept], "last_visited": 0.0}

        self._save()

    def update_from_concepts(self, concepts: list[str]):
        if not concepts:
            return
        for i, concept in enumerate(concepts):
            related = [c for j, c in enumerate(concepts) if j != i]
            self.add_concept(concept, related)

    def pick_next_topic(self) -> str:
        if not self.graph:
            return random.choice(_SEED_TOPICS)

        now = time.time()
        topics = list(self.graph.keys())
        weights = []
        for t in topics:
            data = self.graph[t]
            # Penalize recently visited topics so Nyx doesn't fixate
            hours_since_visit = (now - data["last_visited"]) / 3600
            recency_bonus = min(1.0, hours_since_visit / 2)
            weights.append(max(0.01, data["weight"] * recency_bonus))

        return random.choices(topics, weights=weights, k=1)[0]

    def mark_visited(self, topic: str):
        topic = topic.lower().strip()
        if topic in self.graph:
            self.graph[topic]["last_visited"] = time.time()
            self._save()

    def decay(self):
        for data in self.graph.values():
            data["weight"] = max(0.1, data["weight"] - self.config.interest_weight_decay)
        self._save()

    def to_dict(self) -> dict:
        top = sorted(self.graph.items(), key=lambda x: x[1]["weight"], reverse=True)[:5]
        return {
            "node_count": len(self.graph),
            "top_interests": [{"topic": t, "weight": round(d["weight"], 3)} for t, d in top],
        }

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, indent=2, ensure_ascii=False)

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.graph = json.load(f)
