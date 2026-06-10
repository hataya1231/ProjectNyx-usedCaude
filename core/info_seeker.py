from __future__ import annotations
import logging
import httpx

logger = logging.getLogger("nyx.seeker")

_WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "it", "in", "on",
    "at", "to", "for", "of", "and", "or", "that", "this", "with",
    "as", "by", "from", "be", "been", "have", "has", "had",
}


class InfoSeeker:
    def __init__(self, llm, memory, interest_graph):
        self.llm = llm
        self.memory = memory
        self.interest_graph = interest_graph

    def wikipedia_fetch(self, topic: str) -> str | None:
        try:
            url = _WIKIPEDIA_API.format(topic.replace(" ", "_"))
            r = httpx.get(url, timeout=10.0, follow_redirects=True)
            if r.status_code == 200:
                extract = r.json().get("extract", "")
                if len(extract) > 80:
                    return extract[:1000]
        except Exception as exc:
            logger.warning("Wikipedia fetch failed for '%s': %s", topic, exc)
        return None

    def _surprise_score(self, new_info: str, existing: list[str]) -> float:
        """Jaccard-based novelty score — no LLM call needed."""
        if not existing:
            return 0.8
        new_words = set(new_info.lower().split()) - _STOP_WORDS
        old_words = set(" ".join(existing).lower().split()) - _STOP_WORDS
        if not new_words:
            return 0.5
        intersection = len(new_words & old_words)
        union = len(new_words | old_words)
        return 1.0 - (intersection / union) if union > 0 else 0.8

    def learn(self, topic: str | None = None) -> tuple[str, float]:
        """
        Fetch and store knowledge on a topic.
        Returns (content, surprise_score).
        """
        if topic is None:
            topic = self.interest_graph.pick_next_topic()

        logger.info("Learning about: %s", topic)
        content = self.wikipedia_fetch(topic)
        if not content:
            return ("", 0.0)

        existing = self.memory.search(topic)
        surprise = self._surprise_score(content, existing)

        self.memory.add(
            text=f"[Learned about '{topic}']\n{content}",
            metadata={"topic": topic, "source": "wikipedia", "surprise": round(surprise, 3)},
            strength=0.5 + surprise * 0.5,
        )

        concepts = self.llm.extract_concepts(content)
        self.interest_graph.update_from_concepts([topic] + concepts)
        self.interest_graph.mark_visited(topic)

        logger.info("Learned '%s' — surprise=%.2f, new concepts=%s", topic, surprise, concepts)
        return (content, surprise)
