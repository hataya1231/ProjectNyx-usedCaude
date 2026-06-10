from __future__ import annotations
import time
import uuid
import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("nyx.memory")


class MemorySystem:
    def __init__(self, config):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="nyx_memory",
            embedding_function=ef,
        )

    def add(self, text: str, metadata: dict | None = None, strength: float = 1.0) -> str:
        doc_id = str(uuid.uuid4())
        meta = {
            "strength": strength,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 0,
            **(metadata or {}),
        }
        self.collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        return doc_id

    def search(self, query: str, n: int | None = None) -> list[str]:
        n = n or self.config.memory_max_results
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, total),
        )
        ids: list[str] = results["ids"][0]
        docs: list[str] = results["documents"][0]
        for doc_id in ids:
            self._reinforce(doc_id)
        return docs

    def _reinforce(self, doc_id: str):
        result = self.collection.get(ids=[doc_id])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        meta["strength"] = min(1.0, meta["strength"] + self.config.memory_reinforce_amount)
        meta["last_accessed"] = time.time()
        meta["access_count"] = meta.get("access_count", 0) + 1
        self.collection.update(ids=[doc_id], metadatas=[meta])

    def decay_and_prune(self):
        result = self.collection.get()
        if not result["ids"]:
            return

        now = time.time()
        seconds_per_day = 86400
        to_delete: list[str] = []
        update_ids: list[str] = []
        update_metas: list[dict] = []

        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            days_idle = (now - meta.get("last_accessed", now)) / seconds_per_day
            new_strength = meta["strength"] - self.config.memory_decay_rate * days_idle

            if new_strength < self.config.memory_prune_threshold:
                to_delete.append(doc_id)
            else:
                meta["strength"] = new_strength
                update_ids.append(doc_id)
                update_metas.append(meta)

        if to_delete:
            self.collection.delete(ids=to_delete)
            logger.info("Pruned %d faded memories", len(to_delete))
        if update_ids:
            self.collection.update(ids=update_ids, metadatas=update_metas)

    def count(self) -> int:
        return self.collection.count()
