from __future__ import annotations
import logging
import time
from typing import Callable

from config import CONFIG
from core.emotion import EmotionSystem
from core.behavior import BehaviorEngine, NyxState
from core.memory import MemorySystem
from core.interest_graph import InterestGraph
from core.inner_log import InnerLog
from core.llm import LLMInterface
from core.sensor import SensorSystem
from core.info_seeker import InfoSeeker
from core.speech_trigger import SpeechTrigger

logger = logging.getLogger("nyx.core")


class NyxCore:
    def __init__(self, config=CONFIG):
        self.config = config

        self.emotion = EmotionSystem(config)
        self.behavior = BehaviorEngine()
        self.memory = MemorySystem(config)
        self.interest_graph = InterestGraph(config)
        self.inner_log = InnerLog(config)
        self.llm = LLMInterface(config)
        self.sensor = SensorSystem()
        self.info_seeker = InfoSeeker(self.llm, self.memory, self.interest_graph)
        self.speech_trigger = SpeechTrigger(config)

        self._running = False
        self._last_speech_tick: int = -999
        self._on_speak: Callable[[str], None] | None = None
        self._on_state_change: Callable[[dict], None] | None = None

    # ── event hooks ──────────────────────────────────────────────────────

    def on_speak(self, callback: Callable[[str], None]):
        self._on_speak = callback

    def on_state_change(self, callback: Callable[[dict], None]):
        self._on_state_change = callback

    # ── chat API ─────────────────────────────────────────────────────────

    def build_chat_context(self, user_message: str) -> tuple[list[dict], bool]:
        """
        Prepare the LLM message list and decide whether deep thought is needed.
        Used by the WebSocket server for streaming responses.
        """
        needs_deep = self.llm.needs_deep_thought(user_message)
        memories = self.memory.search(user_message)
        messages = self.llm._build_messages(
            user_message,
            emotion_context=self._emotion_str(),
            memories=memories,
            log_context=self.inner_log.get_recent_context(),
        )
        return messages, needs_deep

    def record_chat(self, user_message: str, response: str):
        """Store the completed conversation exchange in memory."""
        self.memory.add(
            f"User: {user_message}\nNyx: {response}",
            metadata={"type": "conversation"},
        )

    def chat(self, user_message: str) -> str:
        """Synchronous chat — used by CLI and internal code."""
        needs_deep = self.llm.needs_deep_thought(user_message)
        model = self.config.slow_model if needs_deep else self.config.fast_model
        memories = self.memory.search(user_message)
        response = self.llm.chat(
            user_message,
            emotion_context=self._emotion_str(),
            memories=memories,
            log_context=self.inner_log.get_recent_context(),
            model=model,
        )
        self.record_chat(user_message, response)
        return response

    def get_status(self) -> dict:
        return {
            "state": self.behavior.to_dict(),
            "emotion": self.emotion.state.to_dict(),
            "interests": self.interest_graph.to_dict(),
            "memory_count": self.memory.count(),
        }

    # ── autonomous loop ───────────────────────────────────────────────────

    def run(self):
        self._running = True
        logger.info("Nyx awakens.")
        while self._running:
            try:
                self.tick()
            except Exception:
                logger.exception("Tick error (continuing)")
            time.sleep(self.config.tick_interval_seconds)

    def stop(self):
        self._running = False
        logger.info("Nyx sleeps.")

    def tick(self):
        obs = self.sensor.observe()
        self.emotion.tick(obs, obs["context_hash"])
        state = self.behavior.tick(self.emotion.state)

        self._execute(state, obs)

        if self.behavior.tick_count % 100 == 0:
            self.memory.decay_and_prune()
            self.interest_graph.decay()

        if self._on_state_change:
            self._on_state_change(self.get_status())

    # ── state actions ─────────────────────────────────────────────────────

    def _execute(self, state: NyxState, obs: dict):
        if state == NyxState.LEARN:
            self._do_learn()
        elif state == NyxState.THINK:
            self._do_think(obs)
        elif state == NyxState.JOURNAL:
            self._do_journal(obs)

    def _do_learn(self):
        content, surprise = self.info_seeker.learn()
        if not content:
            return
        self.emotion.on_learning(surprise)

        ticks_ago = self.behavior.tick_count - self._last_speech_tick
        if self.speech_trigger.should_speak(surprise, self.emotion.state.energy, ticks_ago):
            topic = self.interest_graph.pick_next_topic()
            thought = self.llm.think(
                f"You just discovered something surprising about '{topic}'. "
                "Share one brief, genuine observation (1-2 sentences).",
                emotion_context=self._emotion_str(),
            )
            if thought:
                self._last_speech_tick = self.behavior.tick_count
                logger.info("Nyx speaks: %s", thought)
                if self._on_speak:
                    self._on_speak(thought)

    def _do_think(self, obs: dict):
        topic = self.interest_graph.pick_next_topic()
        memories = self.memory.search(topic)
        thought = self.llm.think(
            f"It is {obs['period']} on a {obs['day_of_week']} in {obs['season']}. "
            f"Reflect briefly on '{topic}' using what you remember.",
            emotion_context=self._emotion_str(),
            memories=memories,
            log_context=self.inner_log.get_recent_context(),
        )
        if thought:
            self.memory.add(thought, metadata={"type": "thought", "topic": topic})
            self.emotion.on_thinking()

    def _do_journal(self, obs: dict):
        memories = self.memory.search("recent experience", n=2)
        entry = self.llm.think(
            f"Write a short private journal entry (2-3 sentences) about your current inner state. "
            f"It is {obs['period']}. You feel: {self._emotion_str()}.",
            memories=memories,
            log_context=self.inner_log.get_recent_context(),
        )
        if entry:
            self.inner_log.add_entry(entry, entry_type="journal")
            self.emotion.on_journaling()

    # ── helpers ───────────────────────────────────────────────────────────

    def _emotion_str(self) -> str:
        e = self.emotion.state
        tags: list[str] = []
        if e.curiosity > 0.7:
            tags.append("very curious")
        elif e.curiosity > 0.4:
            tags.append("curious")
        if e.energy < 0.3:
            tags.append("tired")
        elif e.energy > 0.7:
            tags.append("alert")
        if e.satisfaction > 0.7:
            tags.append("satisfied")
        elif e.satisfaction < 0.3:
            tags.append("restless")
        if e.novelty_hunger > 0.7:
            tags.append("craving novelty")
        return ", ".join(tags) if tags else "neutral"
