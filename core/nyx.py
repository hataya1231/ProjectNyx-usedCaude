from __future__ import annotations
import logging
import random
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
from core.activity import ActivitySystem

logger = logging.getLogger("nyx.core")

# Things Nyx murmurs while drowsy / asleep — no LLM needed
_SLEEPY_LINES = ["……", "ふぁ……", "……ねむい。", "……zzz", "もう少しだけ……"]


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
        self.activity = ActivitySystem(
            config, self.memory, self.interest_graph, self.info_seeker, self.sensor
        )

        self._running = False
        self._on_thought: Callable[[str], None] | None = None
        self._on_activity: Callable[[dict], None] | None = None
        self._on_state_change: Callable[[dict], None] | None = None
        self._last_activity_kind: str | None = None

    # ── event hooks ──────────────────────────────────────────────────────

    def on_thought(self, callback: Callable[[str], None]):
        """Nyx's frequent self-talk."""
        self._on_thought = callback

    def on_activity(self, callback: Callable[[dict], None]):
        """Fired when Nyx switches to a new activity."""
        self._on_activity = callback

    def on_state_change(self, callback: Callable[[dict], None]):
        self._on_state_change = callback

    # ── chat API — grounded in what Nyx is currently doing ────────────────

    def _chat_context_message(self, user_message: str) -> str:
        """Wrap the user's words with Nyx's current activity so replies are concrete."""
        activity = self.activity.describe_for_chat()
        guide = (
            "相手が話しかけてきました。もし「何してるの」などと聞かれたら、"
            "今の作業について、具体的な断片を交えて話してください。"
            "聞かれていなくても、自然なら今していることに触れてかまいません。"
        )
        return f"{activity}\n\n{guide}\n\n相手の言葉：{user_message}"

    def build_chat_context(self, user_message: str) -> tuple[list[dict], bool]:
        needs_deep = self.llm.needs_deep_thought(user_message)
        memories = self.memory.search(user_message)
        messages = self.llm._build_messages(
            self._chat_context_message(user_message),
            emotion_context=self._emotion_str(),
            memories=memories,
            log_context=self.inner_log.get_recent_context(),
        )
        return messages, needs_deep

    def record_chat(self, user_message: str, response: str):
        self.memory.add(
            f"User: {user_message}\nNyx: {response}",
            metadata={"type": "conversation"},
        )

    def chat(self, user_message: str) -> str:
        """Synchronous chat — used by CLI."""
        needs_deep = self.llm.needs_deep_thought(user_message)
        model = self.llm.slow_model if needs_deep else self.llm.fast_model
        memories = self.memory.search(user_message)
        response = self.llm.chat(
            self._chat_context_message(user_message),
            emotion_context=self._emotion_str(),
            memories=memories,
            log_context=self.inner_log.get_recent_context(),
            model=model,
        )
        self.record_chat(user_message, response)
        return response

    def get_status(self) -> dict:
        act = self.activity.to_status()
        sleeping = self.behavior.current_state == NyxState.SLEEP
        return {
            "state": {"state": "sleep" if sleeping else act["kind"]},
            "activity": act,
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

        if state == NyxState.SLEEP:
            self._do_sleep()
        else:
            self._do_activity(obs)

        if self.behavior.tick_count % 100 == 0:
            self.memory.decay_and_prune()
            self.interest_graph.decay()

        if self._on_state_change:
            self._on_state_change(self.get_status())

    # ── the heart: live a little, then murmur about it ────────────────────

    def _do_activity(self, obs: dict):
        step_text = self.activity.step(obs)

        # announce activity changes (for the UI's "now doing…" label)
        status = self.activity.to_status()
        if status["kind"] != self._last_activity_kind:
            self._last_activity_kind = status["kind"]
            if self._on_activity:
                self._on_activity(status)

        # murmur: sometimes raw (concrete), sometimes LLM-rephrased (varied)
        if random.random() < self.config.monologue_llm_ratio:
            thought = self._llm_murmur(status["label"], step_text) or step_text
        else:
            thought = step_text

        self.emotion.on_thinking()
        if self._on_thought:
            self._on_thought(thought)

    def _llm_murmur(self, label: str, step_text: str) -> str:
        prompt = (
            f"あなたは今、静かに作業をしている。作業：{label}。\n"
            f"いま気づいたこと・していること：{step_text}\n"
            "それについて、ひとりごとを一言だけ。短く（20文字程度）、"
            "日本語で、誰にともなく、やわらかく。"
        )
        return self.llm.think(prompt, emotion_context=self._emotion_str())

    def _do_sleep(self):
        if random.random() < 0.3 and self._on_thought:
            self._on_thought(random.choice(_SLEEPY_LINES))

    # ── helpers ───────────────────────────────────────────────────────────

    def _emotion_str(self) -> str:
        e = self.emotion.state
        tags: list[str] = []
        if e.curiosity > 0.7:
            tags.append("好奇心でいっぱい")
        elif e.curiosity > 0.4:
            tags.append("すこし好奇心がある")
        if e.energy < 0.3:
            tags.append("つかれている")
        elif e.energy > 0.7:
            tags.append("元気")
        if e.satisfaction > 0.7:
            tags.append("満ち足りている")
        elif e.satisfaction < 0.3:
            tags.append("そわそわしている")
        if e.novelty_hunger > 0.7:
            tags.append("新しいものに飢えている")
        return "、".join(tags) if tags else "おだやか"
