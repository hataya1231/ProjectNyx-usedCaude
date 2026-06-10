from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from core.emotion import EmotionState


class NyxState(Enum):
    SLEEP = auto()
    OBSERVE = auto()
    THINK = auto()
    LEARN = auto()
    JOURNAL = auto()


@dataclass
class BehaviorContext:
    state: NyxState = NyxState.OBSERVE
    ticks_in_state: int = 0
    tick_count: int = 0


class BehaviorEngine:
    def __init__(self):
        self.ctx = BehaviorContext()

    def tick(self, emotion: EmotionState) -> NyxState:
        self.ctx.tick_count += 1
        self.ctx.ticks_in_state += 1

        next_state = self._evaluate(emotion)
        if next_state != self.ctx.state:
            self.ctx.state = next_state
            self.ctx.ticks_in_state = 0

        return self.ctx.state

    def _evaluate(self, e: EmotionState) -> NyxState:
        # Sleep dominates when energy is critically low
        if e.energy < 0.2:
            return NyxState.SLEEP

        # Wake from sleep only after sufficient recovery
        if self.ctx.state == NyxState.SLEEP:
            return NyxState.SLEEP if e.energy < 0.4 else NyxState.OBSERVE

        # High curiosity or boredom → go learn something new
        if (e.curiosity > 0.75 or e.novelty_hunger > 0.7) and e.energy > 0.35:
            return NyxState.LEARN

        # Low satisfaction after dwelling in a state → introspect
        if e.satisfaction < 0.25 and e.energy > 0.35 and self.ctx.ticks_in_state > 4:
            return NyxState.JOURNAL

        # Moderate curiosity → reflect on existing memories
        if e.curiosity > 0.45 and e.energy > 0.3:
            return NyxState.THINK

        return NyxState.OBSERVE

    @property
    def current_state(self) -> NyxState:
        return self.ctx.state

    @property
    def tick_count(self) -> int:
        return self.ctx.tick_count

    def to_dict(self) -> dict:
        return {
            "state": self.ctx.state.name.lower(),
            "ticks_in_state": self.ctx.ticks_in_state,
            "tick_count": self.ctx.tick_count,
        }
