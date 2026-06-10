from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EmotionState:
    curiosity: float = 0.5
    energy: float = 0.7
    satisfaction: float = 0.5
    novelty_hunger: float = 0.3

    def to_dict(self) -> dict:
        return {
            "curiosity": round(self.curiosity, 3),
            "energy": round(self.energy, 3),
            "satisfaction": round(self.satisfaction, 3),
            "novelty_hunger": round(self.novelty_hunger, 3),
        }


class EmotionSystem:
    def __init__(self, config):
        self.config = config
        self.state = EmotionState()
        self._last_context_hash: str = ""

    def tick(self, system_stats: dict, context_hash: str):
        hour = datetime.now().hour
        self._update_energy(hour, system_stats.get("cpu_percent", 0.0))
        self._update_curiosity()
        self._update_satisfaction()
        self._update_novelty_hunger(context_hash)

    def _update_energy(self, hour: int, cpu_percent: float):
        # Circadian rhythm: peak ~10am, trough ~10pm
        t = hour / 24
        circadian = 0.5 + self.config.energy_circadian_amplitude * math.cos(
            2 * math.pi * (t - 10 / 24)
        )
        circadian = max(0.1, min(1.0, circadian))

        cpu_drain = (cpu_percent / 100) * 0.08
        self.state.energy += (circadian - self.state.energy) * 0.08 - cpu_drain
        self.state.energy = max(0.0, min(1.0, self.state.energy))

    def _update_curiosity(self):
        # Curiosity rises naturally — Nyx always wants to know more
        self.state.curiosity = min(1.0, self.state.curiosity + self.config.curiosity_rise_rate)

    def _update_satisfaction(self):
        self.state.satisfaction = max(
            0.0, self.state.satisfaction - self.config.satisfaction_decay_rate
        )

    def _update_novelty_hunger(self, context_hash: str):
        if context_hash == self._last_context_hash:
            # Same environment → boredom grows
            self.state.novelty_hunger = min(
                1.0, self.state.novelty_hunger + self.config.novelty_hunger_rise_rate
            )
        else:
            self.state.novelty_hunger = max(0.0, self.state.novelty_hunger - 0.2)
        self._last_context_hash = context_hash

    # --- callbacks from NyxCore after actions ---

    def on_learning(self, surprise_score: float):
        self.state.curiosity = max(0.0, self.state.curiosity - 0.3)
        self.state.satisfaction = min(1.0, self.state.satisfaction + surprise_score * 0.4)
        self.state.novelty_hunger = max(0.0, self.state.novelty_hunger - 0.3)

    def on_thinking(self):
        self.state.satisfaction = min(1.0, self.state.satisfaction + 0.1)

    def on_journaling(self):
        self.state.satisfaction = min(1.0, self.state.satisfaction + 0.15)
        self.state.curiosity = max(0.0, self.state.curiosity - 0.05)
