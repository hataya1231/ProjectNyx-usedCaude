from dataclasses import dataclass


@dataclass
class NyxConfig:
    # LLM — dual model system
    # fast_model: used for instant replies and internal reasoning
    # slow_model: used when the question requires deeper thought
    #   → run: ollama pull qwen2.5:3b  to enable the slow model
    fast_model: str = "qwen2.5:1.5b"
    slow_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"

    # Deep-thought threshold: message length (chars) above which slow model kicks in
    deep_thought_char_threshold: int = 50

    # Behavior loop
    tick_interval_seconds: int = 18    # frequent, lively monologue

    # Monologue: fraction of ticks where the LLM rephrases the concrete step
    # (the rest emit the raw concrete observation, guaranteeing specificity)
    monologue_llm_ratio: float = 0.55
    # How many concrete steps each activity runs before Nyx moves on
    activity_min_steps: int = 3
    activity_max_steps: int = 6

    # ── Watchability (never gets boring) ──
    # ① Obsession: Nyx fixates on one theme for a while, then drifts
    focus_min_ticks: int = 18
    focus_max_ticks: int = 40
    memory_callback_chance: float = 0.18   # chance to recall its own past
    # ② Dreams while asleep
    dream_chance: float = 0.22
    # ④ Rare special moments ("did you see that?")
    special_event_chance: float = 0.03

    # Emotion (rates per tick)
    curiosity_rise_rate: float = 0.03
    satisfaction_decay_rate: float = 0.01
    novelty_hunger_rise_rate: float = 0.05
    energy_circadian_amplitude: float = 0.4

    # Memory
    chroma_persist_dir: str = "./data/chromadb"
    memory_max_results: int = 3          # keep prompts small for tiny LLMs
    memory_decay_rate: float = 0.05      # strength lost per day of inactivity
    memory_prune_threshold: float = 0.1
    memory_reinforce_amount: float = 0.2

    # Interest graph
    interest_weight_decay: float = 0.02  # per decay() call
    max_related_concepts: int = 5

    # Inner log
    max_log_context_entries: int = 3

    # Speech trigger (Nyx speaks unprompted only when conditions are met)
    surprise_threshold: float = 0.65
    active_speech_energy_min: float = 0.45
    min_ticks_between_speech: int = 10

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000


CONFIG = NyxConfig()
