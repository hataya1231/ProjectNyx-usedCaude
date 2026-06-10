class SpeechTrigger:
    """
    Controls when Nyx speaks unprompted.
    Nyx is not a chatbot — it surfaces thoughts rarely, only when something
    genuinely surprising is discovered AND conditions are right.
    """

    def __init__(self, config):
        self.config = config

    def should_speak(
        self,
        surprise_score: float,
        energy: float,
        ticks_since_last_speech: int,
    ) -> bool:
        if energy < self.config.active_speech_energy_min:
            return False
        if surprise_score < self.config.surprise_threshold:
            return False
        if ticks_since_last_speech < self.config.min_ticks_between_speech:
            return False
        return True
