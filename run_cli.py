"""
CLI test runner — start Nyx without the web server.
Useful for verifying core behavior before wiring up FastAPI.

Usage:
    cd nyx
    python run_cli.py
"""
import logging
import threading
from core.nyx import NyxCore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


_TONE_COLOR = {
    "normal":  "\033[37m",   # white
    "recall":  "\033[36m",   # cyan — remembering
    "dream":   "\033[35m",   # magenta — dreaming
    "special": "\033[33m",   # yellow — rare moment
}


def on_thought(thought: str, tone: str = "normal"):
    color = _TONE_COLOR.get(tone, "\033[37m")
    tag = "" if tone == "normal" else f"({tone}) "
    print(f"\n{color}[Nyx] {tag}{thought}\033[0m\n>>> ", end="", flush=True)


def on_activity(status: dict):
    focus = f" / 夢中: {status['focus']}" if status.get("focus") else ""
    print(f"\n\033[90m— いま: {status['label']}（{status['subject']}）{focus} —\033[0m\n>>> ",
          end="", flush=True)


nyx = NyxCore()
nyx.on_thought(on_thought)
nyx.on_activity(on_activity)

thread = threading.Thread(target=nyx.run, daemon=True, name="nyx-core")
thread.start()

print("Nyx is running. Type a message and press Enter. Type 'quit' to exit.\n")

try:
    while True:
        line = input(">>> ").strip()
        if line.lower() in ("quit", "exit", "q"):
            break
        if line:
            response = nyx.chat(line)
            print(f"\033[32mNyx:\033[0m {response}")
except (KeyboardInterrupt, EOFError):
    pass

nyx.stop()
print("\nNyx sleeps.")
