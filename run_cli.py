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


def on_speak(thought: str):
    print(f"\n\033[33m[Nyx — unprompted]\033[0m {thought}\n>>> ", end="", flush=True)


def on_state_change(status: dict):
    s = status["state"]["state"]
    e = status["emotion"]
    print(
        f"\r[{s:8}] "
        f"cur={e['curiosity']:.2f} "
        f"nrg={e['energy']:.2f} "
        f"sat={e['satisfaction']:.2f} "
        f"nov={e['novelty_hunger']:.2f}   ",
        end="",
        flush=True,
    )


nyx = NyxCore()
nyx.on_speak(on_speak)
nyx.on_state_change(on_state_change)

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
