from __future__ import annotations
import asyncio
import json
import logging
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from core.nyx import NyxCore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

app = FastAPI(title="Nyx")
nyx = NyxCore()

_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


async def _broadcast(data: dict):
    if not _clients:
        return
    message = json.dumps(data, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for ws in list(_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _clients -= dead


def _threadsafe_broadcast(data: dict):
    if _loop:
        asyncio.run_coroutine_threadsafe(_broadcast(data), _loop)


def _on_thought(thought: str, tone: str = "normal"):
    _threadsafe_broadcast({"type": "thought", "content": thought, "tone": tone})


def _on_activity(status: dict):
    _threadsafe_broadcast({"type": "activity", **status})


def _on_artifact(info: dict):
    _threadsafe_broadcast({"type": "artifact", **info})


def _on_state_change(status: dict):
    _threadsafe_broadcast({"type": "state", **status})


nyx.on_thought(_on_thought)
nyx.on_activity(_on_activity)
nyx.on_artifact(_on_artifact)
nyx.on_state_change(_on_state_change)


@app.on_event("startup")
async def startup():
    global _loop
    _loop = asyncio.get_running_loop()
    thread = threading.Thread(target=nyx.run, daemon=True, name="nyx-core")
    thread.start()


@app.on_event("shutdown")
async def shutdown():
    nyx.stop()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    await ws.send_text(json.dumps({"type": "state", **nyx.get_status()}))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "chat":
                continue

            user_message: str = msg.get("content", "").strip()
            if not user_message:
                continue

            # --- build context (runs fast: memory lookup only) ---
            messages, needs_deep = await asyncio.get_event_loop().run_in_executor(
                None, nyx.build_chat_context, user_message
            )
            model = nyx.llm.slow_model if needs_deep else nyx.llm.fast_model

            # --- if deep thought: send acknowledgement first ---
            if needs_deep:
                ack = nyx.llm.deep_thought_ack()
                await ws.send_text(json.dumps({
                    "type": "deep_ack",
                    "content": ack,
                }))
                await ws.send_text(json.dumps({"type": "thinking"}))

            # --- stream the actual response token by token ---
            await ws.send_text(json.dumps({"type": "chat_start"}))
            full_response = ""

            async for token in nyx.llm.stream(messages, model):
                full_response += token
                await ws.send_text(json.dumps({
                    "type": "token",
                    "content": token,
                }))

            await ws.send_text(json.dumps({"type": "chat_done"}))

            # --- store the exchange in memory (non-blocking) ---
            asyncio.get_event_loop().run_in_executor(
                None, nyx.record_chat, user_message, full_response
            )

    except WebSocketDisconnect:
        _clients.discard(ws)


@app.get("/state")
async def get_state():
    return nyx.get_status()


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
