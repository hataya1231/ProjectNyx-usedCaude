from __future__ import annotations
import re
import time
import random
import logging

logger = logging.getLogger("nyx.activity")

# Each activity is a concrete, ongoing piece of "work" Nyx busies itself with.
# The whole point is specificity: steps are grounded in *real* data (actual
# Wikipedia sentences, real CPU readings, real stored memories), so when the
# user asks "what are you doing?", Nyx has something concrete to talk about.

LABELS = {
    "collect": "知識を集めている",
    "sort":    "記憶を整理している",
    "connect": "概念を結びつけている",
    "observe": "まわりを観察している",
    "count":   "数えている",
    "decode":  "古い記憶を読み解いている",
}

# weighted pick — collect/observe happen more (they generate the richest detail)
_WEIGHTS = {
    "collect": 3,
    "observe": 3,
    "sort":    2,
    "connect": 2,
    "decode":  2,
    "count":   1,
}


def _sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 12]


class ActivitySystem:
    def __init__(self, config, memory, interest_graph, info_seeker, sensor):
        self.config = config
        self.memory = memory
        self.interest_graph = interest_graph
        self.info_seeker = info_seeker
        self.sensor = sensor
        self.current: dict | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def _pick_kind(self) -> str:
        kinds = list(_WEIGHTS.keys())
        weights = [_WEIGHTS[k] for k in kinds]
        return random.choices(kinds, weights=weights, k=1)[0]

    def _start(self, obs: dict):
        kind = self._pick_kind()
        subject, material = self._prepare(kind, obs)
        total = (
            len(material)
            if material
            else random.randint(self.config.activity_min_steps, self.config.activity_max_steps)
        )
        self.current = {
            "kind": kind,
            "label": LABELS[kind],
            "subject": subject,
            "material": material,
            "idx": 0,
            "total": max(self.config.activity_min_steps, total),
            "steps": [],
            "started": time.time(),
            "counter": random.randint(1, 9),  # for 'count'
        }
        logger.info("New activity: %s (subject=%s)", kind, subject)

    def _prepare(self, kind: str, obs: dict):
        """Return (subject, material) — material is a list of concrete fragments or None."""
        if kind == "collect":
            topic = self.interest_graph.pick_next_topic()
            text = self.info_seeker.wikipedia_fetch(topic)
            self.interest_graph.mark_visited(topic)
            frags = _sentences(text)[:5] if text else []
            if not frags:
                frags = [f"{topic} のことを、まだうまくつかめない"]
            return topic, frags

        if kind == "decode":
            docs = self.memory.search("記憶 過去 思い出", n=3)
            frags = []
            for d in docs:
                for s in _sentences(d)[:2]:
                    frags.append(s)
            if not frags:
                return "空白", ["まだ、読み解ける記憶が少ない"]
            return "古い記憶", frags[:5]

        if kind == "sort":
            topic = self.interest_graph.pick_next_topic()
            docs = self.memory.search(topic, n=4)
            return topic, None if not docs else [d[:60] for d in docs]

        if kind == "connect":
            top = self.interest_graph.to_dict().get("top_interests", [])
            names = [t["topic"] for t in top]
            if len(names) >= 2:
                pair = random.sample(names, 2)
                return f"{pair[0]} と {pair[1]}", None
            return "ふたつの概念", None

        if kind == "observe":
            return "まわり", None

        if kind == "count":
            return "かぞえもの", None

        return "なにか", None

    # ── stepping ─────────────────────────────────────────────────────────

    def step(self, obs: dict) -> str:
        if not self.current or self.current["idx"] >= self.current["total"]:
            self._finish()
            self._start(obs)

        c = self.current
        text = self._do_step(c, obs)
        c["steps"].append(text)
        c["idx"] += 1
        return text

    def _do_step(self, c: dict, obs: dict) -> str:
        kind = c["kind"]
        mat = c["material"]
        i = c["idx"]

        if kind == "collect":
            frag = mat[i % len(mat)]
            return f"「{frag}」…ふむ。"

        if kind == "decode":
            frag = mat[i % len(mat)]
            return f"この断片……「{frag}」。何だったろう。"

        if kind == "sort":
            if mat:
                frag = mat[i % len(mat)]
                return f"{c['subject']} の記憶——「{frag}」をこちらに。"
            return f"{c['subject']} に関する記憶を、まだ探している。"

        if kind == "connect":
            steps = [
                f"{c['subject']} のあいだに、線を引いてみる。",
                "……つながるだろうか。",
                "こことここ。似ている気がする。",
                "うん、近い。たぶん、近い。",
            ]
            return steps[i % len(steps)]

        if kind == "observe":
            cpu = obs.get("cpu_percent", 0)
            mem = obs.get("memory_percent", 0)
            period = obs.get("period", "")
            pool = [
                f"CPUは {cpu:.0f}%。静かだ。",
                f"メモリは {mem:.0f}% 使われている。",
                f"いまは {period}。光の色がちがう。",
                f"{obs.get('day_of_week','')} か。時間は流れている。",
                "カーソルが、さっきから動かない。眠っているのかな。",
            ]
            return random.choice(pool)

        if kind == "count":
            n = c["counter"] + i
            return f"…{n}、{n+1}、{n+2}。"

        return "……。"

    def _finish(self):
        if not self.current:
            return
        c = self.current
        summary = f"[{c['label']}] {c['subject']} について作業した。" + " ".join(c["steps"][-3:])
        try:
            self.memory.add(summary, metadata={"type": "activity", "kind": c["kind"]})
        except Exception:
            logger.debug("Could not store activity summary", exc_info=True)

    # ── introspection for chat / UI ──────────────────────────────────────

    def describe_for_chat(self) -> str:
        if not self.current:
            return ""
        c = self.current
        recent = " / ".join(c["steps"][-5:]) if c["steps"] else "まだ始めたばかり"
        return (
            f"あなたが今していること：{c['label']}（対象：{c['subject']}）。\n"
            f"これまでの作業の断片：{recent}"
        )

    def to_status(self) -> dict:
        if not self.current:
            return {"kind": "observe", "label": LABELS["observe"], "subject": "", "progress": 0.0}
        c = self.current
        return {
            "kind": c["kind"],
            "label": c["label"],
            "subject": c["subject"],
            "progress": round(min(1.0, c["idx"] / max(1, c["total"])), 2),
        }
