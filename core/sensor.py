from __future__ import annotations
import hashlib
import psutil
from datetime import datetime

_SEASONS: list[tuple[tuple[int, ...], str]] = [
    ((3, 4, 5), "spring"),
    ((6, 7, 8), "summer"),
    ((9, 10, 11), "autumn"),
    ((12, 1, 2), "winter"),
]

_TIME_PERIODS: list[tuple[int, int, str]] = [
    (5, 12, "morning"),
    (12, 17, "afternoon"),
    (17, 21, "evening"),
    (21, 24, "night"),
    (0, 5, "deep night"),
]


class SensorSystem:
    def get_time_context(self) -> dict:
        now = datetime.now()
        hour, month = now.hour, now.month

        season = next(s for months, s in _SEASONS if month in months)
        period = next(p for start, end, p in _TIME_PERIODS if start <= hour < end)

        return {
            "hour": hour,
            "day_of_week": now.strftime("%A"),
            "period": period,
            "season": season,
            "datetime_str": now.strftime("%Y-%m-%d %H:%M"),
        }

    def get_system_stats(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.3),
            "memory_percent": mem.percent,
            "memory_available_gb": round(mem.available / 1024**3, 2),
        }

    def _context_hash(self, time_ctx: dict, sys_stats: dict) -> str:
        # Changes every ~10 minutes and when CPU load band shifts
        snapshot = f"{time_ctx['hour']}{time_ctx['period']}{int(sys_stats['cpu_percent'] / 20)}"
        return hashlib.md5(snapshot.encode()).hexdigest()[:8]

    def observe(self) -> dict:
        time_ctx = self.get_time_context()
        sys_stats = self.get_system_stats()
        return {
            **time_ctx,
            **sys_stats,
            "context_hash": self._context_hash(time_ctx, sys_stats),
        }
