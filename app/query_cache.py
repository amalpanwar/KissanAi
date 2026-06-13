from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path


class QueryResponseCache:
    def __init__(self, cache_path: str | Path, *, version: str = "v1", max_entries: int = 256) -> None:
        self.cache_path = Path(cache_path)
        self.version = str(version)
        self.max_entries = max(16, int(max_entries))

    def _load(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, payload: dict[str, dict]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def make_key(
        self,
        *,
        question: str,
        context_part: str = "",
        route: str = "",
        model_name: str = "",
    ) -> str:
        raw = "||".join(
            [
                self.version,
                str(question or "").strip().lower(),
                str(context_part or "").strip().lower(),
                str(route or "").strip().lower(),
                str(model_name or "").strip().lower(),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        *,
        question: str,
        context_part: str = "",
        route: str = "",
        model_name: str = "",
    ) -> dict | None:
        payload = self._load()
        key = self.make_key(question=question, context_part=context_part, route=route, model_name=model_name)
        item = payload.get(key) or {}
        if not item:
            return None
        ttl = int(item.get("ttl_sec") or 0)
        ts = float(item.get("ts") or 0.0)
        if ttl <= 0 or (time.time() - ts) > ttl:
            payload.pop(key, None)
            self._save(payload)
            return None
        result = copy.deepcopy(item.get("result"))
        if isinstance(result, dict):
            result["cache_hit"] = True
        return result

    def put(
        self,
        *,
        question: str,
        result: dict,
        context_part: str = "",
        route: str = "",
        model_name: str = "",
        ttl_sec: int = 6 * 60 * 60,
    ) -> None:
        if ttl_sec <= 0:
            return
        payload = self._load()
        key = self.make_key(question=question, context_part=context_part, route=route, model_name=model_name)
        clean_result = copy.deepcopy(result)
        if isinstance(clean_result, dict):
            clean_result.pop("cache_hit", None)
        payload[key] = {
            "ts": time.time(),
            "ttl_sec": int(ttl_sec),
            "route": route,
            "model_name": model_name,
            "result": clean_result,
        }
        live_items = sorted(
            payload.items(),
            key=lambda item: float((item[1] or {}).get("ts") or 0.0),
            reverse=True,
        )
        trimmed = dict(live_items[: self.max_entries])
        self._save(trimmed)
