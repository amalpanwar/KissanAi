from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


API_BASE = "https://api.data.gov.in/resource"


class DataGovClient:
    def __init__(self, api_key: str, timeout_sec: int = 20, retries: int = 3) -> None:
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.retries = retries

    def fetch_records(
        self,
        resource_id: str,
        limit: int = 1000,
        max_records: int = 100000,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        offset = 0
        all_records: list[dict[str, Any]] = []
        extras = extra_params or {}
        total_hint: int | None = None

        while True:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": limit,
                "offset": offset,
            }
            params.update(extras)
            query = urlencode(params, doseq=True)
            url = f"{API_BASE}/{resource_id}?{query}"

            payload = None
            for attempt in range(1, self.retries + 1):
                try:
                    with urlopen(url, timeout=self.timeout_sec) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    break
                except Exception:
                    if attempt == self.retries:
                        raise
                    time.sleep(1.5 * attempt)
            if payload is None:
                break

            records = payload.get("records", [])
            # data.gov sometimes reports one of these as total available row count.
            if total_hint is None:
                for key in ("total", "count"):
                    val = payload.get(key)
                    if isinstance(val, int) and val > 0:
                        total_hint = val
                        break
            if not records:
                break

            all_records.extend(records)
            offset += len(records)

            # Primary stop condition: reached server-reported total.
            if total_hint is not None and offset >= total_hint:
                break
            # Secondary stop condition when server gives fewer than requested and no total hint.
            if total_hint is None and len(records) < limit:
                break
            if len(all_records) >= max_records:
                all_records = all_records[:max_records]
                break

        return all_records
