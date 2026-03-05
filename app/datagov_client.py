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
            if not records:
                break

            all_records.extend(records)
            offset += len(records)

            if len(records) < limit:
                break
            if len(all_records) >= max_records:
                all_records = all_records[:max_records]
                break

        return all_records
