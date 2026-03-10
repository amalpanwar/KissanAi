from __future__ import annotations

import json
import time
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


API_BASE = "https://api.data.gov.in/resource"


class DataGovClient:
    def __init__(self, api_key: str, timeout_sec: int | None = 20, retries: int = 3) -> None:
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.retries = retries

    def fetch_records(
        self,
        resource_id: str,
        limit: int = 1000,
        max_records: int = 100000,
        extra_params: dict[str, Any] | None = None,
        stop_date: date | None = None,
        date_field: str = "Arrival_Date",
        dayfirst: bool = True,
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
            last_error: Exception | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    if self.timeout_sec is None:
                        resp_ctx = urlopen(url)
                    else:
                        resp_ctx = urlopen(url, timeout=self.timeout_sec)
                    with resp_ctx as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == self.retries:
                        # Return what we have so far instead of hard-failing.
                        return all_records
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

            if stop_date:
                filtered: list[dict[str, Any]] = []
                min_dt: date | None = None
                for r in records:
                    dt = _parse_date(r.get(date_field), dayfirst=dayfirst)
                    if dt:
                        if min_dt is None or dt < min_dt:
                            min_dt = dt
                        if dt >= stop_date:
                            filtered.append(r)
                    else:
                        # Keep rows without dates to avoid losing data.
                        filtered.append(r)
                records = filtered
                if min_dt is not None and min_dt < stop_date:
                    all_records.extend(records)
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


def _parse_date(val: Any, dayfirst: bool = True) -> date | None:
    if not val:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    s = str(val).strip()
    if not s:
        return None
    try:
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 3:
                if dayfirst:
                    d, m, y = parts
                else:
                    m, d, y = parts
                return date(int(y), int(m), int(d))
        if "-" in s:
            parts = s.split("-")
            if len(parts) == 3:
                y, m, d = parts
                return date(int(y), int(m), int(d))
    except Exception:
        return None
    return None
