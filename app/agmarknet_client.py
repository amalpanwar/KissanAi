from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://api.agmarknet.gov.in/v1/all-type-report/all-type-report"


def fetch_page(
    params: dict[str, Any],
    timeout_sec: int = 30,
    retries: int = 3,
) -> dict[str, Any]:
    query = urlencode(params, doseq=True)
    url = f"{API_BASE}?{query}"
    last_error: Exception | None = None
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://agmarknet.gov.in/",
        "Origin": "https://agmarknet.gov.in",
    }
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)
    if last_error:
        raise last_error
    return {}


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    if "records" in payload and isinstance(payload["records"], list):
        return payload["records"]
    # fallback: search for first list value
    for v in payload.values():
        if isinstance(v, list):
            return v
    return []
