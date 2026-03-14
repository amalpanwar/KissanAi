from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


API_BASE = "https://api.agmarknet.gov.in/v1/all-type-report/all-type-report"


def _build_url(params: dict[str, Any]) -> str:
    query = urlencode(params, doseq=True, safe="[],")
    return f"{API_BASE}?{query}"


def build_url(params: dict[str, Any]) -> str:
    return _build_url(params)


def _strip_brackets(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            out[k] = v[1:-1]
        else:
            out[k] = v
    return out


def fetch_page(
    params: dict[str, Any],
    timeout_sec: int = 30,
    retries: int = 3,
    allow_unbracketed: bool = True,
) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://agmarknet.gov.in/",
        "Origin": "https://agmarknet.gov.in",
    }
    last_error: Exception | None = None
    tried_unbracketed = False
    url = _build_url(params)
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if allow_unbracketed and exc.code in {400, 404} and not tried_unbracketed:
                tried_unbracketed = True
                url = _build_url(_strip_brackets(params))
                continue
            if exc.code in {400, 404}:
                # Treat as empty page (invalid combo) and continue caller loop.
                return {"rows": []}
            if attempt == retries:
                raise
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
        time.sleep(1.5 * attempt)
    if last_error:
        raise last_error
    return {}


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "rows" in payload and isinstance(payload["rows"], list):
        return payload["rows"]
    if "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    if "records" in payload and isinstance(payload["records"], list):
        return payload["records"]
    # fallback: search for first list value
    for v in payload.values():
        if isinstance(v, list):
            return v
    return []
