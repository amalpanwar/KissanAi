from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


API_BASE = "https://api.agmarknet.gov.in/v1/dashboard-data/"

# Agmarknet's current frontend uses these sentinel IDs for "all ..." filters.
ALL_GROUPS = 100000
ALL_COMMODITIES = 100001
ALL_MARKET_TYPES = 100004
ALL_DISTRICTS = 100007
ALL_MARKETS = 100009
ALL_GRADES = 100011
ALL_VARIETIES = 100021


def build_url(body: dict[str, Any]) -> str:
    page = body.get("page", 1)
    date = body.get("date") or f"{body.get('from_date', '')}:{body.get('to_date', '')}"
    state = body.get("state")
    return f"{API_BASE}?page={page}&date={date}&state={state}"


def build_snapshot_payload(
    *,
    state_id: str | int,
    date_str: str,
    page: int = 1,
    limit: int = 200,
    district_ids: list[str] | None = None,
) -> dict[str, Any]:
    districts = district_ids or [ALL_DISTRICTS]
    return {
        "dashboard": "cumm_data_sp",
        "date": date_str,
        "group": [ALL_GROUPS],
        "commodity": [ALL_COMMODITIES],
        "state": [int(state_id)],
        "district": [int(x) for x in districts],
        "market": [ALL_MARKETS],
        "market_type": [ALL_MARKET_TYPES],
        "grades": [ALL_GRADES],
        "variety": [ALL_VARIETIES],
        "page": int(page),
        "limit": int(limit),
        "format": "json",
        "type": "M",
    }


def fetch_page(
    body: dict[str, Any],
    timeout_sec: int = 30,
    retries: int = 3,
) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://agmarknet.gov.in/",
        "Origin": "https://agmarknet.gov.in",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    raw = json.dumps(body).encode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            req = Request(API_BASE, headers=headers, data=raw, method="POST")
            with urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            try:
                err_payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                err_payload = {}
            message = str(err_payload.get("message") or "")
            if exc.code == 404 and "No data available" in message:
                return {"status": False, "message": message, "pagination": {}, "data": {"records": []}}
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
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload.get("records"), list):
        return payload["records"]
    return []
