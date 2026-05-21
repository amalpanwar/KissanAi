from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPORT_API_BASE = "https://api.agmarknet.gov.in/v1/all-type-report/all-type-report-agm"
DASHBOARD_API_BASE = "https://api.agmarknet.gov.in/v1/dashboard-data/"

# Agmarknet's current dashboard frontend uses these sentinel IDs for "all ..." filters.
ALL_GROUPS = 100000
ALL_COMMODITIES = 100001
ALL_MARKET_TYPES = 100004
ALL_DISTRICTS = 100007
ALL_MARKETS = 100009
ALL_GRADES = 100011
ALL_VARIETIES = 100021


def build_url(body: dict[str, Any]) -> str:
    mode = str(body.get("_agmarknet_mode") or "").strip().lower()
    if mode == "report" or ("from_date" in body and "dashboard" not in body):
        params = {k: v for k, v in body.items() if not str(k).startswith("_")}
        query = urlencode(params, doseq=True)
        return f"{REPORT_API_BASE}?{query}"
    page = body.get("page", 1)
    date = body.get("date") or f"{body.get('from_date', '')}:{body.get('to_date', '')}"
    state = body.get("state")
    return f"{DASHBOARD_API_BASE}?page={page}&date={date}&state={state}"


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
        "_agmarknet_mode": "dashboard",
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


def build_report_payload(
    *,
    state_id: str | int,
    district_id: str | int | None,
    group_id: str | int,
    commodity_id: str | int | None,
    from_date: str,
    to_date: str,
    option: str | int = "2",
    page: int = 1,
    limit: int = 100,
    period: str = "date",
    type_value: str = "3",
    msp: str = "0",
) -> dict[str, Any]:
    district_txt = str(district_id).strip() if district_id is not None else ""
    commodity_txt = str(commodity_id).strip() if commodity_id is not None else ""
    return {
        "_agmarknet_mode": "report",
        "type": int(type_value),
        "from_date": str(from_date),
        "to_date": str(to_date),
        "msp": int(msp),
        "period": str(period),
        "group": [int(group_id)],
        "commodity": [int(commodity_txt)] if commodity_txt else [],
        "state": [int(state_id)],
        "district": [int(district_txt)] if district_txt else [],
        "market": [],
        "grade": [],
        "page": int(page),
        "options": int(option),
        "itemsPerPage": int(limit),
        "max": int(limit),
        "export": "No",
    }


def _fetch_dashboard_page(
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
    payload = {k: v for k, v in body.items() if not str(k).startswith("_")}
    raw = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            req = Request(DASHBOARD_API_BASE, headers=headers, data=raw, method="POST")
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


def _fetch_report_page(
    params: dict[str, Any],
    timeout_sec: int = 30,
    retries: int = 3,
) -> dict[str, Any]:
    payload = {k: v for k, v in params.items() if not str(k).startswith("_")}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://agmarknet.gov.in/all-type-of-report-table",
        "Origin": "https://agmarknet.gov.in",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    raw = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            req = Request(REPORT_API_BASE, headers=headers, data=raw, method="POST")
            with urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            try:
                err_payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                err_payload = {}
            if exc.code in {400, 404}:
                return {"data": [], "records": [], "message": err_payload.get("message") or ""}
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


def fetch_page(
    body: dict[str, Any],
    timeout_sec: int = 30,
    retries: int = 3,
) -> dict[str, Any]:
    mode = str(body.get("_agmarknet_mode") or "").strip().lower()
    if mode == "report" or ("from_date" in body and "dashboard" not in body):
        return _fetch_report_page(body, timeout_sec=timeout_sec, retries=retries)
    return _fetch_dashboard_page(body, timeout_sec=timeout_sec, retries=retries)


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
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []
