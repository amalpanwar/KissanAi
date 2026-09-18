"""Refresh Agmarknet safely; usable from cron, Streamlit, and the data agent."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data/raw/live/agmarknet_report.csv"
CATALOG_CSV = ROOT / "data/processed/agmarknet_catalog.csv"
STATUS_JSON = ROOT / "data/raw/live/agmarknet_refresh_status.json"


def _latest_report_date(path: Path) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {"rep_date", "Arrival_Date", "reported_date"})
    for col in ("rep_date", "reported_date", "Arrival_Date"):
        if col in df:
            dates = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce").dropna()
            if not dates.empty:
                return dates.max().date().isoformat()
    return ""


def _write_catalog(path: Path) -> None:
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {"state_name", "district_name", "cmdt_name"})
    df = df.rename(columns={"state_name": "State", "district_name": "District", "cmdt_name": "Commodity"})
    df = df.fillna("").astype(str).drop_duplicates().sort_values(["State", "District", "Commodity"])
    CATALOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    temp = CATALOG_CSV.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    temp.replace(CATALOG_CSV)


def _refresh() -> int:
    before = _latest_report_date(OUT_CSV)
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    lookback = int(os.getenv("AGMARKNET_LOOKBACK_DAYS", "14"))
    requested = os.getenv("AGMARKNET_MODE", "auto").lower()
    modes = ["dashboard", "report"] if requested == "auto" else [requested]
    attempts = []
    summary_path = STATUS_JSON.with_name("agmarknet_fetch_summary.json")
    result_status, message, rc = "failed", "No complete refresh succeeded; previous data retained.", 4
    for mode in modes:
        if mode not in {"dashboard", "report"}:
            raise ValueError(f"Unknown AGMARKNET_MODE: {mode}")
        cmd = [sys.executable, str(ROOT / "scripts/agmarknet_fetch.py"),
               "--mode", mode, "--lookback_days", str(lookback),
               "--state_ids", os.getenv("AGMARKNET_STATE_IDS", "34"),
               "--district_ids", os.getenv("AGMARKNET_DISTRICT_IDS", "586,595,604,614,615,640,642,649,653,638"),
               "--group_ids_file", os.getenv("AGMARKNET_GROUP_IDS_FILE", str(ROOT / "data/raw/agmarknet_group_ids.txt")),
               "--group_commodities_json", os.getenv("AGMARKNET_GROUP_MAP", str(ROOT / "data/raw/agmarknet_group_commodities.json")),
               "--options", os.getenv("AGMARKNET_OPTIONS", "2"),
               "--limit", os.getenv("AGMARKNET_LIMIT", "100"),
               "--sleep_sec", os.getenv("AGMARKNET_SLEEP_SEC", "1.5"),
               "--max_pages", os.getenv("AGMARKNET_MAX_PAGES", "2000"),
               "--timeout_sec", os.getenv("AGMARKNET_TIMEOUT_SEC", "20"),
               "--retries", os.getenv("AGMARKNET_RETRIES", "2"),
               "--merge_existing", "--trim_years", os.getenv("AGMARKNET_KEEP_YEARS", "2"),
               "--out", str(OUT_CSV), "--summary_json", str(summary_path), "--require_complete"]
        if os.getenv("AGMARKNET_ALL_DISTRICTS", "0") == "1":
            cmd.append("--all_districts")
        print(f"Fetching Agmarknet mode={mode}, lookback={lookback}", flush=True)
        summary_path.unlink(missing_ok=True)
        try:
            rc = subprocess.call(cmd, cwd=ROOT, timeout=int(os.getenv("AGMARKNET_RUN_TIMEOUT_SEC", "300")))
        except subprocess.TimeoutExpired:
            rc = 124
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        attempts.append({"mode": mode, "return_code": rc, "summary": summary})
        if rc != 0 or summary.get("status") != "success":
            continue
        latest = _latest_report_date(OUT_CSV)
        age = (today - datetime.fromisoformat(latest).date()).days if latest else None
        max_age = int(os.getenv("AGMARKNET_MAX_AGE_DAYS", "3"))
        if age is None or age < 0 or age > max_age:
            result_status, message, rc = "stale", f"Fetched data, but latest report {latest or 'unknown'} is outside the freshness window.", 3
            continue
        _write_catalog(OUT_CSV)
        # Keep SQLite consumers consistent with the published CSV.
        db_rc = subprocess.call([sys.executable, str(ROOT / "scripts/load_agmarknet_prices.py")], cwd=ROOT)
        if db_rc:
            result_status, message, rc = "partial", "CSV updated, but SQLite synchronization failed.", 6
        else:
            result_status, message, rc = "success", f"Updated Agmarknet CSV, catalog and SQLite through {latest}.", 0
        break
    payload = {"status": result_status, "message": message, "before_latest": before,
               "after_latest": _latest_report_date(OUT_CSV), "mode": requested,
               "lookback_days_requested": lookback, "attempts": attempts,
               "updated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), "return_code": rc}
    temp = STATUS_JSON.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp.replace(STATUS_JSON)
    print(message, flush=True)
    return rc


def main() -> int:
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_JSON.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Agmarknet refresh is running.")
            return 75
        return _refresh()


if __name__ == "__main__":
    raise SystemExit(main())
