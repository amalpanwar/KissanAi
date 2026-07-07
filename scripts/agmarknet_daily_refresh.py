from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "raw" / "live" / "agmarknet_report.csv"
CATALOG_CSV = ROOT / "data" / "processed" / "agmarknet_catalog.csv"
STATUS_JSON = ROOT / "data" / "raw" / "live" / "agmarknet_refresh_status.json"


def _latest_report_date(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path, usecols=["rep_date"])
    except Exception:
        return ""
    if "rep_date" not in df.columns or df.empty:
        return ""
    dt = pd.to_datetime(df["rep_date"], errors="coerce", dayfirst=True)
    if dt.dropna().empty:
        return ""
    return dt.max().date().isoformat()


def _write_catalog(path: Path) -> None:
    if not path.exists():
        return
    try:
        df = pd.read_csv(path, usecols=["state_name", "district_name", "cmdt_name"])
        df = df.rename(columns={"state_name": "State", "district_name": "District", "cmdt_name": "Commodity"})
    except Exception:
        try:
            df = pd.read_csv(path, usecols=["State", "District", "Commodity"])
        except Exception:
            return
    df = df.fillna("").astype(str).drop_duplicates().sort_values(["State", "District", "Commodity"])
    CATALOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG_CSV, index=False)


def main() -> int:
    python = sys.executable
    script = ROOT / "scripts" / "agmarknet_fetch.py"

    keep_years = os.getenv("AGMARKNET_KEEP_YEARS", "2")
    lookback_days = int(os.getenv("AGMARKNET_LOOKBACK_DAYS", "14"))
    state_ids = os.getenv("AGMARKNET_STATE_IDS", "34")
    district_ids = os.getenv(
        "AGMARKNET_DISTRICT_IDS",
        "586,595,604,614,615,640,642,649,653,638",
    )
    group_ids_file = os.getenv(
        "AGMARKNET_GROUP_IDS_FILE",
        str(ROOT / "data" / "raw" / "agmarknet_group_ids.txt"),
    )
    group_map = os.getenv(
        "AGMARKNET_GROUP_MAP",
        str(ROOT / "data" / "raw" / "agmarknet_group_commodities.json"),
    )
    options = os.getenv("AGMARKNET_OPTIONS", "2")
    max_pages = os.getenv("AGMARKNET_MAX_PAGES", "2000")
    timeout_sec = os.getenv("AGMARKNET_TIMEOUT_SEC", "90")
    retries = os.getenv("AGMARKNET_RETRIES", "6")
    debug = os.getenv("AGMARKNET_DEBUG", "0")
    limit = os.getenv("AGMARKNET_LIMIT", "100")
    districts_as_list = os.getenv("AGMARKNET_DISTRICTS_AS_LIST", "0") == "1"
    mode = os.getenv("AGMARKNET_MODE", "report").strip().lower() or "report"
    before_latest = _latest_report_date(OUT_CSV)

    def write_status(status: str, message: str, extra: dict | None = None) -> None:
        payload = {
            "status": status,
            "message": message,
            "before_latest": before_latest,
            "after_latest": _latest_report_date(OUT_CSV),
            "mode": mode,
            "lookback_days_requested": lookback_days,
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
        STATUS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_status() -> dict:
        if not STATUS_JSON.exists():
            return {}
        try:
            return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def build_cmd(lb_days: int, limit_value: str) -> list[str]:
        cmd = [
            python,
            str(script),
            "--mode",
            mode,
            "--lookback_days",
            str(lb_days),
            "--state_ids",
            state_ids,
            "--district_ids",
            district_ids,
            "--group_ids_file",
            group_ids_file,
            "--group_commodities_json",
            group_map,
            "--options",
            options,
            "--limit",
            limit_value,
            "--max_pages",
            max_pages,
            "--timeout_sec",
            timeout_sec,
            "--retries",
            retries,
            "--merge_existing",
            "--trim_years",
            keep_years,
            "--summary_json",
            str(STATUS_JSON),
            "--require_complete",
        ]
        if mode == "dashboard":
            cmd.append("--all_districts")
        if districts_as_list:
            cmd.append("--districts_as_list")
        if debug == "1":
            cmd.append("--debug")
        return cmd

    lookbacks = []
    for lb in [lookback_days, 30, 60, 90]:
        if lb not in lookbacks:
            lookbacks.append(lb)
    limit_candidates = []
    for lv in [limit, "100", "50"]:
        if lv not in limit_candidates:
            limit_candidates.append(lv)

    last_rc = 0
    for lb in lookbacks:
        for limit_value in limit_candidates:
            print(f"Trying Agmarknet refresh with lookback_days={lb}, limit={limit_value}")
            rc = subprocess.call(build_cmd(lb, limit_value), cwd=str(ROOT))
            after_latest = _latest_report_date(OUT_CSV)
            last_rc = rc
            status_payload = read_status()
            run_status = str(status_payload.get("status") or "").strip().lower()
            if rc != 0:
                if run_status == "success" and after_latest:
                    msg = (
                        f"Agmarknet refresh completed successfully with lookback_days={lb}, "
                        f"limit={limit_value}; latest report date remains {after_latest}."
                    )
                    print(msg)
                    write_status(
                        "success",
                        msg,
                        {
                            "after_latest": after_latest,
                            "lookback_days_used": lb,
                            "limit_used": int(limit_value),
                        },
                    )
                    return 0
                continue
            if not after_latest:
                continue
            if (not before_latest) or after_latest > before_latest:
                _write_catalog(OUT_CSV)
                msg = (
                    f"Agmarknet refresh advanced report date from {before_latest or 'N/A'} "
                    f"to {after_latest} using lookback_days={lb}, limit={limit_value}"
                )
                print(msg)
                write_status(
                    "success",
                    msg,
                    {
                        "after_latest": after_latest,
                        "lookback_days_used": lb,
                        "limit_used": int(limit_value),
                    },
                )
                return 0
            if run_status == "success":
                _write_catalog(OUT_CSV)
                msg = (
                    f"Agmarknet refresh completed successfully with lookback_days={lb}, "
                    f"limit={limit_value}; latest report date remains {after_latest}."
                )
                print(msg)
                write_status(
                    "success",
                    msg,
                    {
                        "after_latest": after_latest,
                        "lookback_days_used": lb,
                        "limit_used": int(limit_value),
                    },
                )
                return 0

    after_latest = _latest_report_date(OUT_CSV)
    if last_rc != 0:
        write_status("failed", "Agmarknet refresh command failed.", {"return_code": last_rc})
        return last_rc
    if not after_latest:
        print("Agmarknet refresh finished but no output date could be read.", file=sys.stderr)
        write_status("failed", "Agmarknet refresh finished but no output date could be read.")
        return 2
    _write_catalog(OUT_CSV)
    msg = f"Agmarknet refresh finished but report date did not advance (before={before_latest}, after={after_latest})."
    print(msg, file=sys.stderr)
    write_status("stale", msg, {"after_latest": after_latest})
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
