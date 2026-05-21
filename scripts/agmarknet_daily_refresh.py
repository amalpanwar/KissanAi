from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "raw" / "live" / "agmarknet_report.csv"


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
    limit = os.getenv("AGMARKNET_LIMIT", "250")
    districts_as_list = os.getenv("AGMARKNET_DISTRICTS_AS_LIST", "0") == "1"
    mode = os.getenv("AGMARKNET_MODE", "report").strip().lower() or "report"
    before_latest = _latest_report_date(OUT_CSV)

    def build_cmd(lb_days: int) -> list[str]:
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
            limit,
            "--max_pages",
            max_pages,
            "--timeout_sec",
            timeout_sec,
            "--retries",
            retries,
            "--merge_existing",
            "--trim_years",
            keep_years,
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

    last_rc = 0
    for lb in lookbacks:
        print(f"Trying Agmarknet refresh with lookback_days={lb}")
        rc = subprocess.call(build_cmd(lb), cwd=str(ROOT))
        after_latest = _latest_report_date(OUT_CSV)
        last_rc = rc
        if rc != 0:
            continue
        if not after_latest:
            continue
        if (not before_latest) or after_latest > before_latest:
            print(
                f"Agmarknet refresh advanced report date from {before_latest or 'N/A'} to {after_latest} using lookback_days={lb}"
            )
            return 0

    after_latest = _latest_report_date(OUT_CSV)
    if last_rc != 0:
        return last_rc
    if not after_latest:
        print("Agmarknet refresh finished but no output date could be read.", file=sys.stderr)
        return 2
    print(
        f"Agmarknet refresh finished but report date did not advance (before={before_latest}, after={after_latest}).",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
