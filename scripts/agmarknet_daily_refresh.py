from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    python = sys.executable
    script = ROOT / "scripts" / "agmarknet_fetch.py"

    keep_years = os.getenv("AGMARKNET_KEEP_YEARS", "2")
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
    limit = os.getenv("AGMARKNET_LIMIT", "200")
    max_pages = os.getenv("AGMARKNET_MAX_PAGES", "2000")
    timeout_sec = os.getenv("AGMARKNET_TIMEOUT_SEC", "60")
    retries = os.getenv("AGMARKNET_RETRIES", "5")
    debug = os.getenv("AGMARKNET_DEBUG", "0")

    cmd = [
        python,
        str(script),
        "--keep_years",
        keep_years,
        "--state_ids",
        state_ids,
        "--district_ids",
        district_ids,
        "--districts_as_list",
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
    ]

    if debug == "1":
        cmd.append("--debug")

    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
