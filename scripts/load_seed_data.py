from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_conn


def main() -> None:
    cfg = load_config()
    db_path = cfg.paths["sqlite_db"]

    eco = pd.read_csv("data/raw/western_up_crop_economics.csv")
    adv = pd.read_csv("data/raw/sample_advisories.csv")

    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        eco_count = cur.execute("SELECT COUNT(*) FROM crop_economics").fetchone()[0]
        adv_count = cur.execute("SELECT COUNT(*) FROM advisories").fetchone()[0]

        if eco_count == 0:
            eco.to_sql("crop_economics", conn, if_exists="append", index=False)
        if adv_count == 0:
            adv.to_sql("advisories", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()

    print("Seed load completed (inserted only into empty tables).")


if __name__ == "__main__":
    main()
