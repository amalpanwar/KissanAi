from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config


def main() -> None:
    cfg = load_config()
    conn = sqlite3.connect(cfg.paths["sqlite_db"])
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT district, season, crop_name, recommendation_text,
               budget_min_inr, budget_max_inr,
               expected_yield_qtl_per_acre, expected_revenue_inr_per_acre
        FROM advisories
        ORDER BY id
        """
    ).fetchall()
    conn.close()

    out_path = cfg.paths["finetune_data"]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            prompt = (
                f"जिला: {r['district']} | मौसम: {r['season']} | फसल: {r['crop_name']}\n"
                "बजट और उत्पादन के आधार पर सलाह दें।"
            )
            response = (
                f"सलाह: {r['recommendation_text']}\n"
                f"बजट: ₹{r['budget_min_inr']} - ₹{r['budget_max_inr']} प्रति एकड़\n"
                f"अपेक्षित उपज: {r['expected_yield_qtl_per_acre']} क्विंटल/एकड़\n"
                f"अपेक्षित आय: ₹{r['expected_revenue_inr_per_acre']} प्रति एकड़"
            )
            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")

    print(f"Wrote fine-tuning dataset: {out_path}")


if __name__ == "__main__":
    main()
