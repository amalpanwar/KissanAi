"""Export reviewed feedback. Does not train or deploy a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.feedback_learning import build_feedback_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite database; defaults to configured database")
    parser.add_argument("--output-dir", default="data/processed/feedback_learning")
    args = parser.parse_args()
    splits = build_feedback_dataset(args.db or load_config().paths["sqlite_db"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} examples")
    if not splits["train"] or not splits["eval"]:
        print("Collect more reviewed examples before training: both train and eval must be non-empty.")


if __name__ == "__main__":
    main()
