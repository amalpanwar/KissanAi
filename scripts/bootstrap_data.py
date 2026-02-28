from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys

from app.config import load_config


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    run([sys.executable, "scripts/init_db.py"])
    run([sys.executable, "scripts/load_seed_data.py"])

    conn = sqlite3.connect(cfg.paths["sqlite_db"])
    try:
        count = conn.execute("SELECT COUNT(*) FROM research_documents").fetchone()[0]
    finally:
        conn.close()

    should_ingest = args.force or count == 0
    if should_ingest:
        run([sys.executable, "scripts/ingest_documents.py", "--input_dir", args.input_dir])

    idx_exists = os.path.exists(cfg.paths["vector_store"])
    meta_exists = os.path.exists(cfg.paths["metadata_store"])
    should_build_index = args.force or (not idx_exists) or (not meta_exists) or should_ingest
    if should_build_index:
        run([sys.executable, "scripts/build_index.py"])

    print("Bootstrap complete: idempotent initialization finished.")


if __name__ == "__main__":
    main()
