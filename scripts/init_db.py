from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import init_db


def main() -> None:
    cfg = load_config()
    init_db(cfg.paths["sqlite_db"])
    print(f"Initialized DB at {cfg.paths['sqlite_db']}")


if __name__ == "__main__":
    main()
