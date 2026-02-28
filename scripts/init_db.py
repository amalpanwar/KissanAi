from app.config import load_config
from app.db import init_db


def main() -> None:
    cfg = load_config()
    init_db(cfg.paths["sqlite_db"])
    print(f"Initialized DB at {cfg.paths['sqlite_db']}")


if __name__ == "__main__":
    main()
