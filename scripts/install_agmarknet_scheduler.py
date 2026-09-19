"""Generate a launchd plist using this checkout and the active Python interpreter."""
import importlib.util
import plistlib
import sys
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("pandas", "yaml")):
    raise SystemExit("Run this script using a Python environment with pandas and PyYAML installed.")

root = Path(__file__).resolve().parents[1]
output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "scripts/com.kisaanai.agmarknet.plist"
payload = {"Label": "com.kisaanai.agmarknet", "ProgramArguments": [sys.executable, str(root / "scripts/agmarknet_daily_refresh.py")],
           "WorkingDirectory": str(root), "StartCalendarInterval": {"Hour": 6, "Minute": 30},
           "StandardOutPath": str(root / "logs/agmarknet_refresh.out"),
           "StandardErrorPath": str(root / "logs/agmarknet_refresh.err"), "RunAtLoad": True}
(root / "logs").mkdir(exist_ok=True)
output.write_bytes(plistlib.dumps(payload))
print(f"Generated {output}; install with launchctl bootstrap gui/$(id -u) {output}")
