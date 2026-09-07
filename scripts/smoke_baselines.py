"""Exercise native ACORN/NaviX filtering and index save/load on synthetic data."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    adapters = ROOT / "orchid/baseline_adapters"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(adapters / "_install")
    for module in ("acorn_adapter", "navix_adapter"):
        subprocess.run(
            [sys.executable, str(adapters / "smoke_test.py"), module],
            env=env,
            check=True,
        )
