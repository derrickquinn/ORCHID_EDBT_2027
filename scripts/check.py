"""Run artifact configuration checks and the harness regression suite."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for command in (
        [
            "validate_configs.py",
            "config/perf",
            "experiments/grid",
            "experiments/orchid-p",
            "experiments/spool",
        ],
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
        ["-m", "unittest", "discover", "-s", "utils_cpp/tests", "-v"],
    ):
        subprocess.run([sys.executable, *command], cwd=ROOT / "orchid", check=True)


if __name__ == "__main__":
    main()
