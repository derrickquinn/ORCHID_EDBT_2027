"""Run a packaged paper configuration from any working directory."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("orchid", "orchid-p", "ivf", "pre-filter", "acorn", "navix", "cagra")
DATASETS = tuple(
    sorted(path.stem for path in (ROOT / "orchid/config/dataset").glob("*.yaml"))
)


def command_for(method: str, dataset: str, extra: list[str]) -> list[str]:
    config = "ivf" if method == "pre-filter" else method
    if method in ("acorn", "navix", "cagra"):
        command = [
            sys.executable,
            f"perf_{method}.py",
            "--config",
            f"config/perf/{config}.yaml",
            "--dset-cfg",
            f"config/dataset/{dataset}.yaml",
        ]
    else:
        command = [
            sys.executable,
            "perf.py",
            "--config",
            f"config/perf/{config}.yaml",
            "--dset_cfg",
            f"config/dataset/{dataset}.yaml",
        ]
        if method == "pre-filter":
            command += [
                "--val_flat",
                "--low_nprobe",
                "1",
                "--high_nprobe",
                "1",
                "--step_nprobe",
                "1",
                "--exp_name",
                "pre-filter",
            ]
    return command + extra


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Arguments after -- are passed to the selected harness; paths are relative to orchid/.",
    )
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without loading data"
    )
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    command = command_for(args.method, args.dataset, extra)
    env = os.environ.copy()
    for key, value in (
        ("OMP_NUM_THREADS", "32"),
        ("OMP_PROC_BIND", "true"),
        ("OMP_PLACES", "cores"),
    ):
        env.setdefault(key, value)
    print(f"Working directory: {ROOT / 'orchid'}", flush=True)
    import shlex

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT / "orchid", env=env)


if __name__ == "__main__":
    raise SystemExit(main())
