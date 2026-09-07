#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from src.test_ivf import load_test_ivf_config
from src.optimize_ivf import load_optimize_ivf_config
from src.build_ivf_config import load_build_ivf_config
from src.construct_ivf import load_construct_ivf_config
from src.rank_selectivity import load_rank_selectivity_config
from src.baseline_perf_config import KNOWN_DATASETS, load_baseline_perf_config


def _gather_yaml_files(paths: list[str]) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    collected: set[Path] = set()

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            errors.append(f"{raw}: path does not exist")
            continue
        if path.is_dir():
            matches = [p for p in path.rglob("*") if p.suffix in {".yaml", ".yml"}]
            if not matches:
                errors.append(f"{raw}: no YAML files found")
                continue
            collected.update(matches)
            continue
        if path.suffix not in {".yaml", ".yml"}:
            errors.append(f"{raw}: not a YAML file")
            continue
        collected.add(path)

    return sorted(collected), errors


def _validate_file(path: Path) -> list[str]:
    try:
        payload = yaml.safe_load(path.read_text())
    except Exception as exc:
        return [f"{path}: failed to parse YAML ({exc})"]

    if not isinstance(payload, dict):
        return [f"{path}: config must be a mapping with a 'kind' key"]

    kind = payload.get("kind")
    if not kind:
        return [f"{path}: missing required 'kind' field"]

    if kind == "test_ivf":
        try:
            load_test_ivf_config(path)
        except Exception as exc:
            return [f"{path}: test_ivf validation failed ({exc})"]
        return []
    if kind == "optimize_ivf":
        try:
            load_optimize_ivf_config(path)
        except Exception as exc:
            return [f"{path}: optimize_ivf validation failed ({exc})"]
        return []
    if kind in {"build_ivf", "perf"}:
        try:
            if kind == "perf":
                raw_args = payload.get("args")
                if not isinstance(raw_args, dict):
                    raise TypeError("Perf config must include an 'args' mapping")
                per_dataset = raw_args.get("per_dataset")
                if per_dataset is not None and not isinstance(per_dataset, dict):
                    raise TypeError("Perf config 'per_dataset' must be a mapping")
                dataset_names = list(per_dataset or {None: None})
                for dataset_name in dataset_names:
                    load_build_ivf_config(path, dataset_name=dataset_name)
            else:
                load_build_ivf_config(path)
        except Exception as exc:
            return [f"{path}: {kind} validation failed ({exc})"]
        return []
    if kind == "construct_ivf":
        try:
            load_construct_ivf_config(path)
        except Exception as exc:
            return [f"{path}: construct_ivf validation failed ({exc})"]
        return []
    if kind == "rank_sel":
        try:
            load_rank_selectivity_config(path)
        except Exception as exc:
            return [f"{path}: rank_sel validation failed ({exc})"]
        return []
    if kind == "baseline_perf":
        try:
            raw_args = payload.get("args")
            if not isinstance(raw_args, dict):
                raise TypeError("Baseline perf config must include an 'args' mapping")
            baseline = raw_args.get("baseline")
            for dataset_name in KNOWN_DATASETS:
                load_baseline_perf_config(
                    path, dataset_name=dataset_name, baseline=baseline
                )
        except Exception as exc:
            return [f"{path}: baseline_perf validation failed ({exc})"]
        return []

    return [f"{path}: unsupported kind {kind!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate experiment config YAML files."
    )
    parser.add_argument("paths", nargs="+", help="YAML files or directories to scan")
    args = parser.parse_args()

    files, errors = _gather_yaml_files(args.paths)
    if not files:
        errors.append("no YAML files to validate")

    for path in files:
        errors.extend(_validate_file(path))

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    print(f"Validated {len(files)} config file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
