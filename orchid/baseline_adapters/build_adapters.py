#!/usr/bin/env python3
"""Fetch, verify, build, and install the pinned ACORN/NaviX adapters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Dependency:
    url: str
    revision: str


DEPENDENCIES = {
    "acorn": Dependency(
        "https://github.com/nicolelii/ACORN.git",
        "3996dcf90ca2fb58abe46679a13d6a57792274c2",
    ),
    "navix": Dependency(
        "https://github.com/gaurav8297/faiss-navix.git",
        "25d563a2dec9087891f19055b8407d4383a26342",
    ),
}


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def _require_tools() -> None:
    missing = [tool for tool in ("cmake", "git") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing required tools: {', '.join(missing)}")


def _ensure_source(name: str, source: Path, *, fetch: bool) -> None:
    dependency = DEPENDENCIES[name]
    if not (source / ".git").exists():
        if not fetch:
            raise RuntimeError(f"{name} source is absent: {source}")
        source.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", dependency.url, str(source)])
        _run(
            ["git", "fetch", "origin", dependency.revision, "--depth=1"],
            cwd=source,
        )
        _run(["git", "checkout", "--detach", dependency.revision], cwd=source)

    actual = _output(["git", "-C", str(source), "rev-parse", "HEAD"])
    if actual != dependency.revision:
        raise RuntimeError(
            f"{name} source is at {actual}; expected {dependency.revision}"
        )
    dirty = _output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]
    )
    if dirty:
        raise RuntimeError(f"{name} source has tracked modifications")


def _build_one(
    name: str,
    source: Path,
    build_root: Path,
    install_dir: Path,
    jobs: int,
) -> None:
    faiss_build = build_root / name / "faiss"
    adapter_build = build_root / name / "adapter"
    _run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(faiss_build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DBUILD_TESTING=OFF",
            "-DFAISS_ENABLE_C_API=OFF",
            "-DFAISS_ENABLE_GPU=OFF",
            "-DFAISS_ENABLE_PYTHON=OFF",
        ]
    )
    _run(
        [
            "cmake",
            "--build",
            str(faiss_build),
            "--target",
            "faiss",
            "--parallel",
            str(jobs),
        ]
    )
    library = faiss_build / "faiss" / "libfaiss.a"
    if not library.exists():
        raise RuntimeError(f"Faiss build did not produce {library}")

    _run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(adapter_build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DBASELINE={name}",
            f"-DBASELINE_FAISS_SOURCE_DIR={source}",
            f"-DBASELINE_FAISS_LIBRARY={library}",
        ]
    )
    _run(
        [
            "cmake",
            "--build",
            str(adapter_build),
            "--parallel",
            str(jobs),
        ]
    )
    _run(["cmake", "--install", str(adapter_build)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("acorn", "navix", "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=ROOT / "_deps")
    parser.add_argument("--acorn-source", type=Path)
    parser.add_argument("--navix-source", type=Path)
    parser.add_argument("--build-root", type=Path, default=ROOT / "_build")
    parser.add_argument("--install-dir", type=Path, default=ROOT / "_install")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--no-fetch", action="store_true", help="require existing source checkouts"
    )
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help="verify tools and pinned revisions without compiling",
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    _require_tools()
    names = tuple(DEPENDENCIES) if args.baseline == "all" else (args.baseline,)
    sources = {
        "acorn": args.acorn_source or args.source_root / "acorn",
        "navix": args.navix_source or args.source_root / "navix",
    }
    for name in names:
        source = sources[name].resolve()
        _ensure_source(name, source, fetch=not args.no_fetch)
        print(f"{name}: verified {DEPENDENCIES[name].revision}")
        if not args.check_sources:
            _build_one(
                name,
                source,
                args.build_root.resolve(),
                args.install_dir.resolve(),
                args.jobs,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
