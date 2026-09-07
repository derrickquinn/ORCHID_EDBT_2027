"""Build and install the native cuVS filtered-CAGRA adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cagra_native"
DEFAULT_BUILD = SOURCE / "_build"
DEFAULT_INSTALL = ROOT / "baseline_adapters/_install"
ARTIFACT_REPOSITORY = "https://github.com/nicolelii/CAGRA_SIGMOD_Artifacts"
ARTIFACT_REVISION = "79c153ffc4ebf4363f522097958f83d7eec19ad4"


def _run(*command: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _verify_module(install_dir: Path) -> None:
    code = f"""
import sys
sys.path.insert(0, {str(install_dir)!r})
import _cagra_native
assert _cagra_native.artifact_repository == {ARTIFACT_REPOSITORY!r}
assert _cagra_native.artifact_revision == {ARTIFACT_REVISION!r}
required = ("Index", "Metric")
missing = [name for name in required if not hasattr(_cagra_native, name)]
if missing:
    raise RuntimeError(f"native CAGRA module is missing: {{missing}}")
print(f"validated native CAGRA module: {{_cagra_native.__file__}}")
"""
    _run(sys.executable, "-c", code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL)
    parser.add_argument(
        "--cuda-architectures",
        default=os.environ.get("CAGRA_CUDA_ARCHITECTURES", "native"),
        help="CMake CUDA architecture list (default: native)",
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="report the pinned upstream artifact without compiling",
    )
    args = parser.parse_args()

    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if not (SOURCE / "CMakeLists.txt").is_file():
        raise RuntimeError(f"native CAGRA adapter source is absent: {SOURCE}")
    if args.check_source:
        print(f"CAGRA artifact: {ARTIFACT_REPOSITORY}@{ARTIFACT_REVISION}")
        return

    build_dir = args.build_dir.resolve()
    install_dir = args.install_dir.resolve()
    _run(
        "cmake",
        "-S",
        str(SOURCE),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}",
    )
    _run(
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        str(args.jobs),
    )
    _run("cmake", "--install", str(build_dir))
    _verify_module(install_dir)


if __name__ == "__main__":
    main()
