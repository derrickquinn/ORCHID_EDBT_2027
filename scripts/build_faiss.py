"""Build the bundled modified FAISS, then install its generated Python package."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--opt-level",
        choices=("generic", "avx2", "avx512", "avx512_spr"),
        default="avx2",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Requires a CUDA toolkit in the active environment",
    )
    parser.add_argument("--cuda-architectures", default="native")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    source = ROOT / "orchid/faiss"
    build = source / "build_python"
    prefix = Path(sys.prefix)
    # Use the active environment's MKL rather than a machine-specific oneAPI path.
    os.environ.setdefault("MKLROOT", str(prefix))
    run(
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DFAISS_ENABLE_EXTRAS=OFF",
        "-DFAISS_ENABLE_C_API=OFF",
        "-DFAISS_ENABLE_PYTHON=ON",
        "-DFAISS_ENABLE_MKL=ON",
        f"-DFAISS_ENABLE_GPU={'ON' if args.gpu else 'OFF'}",
        "-DFAISS_ENABLE_CUVS=OFF",
        f"-DFAISS_OPT_LEVEL={args.opt_level}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython_ROOT_DIR={prefix}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}",
    )
    targets = ["swigfaiss"]
    if args.opt_level != "generic":
        targets.append(f"swigfaiss_{args.opt_level}")
    run(
        "cmake",
        "--build",
        str(build),
        "--target",
        *targets,
        "--parallel",
        str(args.jobs),
    )
    package = build / "faiss/python"
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        ".",
        cwd=package,
    )
    run(
        sys.executable,
        "-c",
        "import faiss; assert hasattr(faiss, 'IDSelector2DBitmap'); print(faiss.__file__)",
    )


if __name__ == "__main__":
    main()
