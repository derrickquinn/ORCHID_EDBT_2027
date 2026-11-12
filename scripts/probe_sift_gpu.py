"""Check the GPU components used by a small ORCHID SIFT run."""

import argparse
import asyncio
import sys
from pathlib import Path


def cuda_architectures() -> str:
    """Resolve visible GPUs to explicit CMake targets, including PTX."""
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError(
            "no CUDA GPU is visible; use a GPU host or explicit --cuda-architectures"
        )
    capabilities = set()
    for device in range(count):
        major, minor = torch.cuda.get_device_capability(device)
        capabilities.add(major * 10 + minor)
        print(
            f"CUDA device {device}: {torch.cuda.get_device_name(device)} "
            f"(sm_{major}{minor})",
            file=sys.stderr, flush=True,
        )
    # Unsuffixed CMake targets generate both machine code and PTX.
    return ";".join(str(capability) for capability in sorted(capabilities))


async def probe_gpu() -> None:
    import faiss
    import numpy as np

    required = (
        "get_num_gpus", "StandardGpuResources", "GpuClonerOptions",
        "index_cpu_to_gpu", "INDICES_64_BIT",
    )
    if any(not hasattr(faiss, name) for name in required):
        raise RuntimeError("installed FAISS is CPU-only; rebuild with CUDA support")
    if faiss.get_num_gpus() < 1:
        raise RuntimeError("no CUDA GPU is visible on this host")

    # Exercise the same cloning options as the benchmark before clustering.
    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.use_cuvs = False
    print("GPU probe: FAISS cloning", flush=True)
    try:
        index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(1), options)
    except RuntimeError as exc:
        raise RuntimeError(
            f"FAISS GPU cloning failed: {exc}. Rebuild FAISS with menu 13."
        ) from exc
    if index is None:
        raise RuntimeError("FAISS could not create a GPU index")

    from predicate_index import PackedBitmap, PostingList, materialize_results_gpu

    results = [
        PostingList(np.array([0, 8, 18], dtype=np.uint32)),
        PackedBitmap(np.array([2, 128, 0], dtype=np.uint8)),
    ]
    print("GPU probe: FAISS passed; checking utils_cpp mask materialization", flush=True)
    try:
        mi, start_stamp, end_stamp = materialize_results_gpu(results, 19)
        actual = np.asarray(await mi.collect_to_cpu()).reshape(2, 3)
        # Timestamp threads must finish before asyncio.run closes the event loop.
        await start_stamp
        await end_stamp
    except RuntimeError as exc:
        raise RuntimeError(
            f"utils_cpp GPU mask materialization failed: {exc}. "
            "Rebuild utils_cpp for this GPU with menu 12."
        ) from exc
    np.testing.assert_array_equal(
        actual, np.array([[1, 1, 4], [2, 128, 0]], dtype=np.uint8)
    )


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchid"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-architectures", action="store_true",
        help="Print explicit CMake targets for visible GPUs instead of running the probe",
    )
    args = parser.parse_args()
    try:
        if args.cuda_architectures:
            print(cuda_architectures())
            return 0
        asyncio.run(probe_gpu())
    except (
        ImportError, AttributeError, RuntimeError, ValueError, TypeError,
        AssertionError, OSError,
    ) as exc:
        print(
            f"SIFT GPU unavailable: {exc}",
            file=sys.stderr if args.cuda_architectures else sys.stdout,
            flush=True,
        )
        return 1
    print("SIFT GPU probe passed: FAISS cloning and mask materialization", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
