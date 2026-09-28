"""Download and reproduce the fixed SIFT-12 paper workload (CPU only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import h5py
import numpy as np
import zarr
from zarr.codecs import ZstdCodec

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
SOURCE_SHA256 = "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984"
REFERENCE = ROOT / "scripts/data/sift12-reference.npz"
REFERENCE_SHA256 = "7f07540e114a0839147bc106282e13b5eca4bd9f0250eeb6ee06abd3cbac6fa2"
RECIPE = {
    "version": 1,
    "source_sha256": SOURCE_SHA256,
    "reference_sha256": REFERENCE_SHA256,
}
VECTOR_STORE = "sift1m.zarr"
PREDICATE_STORE = "sift1m_c12.zarr"
MANIFEST = "sift12-preparation.json"
SHAPES = {"train": (1_000_000, 128), "test": (10_000, 128)}
# Preserve the paper stores' layout, including column-split vector chunks.
CHUNKS = {
    "train": (62500, 8),
    "test": (2500, 32),
    "pred_mask": (157, 31250),
    "pred_mt": (15625, 313),
    "neighbors": (4096, 10),
    "neighbors_unfiltered": (4096, 10),
}


def require_checksum(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def download(destination: Path) -> None:
    """Reuse only a verified download; never publish a partial or bad file."""
    if destination.exists():
        require_checksum(destination, SOURCE_SHA256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".sift-download-", dir=destination.parent
    ) as tmp:
        partial = Path(tmp) / "source.hdf5"
        print(f"Downloading {SOURCE_URL}", flush=True)
        request = Request(SOURCE_URL, headers={"User-Agent": "ORCHID-SIFT12/1.0"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = 0
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                total += len(block)
                print(f"  downloaded {total / 1024**2:.0f} MiB", flush=True)
        require_checksum(partial, SOURCE_SHA256)
        partial.rename(destination)


def predicate_values(n_docs: int, n_queries: int) -> tuple[np.ndarray, np.ndarray]:
    # Keep int64 draws: drawing directly as uint8 changes the RNG stream.
    rng = np.random.default_rng(42)
    return rng.integers(0, 12, size=n_docs), rng.integers(0, 12, size=n_queries)


def create_array(
    group: zarr.Group, key: str, shape: tuple[int, int], dtype: str
) -> zarr.Array:
    return group.create_array(
        key,
        shape=shape,
        dtype=dtype,
        chunks=tuple(min(a, b) for a, b in zip(CHUNKS[key], shape)),
        compressors=[ZstdCodec(level=0, checksum=False)],
        fill_value=-1 if key.startswith("neighbors") else 0,
    )


def write_stores(
    source: Path, destination: Path, reference: dict[str, np.ndarray]
) -> None:
    """Stream vectors and both masks; the largest mask block is about 157 MB."""
    vg = zarr.open_group(str(destination / VECTOR_STORE), mode="w", zarr_format=3)
    pg = zarr.open_group(str(destination / PREDICATE_STORE), mode="w", zarr_format=3)
    with h5py.File(source, "r") as raw:
        for key in ("train", "test"):
            dataset = raw[key]
            output = create_array(vg, key, dataset.shape, "float32")
            for start in range(0, len(dataset), output.chunks[0]):
                end = min(start + output.chunks[0], len(dataset))
                # Original rotation: float64 multiplication, then float32 storage.
                output[start:end] = (
                    dataset[start:end].astype(np.float64) @ reference["rotation"]
                ).astype(np.float32)
            print(f"Prepared {key}: {dataset.shape}", flush=True)

    n_docs, n_queries = vg["train"].shape[0], vg["test"].shape[0]
    docs, queries = predicate_values(n_docs, n_queries)
    for key, shape in (
        ("pred_mask", (n_queries, n_docs)),
        ("pred_mt", (n_docs, n_queries)),
    ):
        output = create_array(pg, key, shape, "bool")
        for start in range(0, shape[0], output.chunks[0]):
            end = min(start + output.chunks[0], shape[0])
            output[start:end] = (
                queries[start:end, None] == docs[None, :]
                if key == "pred_mask"
                else docs[start:end, None] == queries[None, :]
            )
            if start // output.chunks[0] % 16 == 0:
                print(f"Prepared {key}: {end}/{shape[0]} rows", flush=True)

    # Preserve canonical IDs and ordering, including ties in the original runs.
    for key, group in (("neighbors", pg), ("neighbors_unfiltered", vg)):
        data = reference[key]
        output = create_array(group, key, data.shape, "int64")
        output[:] = data
        if key == "neighbors_unfiltered":
            output.attrs.update(
                {
                    "complete": True,
                    "unfiltered": True,
                    "metric": "l2",
                    "k": 10,
                    "document_count": n_docs,
                    "query_count": n_queries,
                    "train_key": "train",
                    "test_key": "test",
                    "dtype": "float32",
                    "generator": "canonical SIFT-12 reference",
                }
            )


def validate_source(source: Path) -> None:
    require_checksum(source, SOURCE_SHA256)
    with h5py.File(source, "r") as raw:
        for key, shape in SHAPES.items():
            if (
                key not in raw
                or raw[key].shape != shape
                or raw[key].dtype != np.dtype("float32")
            ):
                raise ValueError(
                    f"Source must contain float32 {key} with shape {shape}"
                )


def load_reference() -> dict[str, np.ndarray]:
    require_checksum(REFERENCE, REFERENCE_SHA256)
    with np.load(REFERENCE, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_layout(directory: Path) -> None:
    """Check completed output structure before reuse (not a content audit)."""
    n_docs, dim = SHAPES["train"]
    n_queries = SHAPES["test"][0]
    for store, expected in (
        (
            VECTOR_STORE,
            {
                "train": ((n_docs, dim), "float32"),
                "test": ((n_queries, dim), "float32"),
                "neighbors_unfiltered": ((n_queries, 10), "int64"),
            },
        ),
        (
            PREDICATE_STORE,
            {
                "pred_mask": ((n_queries, n_docs), "bool"),
                "pred_mt": ((n_docs, n_queries), "bool"),
                "neighbors": ((n_queries, 10), "int64"),
            },
        ),
    ):
        group = zarr.open_group(str(directory / store), mode="r")
        for key, (shape, dtype) in expected.items():
            if (
                key not in group
                or group[key].shape != shape
                or group[key].dtype != np.dtype(dtype)
            ):
                raise ValueError(f"Incomplete or incompatible output: {store}/{key}")


def prepare(output_dir: Path, source: Path | None = None) -> None:
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    assert zarr.config.get("codec_pipeline.path") == "zarrs.ZarrsCodecPipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".sift12-preparation.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        raise ValueError(
            f"Preparation lock exists: {lock}. After a killed run, remove it only after confirming no preparation is running."
        ) from None
    try:
        with handle:
            manifest = output_dir / MANIFEST
            if manifest.exists():
                if json.loads(manifest.read_text()) != RECIPE:
                    raise ValueError(
                        f"Different recipe at {manifest}; use a fresh output directory"
                    )
                validate_layout(output_dir)
                print(f"SIFT-12 is already prepared in {output_dir}", flush=True)
                return
            for name in (VECTOR_STORE, PREDICATE_STORE):
                if (output_dir / name).exists() or (output_dir / name).is_symlink():
                    raise ValueError(
                        f"Refusing to replace {output_dir / name}; use a fresh output directory"
                    )
            reference = load_reference()
            if source is None:
                source = output_dir / "raw/sift-128-euclidean.hdf5"
                download(source)
            validate_source(source)
            with tempfile.TemporaryDirectory(
                prefix=".sift12-preparation-", dir=output_dir
            ) as tmp:
                staging = Path(tmp)
                write_stores(source, staging, reference)
                validate_layout(staging)
                (staging / MANIFEST).write_text(json.dumps(RECIPE, indent=2) + "\n")
                for name in (VECTOR_STORE, PREDICATE_STORE, MANIFEST):
                    (staging / name).rename(output_dir / name)
            print(f"SIFT-12 ready in {output_dir}", flush=True)
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="Store directory (default: repository data/)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Reuse a local ANN-Benchmarks HDF5 file (still checksum verified)",
    )
    args = parser.parse_args()
    try:
        prepare(
            args.output_dir.resolve(), args.source.resolve() if args.source else None
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"SIFT-12 preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
