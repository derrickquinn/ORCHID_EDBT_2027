"""End-to-end correctness smoke test on a bounded slice of a real dataset."""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path

import numpy as np
import zarr

from perf_baseline import _normalize_rows
from predicate_index import materialize_byte_masks
from predicate_workload import (
    acorn_construction_metadata,
    load_or_build_predicate_workload,
)
from src.dataset_config import load_dataset_config


def _assert_legal(labels: np.ndarray, masks: np.ndarray) -> None:
    for query, row in enumerate(labels):
        valid = row[row >= 0]
        if valid.size == 0:
            raise AssertionError(f"search returned no valid IDs for query {query}")
        if not np.all(masks[query, valid] != 0):
            raise AssertionError(f"search returned a disallowed ID for query {query}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=("acorn", "navix"))
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    if zarr.config.get("codec_pipeline.path") != "zarrs.ZarrsCodecPipeline":
        raise RuntimeError("real-data validation requires the Zarrs codec pipeline")

    dataset = load_dataset_config(args.dataset_config)
    dataset.vec_zarr = str(args.data_root / Path(dataset.vec_zarr).name)
    vectors = zarr.open_group(dataset.vec_zarr, mode="r")
    total_documents = int(vectors["train"].shape[0])
    if not 0 < args.documents <= total_documents:
        raise ValueError("documents must fit within the real dataset")

    documents = np.ascontiguousarray(
        vectors["train"][: args.documents], dtype=np.float32
    )
    queries = np.ascontiguousarray(vectors["test"][: args.queries], dtype=np.float32)
    _normalize_rows(documents)
    _normalize_rows(queries)

    workload = load_or_build_predicate_workload(
        dataset,
        args.cache,
        np.arange(total_documents, dtype=np.int64),
        total_documents,
        args.queries,
        mask_format="byte",
    )
    full_buffer = np.empty((args.queries, total_documents), dtype=np.uint8)
    masks = np.ascontiguousarray(
        materialize_byte_masks(
            workload.evaluate_batch(0, args.queries, threads=args.threads),
            total_documents,
            full_buffer,
            threads=args.threads,
        )[:, : args.documents]
    )

    adapter = importlib.import_module(f"{args.baseline}_adapter")
    if args.baseline == "acorn":
        metadata = acorn_construction_metadata(
            dataset, total_documents, "workload_scalar"
        )[: args.documents]
        index = adapter.Index.build(
            documents,
            np.ascontiguousarray(metadata),
            m=32,
            gamma=12,
            m_beta=64,
            ef_construction=40,
            metric=adapter.Metric.L2,
        )
    else:
        index = adapter.Index.build(
            documents,
            m=32,
            ef_construction=200,
            metric=adapter.Metric.L2,
        )

    _, labels = index.search(queries, masks, k=10, ef_search=64)
    _assert_legal(labels, masks)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{args.baseline}.faiss"
        index.save(str(path))
        restored = adapter.Index.load(str(path))
        _, restored_labels = restored.search(queries, masks, k=10, ef_search=64)
        np.testing.assert_array_equal(restored_labels, labels)
        _assert_legal(restored_labels, masks)

    print(
        f"{args.baseline}: real-data smoke test passed "
        f"({args.documents} documents, {args.queries} queries)"
    )


if __name__ == "__main__":
    main()
