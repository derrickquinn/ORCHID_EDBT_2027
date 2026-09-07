"""Bounded native filtered-search and serialization check for CAGRA."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np


SIGMOD_ROOT = Path(__file__).resolve().parents[1]
INSTALL_DIR = SIGMOD_ROOT / "baseline_adapters/_install"
sys.path.insert(0, str(INSTALL_DIR))
sys.path.insert(0, str(SIGMOD_ROOT))

import cagra_adapter  # noqa: E402


def _assert_filtered(index, queries: np.ndarray, masks: np.ndarray) -> None:
    _, labels = index.search(queries, masks, k=5, ef_search=128)
    allowed = np.unpackbits(
        masks, axis=1, count=index.ntotal, bitorder="little"
    ).astype(bool)
    for query, row in enumerate(labels):
        valid = row[row < index.ntotal]
        if valid.size == 0 or not np.all(allowed[query, valid]):
            raise AssertionError("CAGRA returned a vector rejected by its query bitmap")


def main() -> None:
    rng = np.random.default_rng(1234)
    n_docs, dimension, n_queries = 2051, 32, 8
    vectors = rng.standard_normal((n_docs, dimension), dtype=np.float32)
    query_ids = np.arange(n_queries) * 17
    queries = np.ascontiguousarray(vectors[query_ids])
    ids = np.arange(n_docs)
    allowed = np.stack([(ids % n_queries) == query for query in range(n_queries)])
    masks = np.packbits(allowed, axis=1, bitorder="little")

    index = cagra_adapter.Index.build(
        vectors,
        graph_degree=32,
        intermediate_graph_degree=64,
        metric=cagra_adapter.Metric.L2,
    )
    _assert_filtered(index, queries, masks)

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "smoke.cagra")
        index.save(path)
        restored = cagra_adapter.Index.load(path)
        _assert_filtered(restored, queries, masks)

    print(
        "native CAGRA smoke test passed "
        f"({cagra_adapter.artifact_revision})"
    )


if __name__ == "__main__":
    main()
