"""Small contract test for one compiled baseline adapter.

Run this separately for each module by putting its CMake build directory on
PYTHONPATH and passing either ``acorn_adapter`` or ``navix_adapter``.
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", choices=("acorn_adapter", "navix_adapter"))
    args = parser.parse_args()

    adapter = importlib.import_module(args.module)
    rng = np.random.default_rng(7)
    vectors = np.ascontiguousarray(rng.normal(size=(96, 12)), dtype=np.float32)
    queries = vectors[:8].copy()
    metadata = np.arange(len(vectors), dtype=np.int32) % 4
    masks = np.zeros((len(queries), len(vectors)), dtype=np.uint8)
    for query_id in range(len(queries)):
        masks[query_id] = metadata == query_id % 4

    if args.module == "acorn_adapter":
        index = adapter.Index.build(
            vectors, metadata, m=16, gamma=4, m_beta=32, ef_construction=80
        )
    else:
        index = adapter.Index.build(vectors, m=16, ef_construction=80)

    distances, labels = index.search(queries, masks, k=5, ef_search=96)
    assert distances.shape == labels.shape == (len(queries), 5)
    assert labels.dtype == np.int64
    for query_id, row in enumerate(labels):
        valid = row[row >= 0]
        assert len(valid) > 0
        assert np.all(masks[query_id, valid] != 0)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "index.faiss"
        index.save(str(path))
        restored = adapter.Index.load(str(path))
        _, restored_labels = restored.search(queries, masks, k=5, ef_search=96)
        np.testing.assert_array_equal(restored_labels, labels)

    try:
        index.search(queries, masks.astype(bool), k=5, ef_search=96)
    except TypeError:
        pass
    else:
        raise AssertionError("the adapter silently converted a non-uint8 mask")

    print(f"{args.module}: smoke test passed ({adapter.baseline_revision})")


if __name__ == "__main__":
    main()
