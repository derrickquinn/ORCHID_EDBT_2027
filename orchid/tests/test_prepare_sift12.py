"""Preparation regression tests use only tiny, temporary datasets."""

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import zarr

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare_sift12.py"
spec = importlib.util.spec_from_file_location("prepare_sift12", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PrepareSift12Test(unittest.TestCase):
    def setUp(self):
        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.hdf5"
        self.train = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
        self.test = self.train[:5] + 0.5
        with h5py.File(self.source, "w") as raw:
            raw["train"] = self.train
            raw["test"] = self.test
        self.rotation = np.array(
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=np.float64
        )
        self.reference = {
            "rotation": self.rotation,
            "neighbors": np.tile(np.arange(10, dtype=np.int64), (5, 1)),
            "neighbors_unfiltered": np.tile(np.arange(10, 20, dtype=np.int64), (5, 1)),
        }
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(prep, "SHAPES", {"train": (32, 4), "test": (5, 4)})
        )
        self.stack.enter_context(
            patch.object(
                prep,
                "SOURCE_SHA256",
                hashlib.sha256(self.source.read_bytes()).hexdigest(),
            )
        )
        self.stack.enter_context(
            patch.object(prep, "CHUNKS", {key: (3, 2) for key in prep.CHUNKS})
        )
        self.stack.enter_context(
            patch.object(prep, "load_reference", return_value=self.reference)
        )
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_complete_preparation_preserves_vectors_masks_and_neighbors(self):
        destination = self.root / "output"
        prep.prepare(destination, self.source)
        v = zarr.open_group(str(destination / prep.VECTOR_STORE), mode="r")
        p = zarr.open_group(str(destination / prep.PREDICATE_STORE), mode="r")
        np.testing.assert_array_equal(v["train"][:], self.train @ self.rotation)
        np.testing.assert_array_equal(v["test"][:], self.test @ self.rotation)
        # Golden seed-42 labels for 32 documents followed by five queries.
        docs = np.array(
            [
                1,
                9,
                7,
                5,
                5,
                10,
                1,
                8,
                2,
                1,
                6,
                11,
                8,
                9,
                8,
                9,
                6,
                1,
                10,
                5,
                6,
                4,
                2,
                11,
                9,
                7,
                4,
                9,
                6,
                5,
                5,
                2,
            ]
        )
        queries = np.array([1, 6, 10, 0, 10])
        expected = queries[:, None] == docs[None, :]
        np.testing.assert_array_equal(p["pred_mask"][:], expected)
        np.testing.assert_array_equal(p["pred_mt"][:], expected.T)
        np.testing.assert_array_equal(p["neighbors"][:], self.reference["neighbors"])
        np.testing.assert_array_equal(
            v["neighbors_unfiltered"][:], self.reference["neighbors_unfiltered"]
        )
        self.assertEqual(v["train"].dtype, np.dtype("float32"))
        self.assertEqual(v["train"].chunks, (3, 2))
        self.assertTrue((destination / prep.MANIFEST).is_file())
        self.assertFalse((destination / ".sift12-preparation.lock").exists())
        with patch.object(
            prep, "write_stores", side_effect=AssertionError("must reuse")
        ):
            prep.prepare(destination)

    def test_refuses_existing_store_without_completion_marker(self):
        destination = self.root / "output"
        store = destination / prep.VECTOR_STORE
        store.mkdir(parents=True)
        (store / "keep").write_text("existing data")
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            prep.prepare(destination, self.source)
        self.assertEqual((store / "keep").read_text(), "existing data")

    def test_failed_generation_does_not_publish_partial_stores(self):
        destination = self.root / "output"
        with patch.object(
            prep, "write_stores", side_effect=RuntimeError("interrupted")
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                prep.prepare(destination, self.source)
        self.assertEqual(list(destination.iterdir()), [])

    def test_wrong_source_rejected_before_writing(self):
        with self.source.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            prep.prepare(self.root / "output", self.source)
        self.assertFalse((self.root / "output" / prep.VECTOR_STORE).exists())

    def test_download_checks_bytes_and_reuses_cache(self):
        destination = self.root / "raw/download.hdf5"
        with patch.object(
            prep, "urlopen", return_value=io.BytesIO(self.source.read_bytes())
        ) as request:
            prep.download(destination)
            prep.download(destination)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())

    def test_bad_download_is_not_cached(self):
        destination = self.root / "raw/download.hdf5"
        with patch.object(prep, "urlopen", return_value=io.BytesIO(b"incomplete")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prep.download(destination)
        self.assertEqual(list(destination.parent.iterdir()), [])

    def test_incomplete_rerun_is_not_reported_complete(self):
        destination = self.root / "output"
        prep.prepare(destination, self.source)
        p = zarr.open_group(str(destination / prep.PREDICATE_STORE), mode="a")
        del p["neighbors"]
        with self.assertRaisesRegex(ValueError, "Incomplete or incompatible"):
            prep.prepare(destination)

    def test_concurrent_preparation_lock_is_preserved(self):
        destination = self.root / "output"
        destination.mkdir()
        lock = destination / ".sift12-preparation.lock"
        lock.touch()
        with self.assertRaisesRegex(ValueError, "Preparation lock exists"):
            prep.prepare(destination, self.source)
        self.assertTrue(lock.exists())


if __name__ == "__main__":
    unittest.main()
