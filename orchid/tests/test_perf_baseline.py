import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from perf_baseline import (
    TIMING_STAGES,
    parse_args,
    run_online_pass,
    run_precomputed_mask_pass,
)
from predicate_index import ByteMask, PackedBitmap, PostingList


class _FakePredicateWorkload:
    def __init__(self, rows: np.ndarray, *, packed: bool = False) -> None:
        self.rows = rows
        self.packed = packed

    def evaluate_batch(self, start: int, stop: int, *, threads: int = 0) -> list:
        del threads
        results = []
        for query, row in enumerate(self.rows[start:stop], start=start):
            ids = np.flatnonzero(row).astype(np.uint32)
            if query % 2:
                results.append(
                    PackedBitmap(np.packbits(row, bitorder="little"))
                    if self.packed
                    else ByteMask(row)
                )
            else:
                results.append(PostingList(ids))
        return results


class _FakeIndex:
    def __init__(self, n_docs: int, dimension: int) -> None:
        self.ntotal = n_docs
        self.d = dimension
        self.seen_masks: list[np.ndarray] = []

    def search(
        self,
        queries: np.ndarray,
        masks: np.ndarray,
        k: int,
        ef_search: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del queries, ef_search
        self.seen_masks.append(masks.copy())
        labels = np.full((len(masks), k), -1, dtype=np.int64)
        for query, row in enumerate(masks):
            allowed = np.flatnonzero(row)
            labels[query, : min(k, len(allowed))] = allowed[:k]
        return np.zeros_like(labels, dtype=np.float32), labels


class PerfBaselineTest(unittest.TestCase):
    def test_acorn_config_is_resolved_for_the_dataset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        argv = [
            "perf_acorn.py",
            "--config",
            str(root / "config/perf/acorn.yaml"),
            "--dset-cfg",
            str(root / "config/dataset/sift24.yaml"),
            "--ef-search",
            "32",
            "64",
        ]
        with patch("sys.argv", argv):
            args, dataset, resolved = parse_args("acorn")

        self.assertEqual(dataset.name, "sift24")
        self.assertEqual(args.index, Path("cached/baselines/acorn/sift24_efc768.faiss"))
        self.assertEqual(args.ef_search, [32, 64])
        self.assertTrue(args.acorn_workload_metadata)
        self.assertEqual(resolved["metadata_mode"], "workload_scalar")

    def test_navix_sift_configs_share_one_index(self) -> None:
        root = Path(__file__).resolve().parents[1]
        indexes = set()
        for dataset in (
            "sift12",
            "sift24",
            "sift48",
            "sift_r10",
            "sift_r20",
            "sift_r40",
        ):
            argv = [
                "perf_navix.py",
                "--config",
                str(root / "config/perf/navix.yaml"),
                "--dset-cfg",
                str(root / f"config/dataset/{dataset}.yaml"),
            ]
            with patch("sys.argv", argv):
                args, _, _ = parse_args("navix")
            indexes.add(args.index)
        self.assertEqual(indexes, {Path("cached/baselines/navix/sift.faiss")})

    def test_navix_vector_families_share_indexes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        families = {
            "laion": ("laion_all", "laion_neg", "laion_pos"),
            "yfcc": ("yfcc", "yfcc_single"),
        }
        for index_name, datasets in families.items():
            indexes = set()
            for dataset in datasets:
                argv = [
                    "perf_navix.py",
                    "--config",
                    str(root / "config/perf/navix.yaml"),
                    "--dset-cfg",
                    str(root / f"config/dataset/{dataset}.yaml"),
                ]
                with patch("sys.argv", argv):
                    args, _, _ = parse_args("navix")
                indexes.add(args.index)
            self.assertEqual(
                indexes, {Path(f"cached/baselines/navix/{index_name}.faiss")}
            )

    def test_cagra_vector_families_share_gpu_indexes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        families = {
            "laion": ("laion_all", "laion_neg", "laion_pos"),
            "sift": (
                "sift12",
                "sift24",
                "sift48",
                "sift_r10",
                "sift_r20",
                "sift_r40",
            ),
            "yfcc": ("yfcc", "yfcc_single"),
        }
        for index_name, datasets in families.items():
            indexes = set()
            for dataset in datasets:
                argv = [
                    "perf_cagra.py",
                    "--config",
                    str(root / "config/perf/cagra.yaml"),
                    "--dset-cfg",
                    str(root / f"config/dataset/{dataset}.yaml"),
                ]
                with patch("sys.argv", argv):
                    args, _, resolved = parse_args("cagra")
                indexes.add(args.index)
                self.assertIsNone(resolved.get("ef_search"))
                self.assertEqual(resolved["itopk_size"][0], 16)
            self.assertEqual(
                indexes, {Path(f"cached/baselines/cagra/{index_name}.cagra")}
            )

    def test_acorn_range_configs_use_separate_indexes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        indexes = set()
        for dataset in ("sift_r10", "sift_r20", "sift_r40"):
            argv = [
                "perf_acorn.py",
                "--config",
                str(root / "config/perf/acorn.yaml"),
                "--dset-cfg",
                str(root / f"config/dataset/{dataset}.yaml"),
            ]
            with patch("sys.argv", argv):
                args, _, _ = parse_args("acorn")
            indexes.add(args.index)
        self.assertEqual(indexes, {
            Path("cached/baselines/acorn/sift_r10_efc3200.faiss"),
            Path("cached/baselines/acorn/sift_r20_efc1600.faiss"),
            Path("cached/baselines/acorn/sift_range_efc32000.faiss"),
        })

    def test_acorn_laion_configs_share_one_index(self) -> None:
        root = Path(__file__).resolve().parents[1]
        indexes = set()
        for dataset in ("laion_all", "laion_neg", "laion_pos"):
            argv = [
                "perf_acorn.py",
                "--config",
                str(root / "config/perf/acorn.yaml"),
                "--dset-cfg",
                str(root / f"config/dataset/{dataset}.yaml"),
            ]
            with patch("sys.argv", argv):
                args, _, _ = parse_args("acorn")
            indexes.add(args.index)
        self.assertEqual(indexes, {Path("cached/baselines/acorn/laion_efc960.faiss")})

    def test_online_pass_batches_and_materializes_mixed_results(self) -> None:
        rows = np.zeros((5, 11), dtype=np.uint8)
        rows[0, [0, 2]] = 1
        rows[1, [1, 10]] = 1
        rows[2, [3]] = 1
        rows[3, [4, 8]] = 1
        rows[4, [5, 9]] = 1
        workload = _FakePredicateWorkload(rows)
        index = _FakeIndex(n_docs=11, dimension=3)
        queries = np.zeros((5, 3), dtype=np.float32)
        mask_buffer = np.empty((2, 11), dtype=np.uint8)

        result = run_online_pass(
            index,
            workload,
            queries,
            mask_buffer,
            k=2,
            ef_search=32,
            batch_size=2,
            threads=2,
        )

        np.testing.assert_array_equal(np.concatenate(index.seen_masks), rows)
        self.assertEqual(result.labels.shape, (5, 2))
        self.assertEqual(result.timings.shape, (3, len(TIMING_STAGES)))
        self.assertTrue(np.all(result.timings >= 0))
        np.testing.assert_allclose(
            result.timings[:, :-1].sum(axis=1), result.timings[:, -1]
        )
        np.testing.assert_array_equal(result.labels[:, 0], [0, 1, 3, 4, 5])

    def test_precomputed_mask_pass_only_times_search(self) -> None:
        masks = np.zeros((5, 11), dtype=np.uint8)
        masks[0, [0, 2]] = 1
        masks[1, [1, 10]] = 1
        masks[2, [3]] = 1
        masks[3, [4, 8]] = 1
        masks[4, [5, 9]] = 1
        index = _FakeIndex(n_docs=11, dimension=3)
        queries = np.zeros((5, 3), dtype=np.float32)

        result = run_precomputed_mask_pass(
            index,
            queries,
            masks,
            k=2,
            ef_search=32,
            batch_size=2,
        )

        np.testing.assert_array_equal(np.concatenate(index.seen_masks), masks)
        self.assertEqual(result.labels.shape, (5, 2))
        self.assertEqual(result.timings.shape, (3, len(TIMING_STAGES)))
        np.testing.assert_array_equal(result.timings[:, :2], 0.0)
        np.testing.assert_allclose(
            result.timings[:, :-1].sum(axis=1), result.timings[:, -1]
        )

    def test_online_pass_materializes_packed_masks_for_cagra(self) -> None:
        rows = np.zeros((5, 11), dtype=np.uint8)
        rows[0, [0, 2]] = 1
        rows[1, [1, 10]] = 1
        rows[2, [3]] = 1
        rows[3, [4, 8]] = 1
        rows[4, [5, 9]] = 1
        workload = _FakePredicateWorkload(rows, packed=True)
        index = _FakeIndex(n_docs=11, dimension=3)
        queries = np.zeros((5, 3), dtype=np.float32)
        mask_buffer = np.empty((2, 2), dtype=np.uint8)

        run_online_pass(
            index,
            workload,
            queries,
            mask_buffer,
            k=2,
            ef_search=64,
            batch_size=2,
            threads=2,
            packed_masks=True,
        )

        actual = np.concatenate(index.seen_masks)
        np.testing.assert_array_equal(
            actual, np.packbits(rows, axis=1, bitorder="little")
        )


if __name__ == "__main__":
    unittest.main()
