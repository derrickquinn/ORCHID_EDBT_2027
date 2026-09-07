import unittest

import faiss
import numpy as np

from predicate_index import HybridCategoricalIndex, materialize_results
from predicate_workload import PredicateWorkload
from utils_cpp import cluster_legals, prepare_cluster_order


class ClusterMajorReindexingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clusters = np.array([2, 0, 1, 2, 0, 2], dtype=np.int32)
        (
            self.internal_to_external,
            self.cluster_offsets,
            self.cluster_counts,
        ) = prepare_cluster_order(self.clusters, n_list=4)

    def test_builds_stable_cluster_order(self) -> None:
        np.testing.assert_array_equal(
            self.internal_to_external,
            np.array([1, 4, 2, 0, 3, 5], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            self.cluster_counts,
            np.array([2, 1, 3, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            self.cluster_offsets,
            np.array([0, 2, 3, 6, 6], dtype=np.int32),
        )
        external_to_internal = np.empty(6, dtype=np.int64)
        external_to_internal[self.internal_to_external] = np.arange(6)
        np.testing.assert_array_equal(
            external_to_internal,
            np.array([3, 0, 2, 4, 1, 5], dtype=np.int64),
        )

    def test_reorders_documents_and_masks_consistently(self) -> None:
        docs = np.arange(12, dtype=np.float32).reshape(6, 2)
        masks = np.array(
            [
                [True, False, True, False, False, True],
                [False, True, False, True, True, False],
            ]
        )

        docs_internal = docs[self.internal_to_external]
        masks_internal = masks[:, self.internal_to_external]

        np.testing.assert_array_equal(
            docs_internal, docs[self.internal_to_external]
        )
        for query, mask_internal in zip(masks, masks_internal):
            selected_external = np.flatnonzero(query)
            selected_internal = np.flatnonzero(mask_internal)
            np.testing.assert_array_equal(
                np.sort(self.internal_to_external[selected_internal]),
                selected_external,
            )

    def test_result_mapping_preserves_missing_ids(self) -> None:
        internal = np.array([[0, 4, -1], [2, 5, 1]], dtype=np.int64)
        result = np.full_like(internal, -1)
        valid = internal >= 0
        result[valid] = self.internal_to_external[internal[valid]]
        np.testing.assert_array_equal(
            result,
            np.array([[1, 3, -1], [2, 5, 4]], dtype=np.int64),
        )

    def test_same_internal_bitmap_produces_cluster_selectivities(self) -> None:
        masks_external = np.array(
            [
                [True, False, True, False, False, True],
                [False, True, False, True, True, False],
            ]
        )
        masks_internal = masks_external[:, self.internal_to_external]
        packed = np.packbits(masks_internal, axis=1, bitorder="little")

        actual = cluster_legals(
            packed,
            self.cluster_offsets,
            self.cluster_counts,
            threads=1,
        )
        expected = np.zeros((2, 4), dtype=np.float32)
        for cluster in range(4):
            members = self.clusters == cluster
            if np.any(members):
                expected[:, cluster] = masks_external[:, members].mean(axis=1)

        np.testing.assert_allclose(actual, expected)

    def test_reindexed_faiss_search_matches_external_id_index(self) -> None:
        docs = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        clusters = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        internal_to_external, _, _ = prepare_cluster_order(clusters, n_list=2)
        queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        masks_external = np.array(
            [
                [True, False, True, True, True, False],
                [False, True, True, False, False, True],
            ],
            dtype=np.uint8,
        )

        external_index = self._build_ivf(docs, clusters)
        external_to_internal = np.empty(docs.shape[0], dtype=np.int64)
        external_to_internal[internal_to_external] = np.arange(docs.shape[0])
        internal_index = self._build_ivf(docs, clusters, external_to_internal)
        iq = np.array([[0, 1], [1, 0]], dtype=np.int64)
        dq = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        expected = self._search(
            external_index,
            queries,
            iq,
            dq,
            np.packbits(masks_external, axis=1, bitorder="little"),
        )
        document_offsets = np.array([0, 1, 2, 4, 5, 6, 7], dtype=np.int64)
        document_terms = np.array([10, 20, 10, 20, 10, 10, 20], dtype=np.int64)
        predicate_index = HybridCategoricalIndex.build(
            document_offsets,
            document_terms,
            document_ids=external_to_internal,
            dense_threshold=10,
        )
        predicate_workload = PredicateWorkload(
            predicate_index, "equality", np.array([10, 20])
        )
        predicate_results = predicate_workload.evaluate_batch(0, 2, threads=2)
        internal_masks = materialize_results(
            predicate_results,
            docs.shape[0],
            np.empty((2, 1), dtype=np.uint8),
            threads=2,
        )

        internal = self._search(
            internal_index,
            queries,
            iq,
            dq,
            internal_masks,
        )
        actual = np.full_like(internal, -1)
        valid = internal >= 0
        actual[valid] = internal_to_external[internal[valid]]

        np.testing.assert_array_equal(actual, expected)

    @staticmethod
    def _build_ivf(
        docs: np.ndarray,
        clusters: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> faiss.IndexIVFFlat:
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = faiss.index_factory(2, "IVF2,Flat", faiss.METRIC_INNER_PRODUCT)
        index.quantizer.add(centroids)
        index.is_trained = True
        if ids is None:
            ids = np.arange(docs.shape[0], dtype=np.int64)
        index.add_core(
            docs.shape[0],
            faiss.swig_ptr(np.ascontiguousarray(docs)),
            faiss.swig_ptr(ids),
            faiss.swig_ptr(np.ascontiguousarray(clusters, dtype=np.int64)),
        )
        return index

    @staticmethod
    def _search(
        index: faiss.IndexIVFFlat,
        queries: np.ndarray,
        iq: np.ndarray,
        dq: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        index.nprobe = 2
        params = faiss.SearchParametersIVF(nprobe=2)
        params.sel2d = faiss.IDSelector2DBitmap(np.ascontiguousarray(masks))
        _, indices = index.search_preassigned(
            queries, 3, iq, dq, params=params
        )
        return indices


if __name__ == "__main__":
    unittest.main()
