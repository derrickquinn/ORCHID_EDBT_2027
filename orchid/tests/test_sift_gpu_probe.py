"""GPU readiness checks run with mocked devices and real bitmap result types."""

import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probe_sift_gpu.py"
spec = importlib.util.spec_from_file_location("probe_sift_gpu", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SiftGpuProbeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.options = SimpleNamespace()
        self.faiss = SimpleNamespace(
            get_num_gpus=Mock(return_value=1),
            StandardGpuResources=Mock(return_value=object()),
            GpuClonerOptions=Mock(return_value=self.options),
            index_cpu_to_gpu=Mock(return_value=object()),
            IndexFlatIP=Mock(return_value=object()),
            INDICES_64_BIT=0,
        )
        self.span = Mock()
        self.cpu_result = np.array(
            [1, 1, 4, 2, 128, 0], dtype=np.uint8
        )
        def collect():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            loop.call_soon(future.set_result, self.cpu_result)
            return future
        self.span.collect_to_cpu.side_effect = collect

    def materialized(self):
        loop = asyncio.get_running_loop()
        stamps = [loop.create_future(), loop.create_future()]
        for stamp in stamps:
            stamp.set_result(0.0)
        return (self.span, *stamps)

    async def test_rejects_cpu_only_faiss(self):
        with (
            patch.dict("sys.modules", {"faiss": SimpleNamespace()}),
            self.assertRaisesRegex(RuntimeError, "FAISS is CPU-only"),
        ):
            await probe.probe_gpu()

    async def test_rejects_host_without_visible_gpu(self):
        self.faiss.get_num_gpus.return_value = 0
        with (
            patch.dict("sys.modules", {"faiss": self.faiss}),
            self.assertRaisesRegex(RuntimeError, "no CUDA GPU"),
        ):
            await probe.probe_gpu()
        self.faiss.StandardGpuResources.assert_not_called()

    async def test_rejects_failed_gpu_clone(self):
        self.faiss.index_cpu_to_gpu.side_effect = RuntimeError("clone failed")
        with (
            patch.dict("sys.modules", {"faiss": self.faiss}),
            self.assertRaisesRegex(RuntimeError, "clone failed"),
        ):
            await probe.probe_gpu()

    async def test_exercises_gpu_clone_and_mixed_bitmap_results(self):
        with (
            patch.dict("sys.modules", {"faiss": self.faiss}),
            patch(
                "predicate_index.materialize_results_gpu", return_value=self.materialized()
            ) as materialize,
        ):
            await probe.probe_gpu()
        self.assertEqual(self.options.indicesOptions, self.faiss.INDICES_64_BIT)
        self.assertFalse(self.options.use_cuvs)
        self.faiss.index_cpu_to_gpu.assert_called_once_with(
            self.faiss.StandardGpuResources.return_value,
            0,
            self.faiss.IndexFlatIP.return_value,
            self.options,
        )
        results, n_docs = materialize.call_args.args
        self.assertEqual(n_docs, 19)
        np.testing.assert_array_equal(results[0].ids, [0, 8, 18])
        np.testing.assert_array_equal(results[1].bits, [2, 128, 0])
        self.span.collect_to_cpu.assert_called_once_with()

    async def test_rejects_incorrect_gpu_materialization(self):
        self.cpu_result = np.zeros(6, dtype=np.uint8)
        with (
            patch.dict("sys.modules", {"faiss": self.faiss}),
            patch("predicate_index.materialize_results_gpu", return_value=self.materialized()),
            self.assertRaises(AssertionError),
        ):
            await probe.probe_gpu()


    async def test_identifies_utils_cpp_kernel_image_failure(self):
        with (
            patch.dict("sys.modules", {"faiss": self.faiss}),
            patch(
                "predicate_index.materialize_results_gpu",
                side_effect=RuntimeError("cudaErrorNoKernelImageForDevice"),
            ),
            self.assertRaisesRegex(
                RuntimeError, r"utils_cpp GPU mask.*NoKernelImage.*menu 12"
            ),
        ):
            await probe.probe_gpu()


class CudaArchitecturesTest(unittest.TestCase):
    def test_resolves_explicit_targets_for_all_visible_gpus(self):
        cuda = Mock()
        cuda.device_count.return_value = 3
        cuda.get_device_capability.side_effect = [(8, 9), (7, 5), (8, 9)]
        cuda.get_device_name.return_value = "Mock GPU"
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertEqual(probe.cuda_architectures(), "75;89")

    def test_single_gpu_target_includes_no_native_or_real_only_suffix(self):
        cuda = Mock()
        cuda.device_count.return_value = 1
        cuda.get_device_capability.return_value = (7, 5)
        cuda.get_device_name.return_value = "Mock T4"
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertEqual(probe.cuda_architectures(), "75")

    def test_refuses_to_guess_without_visible_gpu(self):
        cuda = Mock()
        cuda.device_count.return_value = 0
        with (
            patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}),
            self.assertRaisesRegex(RuntimeError, "no CUDA GPU is visible"),
        ):
            probe.cuda_architectures()



if __name__ == "__main__":
    unittest.main()
