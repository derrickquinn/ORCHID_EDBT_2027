import unittest
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import perf
from perf import _cli_override_dests, _clone_index_to_gpu, build_arg_parser
from src.baseline_perf_config import KNOWN_DATASETS
from src.build_ivf_config import BuildIvfConfig, load_build_ivf_config
from validate_configs import _validate_file
from query_settings import batch_size_for


class PerfConfigTest(unittest.TestCase):
    def test_cli_resolves_paper_config_without_external_services(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with patch("sys.argv", [
            "perf.py", "--config", str(root / "config/perf/orchid.yaml"),
            "--dset_cfg", str(root / "config/dataset/sift12.yaml"),
        ]):
            config, dataset = perf.parse_args()
        self.assertEqual(dataset.name, "sift12")
        self.assertEqual(batch_size_for(config.gpu, config.batch_size), 100)
        self.assertAlmostEqual(config.alpha, sqrt(0.205353))

    def test_defaults_use_paper_batch_and_warmup(self) -> None:
        config = BuildIvfConfig()

        self.assertEqual(batch_size_for(config.gpu, config.batch_size), 100)
        self.assertEqual(config.warmup, 3)
        self.assertFalse(config.gpu)
        self.assertEqual(config.gpu_device, 0)
        self.assertFalse(config.gpu_build)

    def test_gpu_clone_uses_explicit_device_and_64_bit_ids(self) -> None:
        cpu_index = object()
        gpu_index = object()
        resources = object()
        options = SimpleNamespace(indicesOptions=None, use_cuvs=True)
        clone = Mock(return_value=gpu_index)

        with (
            patch.object(perf.faiss, "get_num_gpus", return_value=4, create=True),
            patch.object(
                perf.faiss, "StandardGpuResources", return_value=resources, create=True
            ),
            patch.object(perf.faiss, "GpuClonerOptions", return_value=options, create=True),
            patch.object(perf.faiss, "index_cpu_to_gpu", clone, create=True),
            patch.object(perf.faiss, "INDICES_64_BIT", 0, create=True),
        ):
            actual_resources, actual_index = _clone_index_to_gpu(cpu_index, 2)

        self.assertIs(actual_resources, resources)
        self.assertIs(actual_index, gpu_index)
        self.assertEqual(options.indicesOptions, 0)
        self.assertFalse(options.use_cuvs)
        clone.assert_called_once_with(resources, 2, cpu_index, options)

    def test_gpu_clone_rejects_unavailable_device(self) -> None:
        with patch.object(perf.faiss, "get_num_gpus", return_value=2):
            with self.assertRaisesRegex(ValueError, "found 2 GPUs"):
                _clone_index_to_gpu(object(), 2)

    def test_orchid_config_uses_calibrated_parameters(self) -> None:
        config_path = Path(__file__).parents[1] / "config/perf/orchid.yaml"
        calibrated = {
            "laion_all": (0.00273842, 0.0237137),
            "laion_neg": (0.649382, 0.0237137),
            "laion_pos": (0.0365174, 0.0153993),
            "sift12": (0.205353, 0.01),
            "sift24": (0.133352, 0.01),
            "sift48": (0.205353, 0.01),
            "sift_r10": (0.316228, 0.01),
            "sift_r20": (0.205353, 0.01),
            "sift_r40": (0.205353, 0.0237137),
            "yfcc": (0.000133352, 0.000486968),
            "yfcc_single": (0.000749894, 0.000486968),
        }

        for dataset, (alpha, beta) in calibrated.items():
            with self.subTest(dataset=dataset):
                config = load_build_ivf_config(config_path, dataset_name=dataset)
                self.assertEqual(batch_size_for(config.gpu, config.batch_size), 100)
                self.assertEqual(config.warmup, 1)
                self.assertEqual(config.iterations, 100)
                self.assertEqual(config.min_iterations, 3)
                self.assertEqual(config.qps_relative_ci_half_width, 0.05)
                self.assertAlmostEqual(config.alpha, sqrt(alpha))
                self.assertAlmostEqual(config.beta, beta)

    def test_all_perf_configs_validate(self) -> None:
        config_dir = Path(__file__).parents[1] / "config/perf"

        for config_path in config_dir.glob("*.yaml"):
            with self.subTest(config=config_path.name):
                self.assertEqual(_validate_file(config_path), [])

    def test_all_perf_configs_keep_detailed_timings_local(self) -> None:
        config_dir = Path(__file__).parents[1] / "config/perf"

        for config_path in config_dir.glob("*.yaml"):
            if config_path.stem in {"acorn", "navix", "cagra"}:
                continue
            with self.subTest(config=config_path.name):
                config = load_build_ivf_config(config_path)
                self.assertEqual(config.log_base, "logs")

    def test_orchid_p_config_uses_calibrated_sift_betas(self) -> None:
        config_path = Path(__file__).parents[1] / "config/perf/orchid-p.yaml"
        calibrated = {
            "sift12": 0.0156152,
            "sift24": 0.0156152,
            "sift48": 0.0210175,
            "sift_r10": 0.0116016,
            "sift_r20": 0.0156152,
            "sift_r40": 0.0116016,
        }

        for dataset, beta in calibrated.items():
            with self.subTest(dataset=dataset):
                config = load_build_ivf_config(config_path, dataset_name=dataset)
                self.assertEqual(config.alpha, 0.0)
                self.assertAlmostEqual(config.beta, beta)
                self.assertEqual(config.warmup, 1)
                self.assertEqual(config.iterations, 100)
                self.assertEqual(config.min_iterations, 3)
                self.assertEqual(config.qps_relative_ci_half_width, 0.05)

    def test_orchid_p_config_uses_calibrated_yfcc_betas(self) -> None:
        config_dir = Path(__file__).parents[1] / "config/perf"
        calibrated = {
            "yfcc": 0.000594557,
            "yfcc_single": 0.000441734,
        }

        for dataset, beta in calibrated.items():
            with self.subTest(dataset=dataset, method="orchid-p"):
                config = load_build_ivf_config(
                    config_dir / "orchid-p.yaml", dataset_name=dataset
                )
                self.assertEqual(config.alpha, 0.0)
                self.assertAlmostEqual(config.beta, beta)
                self.assertEqual(config.warmup, 1)
                self.assertEqual(config.iterations, 100)
                self.assertEqual(config.min_iterations, 3)
                self.assertEqual(config.qps_relative_ci_half_width, 0.05)

            with self.subTest(dataset=dataset, method="ivf"):
                config = load_build_ivf_config(
                    config_dir / "ivf.yaml", dataset_name=dataset
                )
                self.assertEqual(config.high_nprobe, 4096)

    def test_ivf_uses_adaptive_measurements_for_every_dataset(self) -> None:
        config_path = Path(__file__).parents[1] / "config/perf/ivf.yaml"

        for dataset in KNOWN_DATASETS:
            with self.subTest(dataset=dataset):
                config = load_build_ivf_config(config_path, dataset_name=dataset)
                self.assertEqual(config.warmup, 1)
                self.assertEqual(config.iterations, 100)
                self.assertEqual(config.min_iterations, 3)
                self.assertEqual(config.qps_relative_ci_half_width, 0.05)

    def test_laion_baselines_use_calibrated_parameters(self) -> None:
        config_dir = Path(__file__).parents[1] / "config/perf"
        expected_ivf_high_nprobe = {
            "laion_all": 2048,
            "laion_pos": 1024,
            "laion_neg": 2048,
        }

        for dataset in expected_ivf_high_nprobe:
            with self.subTest(dataset=dataset, method="orchid-p"):
                config = load_build_ivf_config(
                    config_dir / "orchid-p.yaml", dataset_name=dataset
                )
                self.assertEqual(config.alpha, 0.0)
                self.assertAlmostEqual(config.beta, 0.0210175)
                self.assertEqual(config.warmup, 1)
                self.assertEqual(config.iterations, 100)
                self.assertEqual(config.min_iterations, 3)
                self.assertEqual(config.qps_relative_ci_half_width, 0.05)

            with self.subTest(dataset=dataset, method="ivf"):
                config = load_build_ivf_config(
                    config_dir / "ivf.yaml", dataset_name=dataset
                )
                self.assertEqual(config.alpha, 0.0)
                self.assertEqual(config.beta, 0.0)
                self.assertEqual(config.warmup, 1)
                self.assertEqual(config.iterations, 100)
                self.assertEqual(config.min_iterations, 3)
                self.assertEqual(config.qps_relative_ci_half_width, 0.05)
                self.assertEqual(config.high_nprobe, expected_ivf_high_nprobe[dataset])


if __name__ == "__main__":
    unittest.main()
