import tempfile
import unittest
from pathlib import Path

import yaml

from src.baseline_perf_config import KNOWN_DATASETS, load_baseline_perf_config


ROOT = Path(__file__).resolve().parents[1]


class BaselinePerfConfigTest(unittest.TestCase):
    def test_configs_cover_every_published_dataset(self) -> None:
        for baseline in ("acorn", "navix", "cagra"):
            path = ROOT / "config" / "perf" / f"{baseline}.yaml"
            for dataset in KNOWN_DATASETS:
                config = load_baseline_perf_config(
                    path, dataset_name=dataset, baseline=baseline
                )
                self.assertEqual(config.baseline, baseline)
                self.assertEqual(config.batch_size, 1 if baseline == "cagra" else 100)
                self.assertEqual(config.log_base, "logs")
                self.assertTrue(config.normalize)

    def test_cagra_uses_native_construction_and_search_parameters(self) -> None:
        config = load_baseline_perf_config(
            ROOT / "config" / "perf" / "cagra.yaml",
            dataset_name="sift12",
            baseline="cagra",
        )
        self.assertEqual(config.graph_degree, 32)
        self.assertEqual(config.intermediate_graph_degree, 64)
        self.assertEqual(
            config.itopk_size,
            [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        )
        self.assertIsNone(config.ef_search)
        self.assertIsNone(config.ef_construction)

    def test_acorn_metadata_policy_is_explicit(self) -> None:
        path = ROOT / "config" / "perf" / "acorn.yaml"
        for dataset in KNOWN_DATASETS:
            config = load_baseline_perf_config(
                path, dataset_name=dataset, baseline="acorn"
            )
            expected = (
                "workload_scalar"
                if dataset.startswith("sift") and "_r" not in dataset
                else "dummy"
            )
            self.assertEqual(config.metadata_mode, expected)

    def test_all_baselines_use_adaptive_measurements_for_every_dataset(self) -> None:
        for baseline in ("acorn", "navix", "cagra"):
            path = ROOT / "config" / "perf" / f"{baseline}.yaml"
            for dataset in KNOWN_DATASETS:
                with self.subTest(baseline=baseline, dataset=dataset):
                    config = load_baseline_perf_config(
                        path, dataset_name=dataset, baseline=baseline
                    )
                    self.assertEqual(config.warmup, 1)
                    self.assertEqual(config.iterations, 100)
                    self.assertEqual(config.min_iterations, 3)
                    self.assertEqual(config.qps_relative_ci_half_width, 0.05)

    def test_acorn_extends_selected_sift_sweeps_to_2048(self) -> None:
        path = ROOT / "config" / "perf" / "acorn.yaml"
        for dataset in ("sift24", "sift48", "sift_r10", "sift_r20"):
            with self.subTest(dataset=dataset):
                config = load_baseline_perf_config(
                    path, dataset_name=dataset, baseline="acorn"
                )
                self.assertEqual(config.ef_search[-1], 2048)

    def test_production_construction_parameters_are_baseline_specific(self) -> None:
        for baseline, expected in (("acorn", None), ("navix", 200)):
            config = load_baseline_perf_config(
                ROOT / "config" / "perf" / f"{baseline}.yaml",
                dataset_name="sift12",
                baseline=baseline,
            )
            self.assertEqual(config.ef_construction, expected)

    def test_rejects_scalar_metadata_for_multilabel_workload(self) -> None:
        source = yaml.safe_load((ROOT / "config" / "perf" / "acorn.yaml").read_text())
        source["args"]["per_dataset"]["yfcc"]["metadata_mode"] = "workload_scalar"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "SIFT equality"):
                load_baseline_perf_config(path, dataset_name="yfcc", baseline="acorn")


if __name__ == "__main__":
    unittest.main()
