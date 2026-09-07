import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from perf_results import (
    BASELINE_TIMING_STAGES,
    CLUSTER_TIMING_STAGES,
    IVF_TIMING_STAGES,
    ORCHID_TIMING_STAGES,
    Measurement,
    SweepResult,
    recall_at_k,
    require_iteration_agreement,
    require_measurement_convergence,
    run_measured_iterations,
    summarize_timings,
    write_timing_results,
    write_sweep_results,
)


class PerfResultsTest(unittest.TestCase):
    def test_public_timing_stage_vocabularies(self) -> None:
        self.assertEqual(
            BASELINE_TIMING_STAGES,
            (
                "predicate_eval",
                "mask_materialize",
                "vector_search",
                "other",
                "total",
            ),
        )
        self.assertEqual(
            CLUSTER_TIMING_STAGES,
            (
                "predicate_eval",
                "mask_materialize",
                "cluster_scoring",
                "cluster_rank",
                "search_prep",
                "vector_search",
                "other",
                "total",
            ),
        )
        self.assertIs(ORCHID_TIMING_STAGES, CLUSTER_TIMING_STAGES)
        self.assertIs(IVF_TIMING_STAGES, CLUSTER_TIMING_STAGES)

    def test_measured_iterations_handle_warmup_and_qps(self) -> None:
        calls: list[bool] = []
        resets: list[bool] = []

        with patch(
            "perf_results.perf_counter", side_effect=[10.0, 11.0, 12.0, 13.0]
        ):
            measurement = run_measured_iterations(
                calls.append,
                warmup=2,
                iterations=3,
                num_queries=100,
                before_measure=lambda: resets.append(True),
            )

        self.assertEqual(calls, [False, False, True, True, True])
        self.assertEqual(resets, [True])
        self.assertEqual(measurement.elapsed_seconds, 3.0)
        self.assertEqual(measurement.qps, 100.0)
        self.assertEqual(measurement.iteration_qps, (100.0, 100.0, 100.0))

    def test_adaptive_measurement_stops_at_minimum_when_precise(self) -> None:
        calls: list[bool] = []

        with patch(
            "perf_results.perf_counter", side_effect=[10.0, 11.0, 12.0, 13.0]
        ):
            measurement = run_measured_iterations(
                calls.append,
                warmup=1,
                iterations=30,
                min_iterations=3,
                qps_relative_ci_half_width=0.05,
                num_queries=100,
            )

        self.assertEqual(calls, [False, True, True, True])
        self.assertEqual(measurement.iteration_qps, (100.0, 100.0, 100.0))
        self.assertEqual(measurement.qps_ci_lower, 100.0)
        self.assertEqual(measurement.qps_ci_upper, 100.0)
        self.assertEqual(measurement.qps_relative_ci_half_width, 0.0)
        self.assertTrue(measurement.converged)

    def test_adaptive_measurement_uses_cap_when_not_precise(self) -> None:
        with patch(
            "perf_results.perf_counter",
            side_effect=[0.0, 1.0, 3.0, 4.0, 6.0, 7.0],
        ):
            measurement = run_measured_iterations(
                lambda _: None,
                warmup=0,
                iterations=5,
                min_iterations=3,
                qps_relative_ci_half_width=0.05,
                num_queries=100,
            )

        self.assertEqual(len(measurement.iteration_qps), 5)
        self.assertAlmostEqual(measurement.qps, 500 / 7)
        self.assertGreater(measurement.qps_relative_ci_half_width, 0.05)
        self.assertFalse(measurement.converged)
        with self.assertRaisesRegex(RuntimeError, r"did not converge.*runs=5"):
            require_measurement_convergence(
                measurement,
                context="n_probe=1",
            )

    def test_two_iteration_agreement_accepts_five_percent_or_less(self) -> None:
        require_iteration_agreement(
            Measurement(2.0, 97.5, (100.0, 95.0)),
            context="n_probe=16",
        )

    def test_two_iteration_agreement_rejects_more_than_five_percent(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"n_probe=16.*run 1=100\.00 QPS.*run 2=94\.00 QPS.*6\.00%",
        ):
            require_iteration_agreement(
                Measurement(2.0, 97.0, (100.0, 94.0)),
                context="n_probe=16",
            )

    def test_iteration_agreement_ignores_non_two_run_measurements(self) -> None:
        require_iteration_agreement(
            Measurement(3.0, 90.0, (100.0, 70.0, 100.0)),
            context="n_probe=16",
        )

    def test_recall_matches_perf_semantics(self) -> None:
        labels = np.array([[1, 2], [8, 9]], dtype=np.int64)
        ground_truth = np.array([[2, 3], [8, 7]], dtype=np.int64)
        self.assertEqual(recall_at_k(labels, ground_truth, 2), 0.5)

    def test_timing_summary_uses_seconds_and_expected_samples(self) -> None:
        timings = np.array([[0.001, 0.004], [0.003, 0.008]])
        summary = summarize_timings(timings, ("prepare", "search"), expected_samples=2)
        self.assertEqual(summary["prepare_median_ms"], 2.0)
        self.assertEqual(summary["search_median_ms"], 6.0)
        self.assertAlmostEqual(summary["prepare_p95_ms"], 2.9)

    def test_writers_keep_recall_qps_and_stage_timings_in_local_csvs(self) -> None:
        results = [
            SweepResult(
                parameter=16,
                recall=0.75,
                qps=123.456,
                timing_samples=3,
                timing_summary={"search_median_ms": 1.5},
            )
        ]
        with TemporaryDirectory() as directory:
            timings = write_timing_results(
                results, directory, "method", parameter_name="ef_search"
            )
            self.assertEqual(
                Path(timings).read_text(),
                "ef_search,samples,qps,search_median_ms\n16,3,123.456,1.5\n",
            )
            curve = write_sweep_results(
                results, directory, "method", parameter_name="ef_search"
            )
            self.assertEqual(
                curve.read_text(), "ef_search,recall,qps\n16,0.75,123.456\n"
            )


if __name__ == "__main__":
    unittest.main()
