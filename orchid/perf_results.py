"""Shared result computation for ORCHID and baseline performance runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

import numpy as np


BASELINE_TIMING_STAGES = (
    "predicate_eval",
    "mask_materialize",
    "vector_search",
    "other",
    "total",
)
CLUSTER_TIMING_STAGES = (
    "predicate_eval",
    "mask_materialize",
    "cluster_scoring",
    "cluster_rank",
    "search_prep",
    "vector_search",
    "other",
    "total",
)
ORCHID_TIMING_STAGES = CLUSTER_TIMING_STAGES
IVF_TIMING_STAGES = CLUSTER_TIMING_STAGES

# Two-sided 95% Student-t critical values for 1 through 30 degrees of freedom.
# Values above 30 degrees of freedom conservatively use the df=30 value.
_T_95_CRITICAL = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


@dataclass(frozen=True, slots=True)
class Measurement:
    elapsed_seconds: float
    qps: float
    iteration_qps: tuple[float, ...] = ()
    qps_ci_lower: float | None = None
    qps_ci_upper: float | None = None
    qps_relative_ci_half_width: float | None = None
    converged: bool = True


@dataclass(frozen=True, slots=True)
class SweepResult:
    parameter: int
    recall: float
    qps: float
    timing_samples: int
    timing_summary: dict[str, float]
    transfer_ratio: float = 0.0


def run_measured_iterations(
    execute: Callable[[bool], None],
    *,
    warmup: int,
    iterations: int,
    num_queries: int,
    before_measure: Callable[[], None] | None = None,
    min_iterations: int | None = None,
    qps_relative_ci_half_width: float | None = None,
) -> Measurement:
    """Run fixed iterations, or adaptively stop at a precise mean QPS estimate.

    In adaptive mode, ``iterations`` is the maximum number of measured runs.
    The confidence interval is computed on complete-trace durations and then
    transformed to QPS so the point estimate remains total queries / total time.
    """
    if warmup < 0 or iterations <= 0 or num_queries <= 0:
        raise ValueError("warmup must be non-negative; iterations and queries positive")
    adaptive = min_iterations is not None or qps_relative_ci_half_width is not None
    if adaptive:
        if min_iterations is None or qps_relative_ci_half_width is None:
            raise ValueError(
                "adaptive measurement requires min_iterations and "
                "qps_relative_ci_half_width"
            )
        if not 3 <= min_iterations <= iterations:
            raise ValueError("min_iterations must be between 3 and iterations")
        if qps_relative_ci_half_width <= 0:
            raise ValueError("qps_relative_ci_half_width must be positive")
        adaptive_min_iterations = min_iterations
        adaptive_relative_half_width = qps_relative_ci_half_width
    else:
        adaptive_min_iterations = iterations + 1
        adaptive_relative_half_width = 0.0
    for _ in range(warmup):
        execute(False)
    if before_measure is not None:
        before_measure()

    start = perf_counter()
    iteration_start = start
    iteration_qps: list[float] = []
    iteration_seconds: list[float] = []
    qps_ci_lower: float | None = None
    qps_ci_upper: float | None = None
    relative_half_width: float | None = None
    converged = not adaptive
    for _ in range(iterations):
        execute(True)
        iteration_stop = perf_counter()
        iteration_elapsed = iteration_stop - iteration_start
        if iteration_elapsed <= 0:
            raise RuntimeError("measured iteration time must be positive")
        iteration_seconds.append(iteration_elapsed)
        iteration_qps.append(num_queries / iteration_elapsed)
        iteration_start = iteration_stop

        if adaptive and len(iteration_seconds) >= adaptive_min_iterations:
            mean_seconds = fmean(iteration_seconds)
            degrees_of_freedom = len(iteration_seconds) - 1
            critical = _T_95_CRITICAL[min(degrees_of_freedom, 30)]
            margin_seconds = (
                critical * stdev(iteration_seconds) / sqrt(len(iteration_seconds))
            )
            duration_lower = mean_seconds - margin_seconds
            duration_upper = mean_seconds + margin_seconds
            qps_ci_lower = num_queries / duration_upper
            if duration_lower <= 0:
                qps_ci_upper = float("inf")
                relative_half_width = float("inf")
            else:
                qps_ci_upper = num_queries / duration_lower
                estimated_qps = num_queries / mean_seconds
                relative_half_width = max(
                    estimated_qps - qps_ci_lower,
                    qps_ci_upper - estimated_qps,
                ) / estimated_qps
            if relative_half_width <= adaptive_relative_half_width:
                converged = True
                break

    elapsed = sum(iteration_seconds)
    return Measurement(
        elapsed,
        num_queries * len(iteration_seconds) / elapsed,
        tuple(iteration_qps),
        qps_ci_lower,
        qps_ci_upper,
        relative_half_width,
        converged,
    )


def require_measurement_convergence(
    measurement: Measurement,
    *,
    context: str,
) -> None:
    """Reject an adaptive measurement that exhausted its iteration cap."""
    if measurement.converged:
        return
    relative_half_width = measurement.qps_relative_ci_half_width
    if relative_half_width is None:
        raise ValueError("measurement has no QPS confidence interval")
    raise RuntimeError(
        f"QPS confidence interval did not converge for {context}: "
        f"runs={len(measurement.iteration_qps)}, "
        f"QPS={measurement.qps:.2f}, "
        f"relative half-width={relative_half_width:.2%}"
    )


def require_iteration_agreement(
    measurement: Measurement,
    *,
    context: str,
    threshold: float = 0.05,
) -> None:
    """Reject a two-run measurement when run 2 diverges too far from run 1."""
    if threshold < 0:
        raise ValueError("agreement threshold must be non-negative")
    if len(measurement.iteration_qps) != 2:
        return

    first_qps, second_qps = measurement.iteration_qps
    absolute_difference = abs(second_qps - first_qps)
    if absolute_difference <= threshold * first_qps:
        return

    disagreement = absolute_difference / first_qps
    raise RuntimeError(
        f"measured iterations disagree for {context}: "
        f"run 1={first_qps:.2f} QPS, run 2={second_qps:.2f} QPS, "
        f"difference={disagreement:.2%} exceeds {threshold:.2%}"
    )


def recall_at_k(labels: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall using the result semantics historically used by perf.py."""
    if k <= 0:
        raise ValueError("k must be positive")
    if labels.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("labels and ground truth must be matrices")
    if labels.shape[0] != ground_truth.shape[0] or ground_truth.shape[1] < k:
        raise ValueError("labels and ground truth shapes do not support recall@k")

    recalls = np.empty(labels.shape[0], dtype=np.float64)
    for query in range(labels.shape[0]):
        recalls[query] = np.intersect1d(ground_truth[query, :k], labels[query]).size / k
    return float(recalls.mean())


def summarize_timings(
    timings: np.ndarray,
    stages: Sequence[str],
    *,
    expected_samples: int | None = None,
) -> dict[str, float]:
    """Return the median and p95 milliseconds for each timing stage."""
    values = np.asarray(timings, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(stages):
        raise ValueError("unexpected timing matrix shape")
    if values.shape[0] == 0:
        raise ValueError("at least one timing sample is required")
    if expected_samples is not None and values.shape[0] != expected_samples:
        raise ValueError("unexpected number of timing samples")

    values_ms = values * 1e3
    summary: dict[str, float] = {}
    for column, stage in enumerate(stages):
        summary[f"{stage}_median_ms"] = float(np.median(values_ms[:, column]))
        summary[f"{stage}_p95_ms"] = float(np.percentile(values_ms[:, column], 95))
    return summary


def write_timing_results(
    results: Sequence[SweepResult],
    directory: str | Path,
    experiment: str,
    *,
    parameter_name: str,
) -> Path:
    """Write detailed local timing diagnostics for a completed sweep."""
    if not results:
        raise ValueError("at least one sweep result is required")
    if not experiment:
        raise ValueError("experiment must be non-empty")

    timing_metric_fields = list(results[0].timing_summary)
    if any(list(result.timing_summary) != timing_metric_fields for result in results):
        raise ValueError("all sweep results must use the same timing fields")

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    timing_path = output / f"{experiment}.timings.csv"

    fields = [parameter_name, "samples", "qps", *timing_metric_fields]
    with timing_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    parameter_name: result.parameter,
                    "samples": result.timing_samples,
                    "qps": result.qps,
                    **result.timing_summary,
                }
            )
    return timing_path


def write_sweep_results(
    results: Sequence[SweepResult],
    directory: str | Path,
    experiment: str,
    *,
    parameter_name: str,
) -> Path:
    """Write the recall/QPS curve after every sweep point has passed validation."""
    if not results or not experiment:
        raise ValueError("results and experiment must be non-empty")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{experiment}.csv"
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow((parameter_name, "recall", "qps"))
        writer.writerows((point.parameter, point.recall, point.qps) for point in results)
    return path


__all__ = [
    "BASELINE_TIMING_STAGES",
    "CLUSTER_TIMING_STAGES",
    "IVF_TIMING_STAGES",
    "Measurement",
    "ORCHID_TIMING_STAGES",
    "SweepResult",
    "recall_at_k",
    "require_iteration_agreement",
    "require_measurement_convergence",
    "run_measured_iterations",
    "summarize_timings",
    "write_timing_results",
    "write_sweep_results",
]
