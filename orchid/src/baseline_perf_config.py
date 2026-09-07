from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from query_settings import batch_size_for, validate_split


Metric = Literal["l2", "ip"]
MetadataMode = Literal["dummy", "workload_scalar"]
class BaselineDatasetOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int | None = None
    warmup: int | None = None
    min_iterations: int | None = None
    qps_relative_ci_half_width: float | None = None
    ef_search: list[int] | None = None
    itopk_size: list[int] | None = None
    metric: Metric | None = None
    normalize: bool | None = None
    gamma: int | None = None
    m_beta: int | None = None
    metadata_mode: MetadataMode | None = None
    index_name: str | None = None


class BaselinePerfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: Literal["acorn", "navix", "cagra"]
    num_queries: int = 10000
    query_start: int = 1024
    batch_size: int | None = None
    iterations: int = 10
    warmup: int = 3
    min_iterations: int | None = None
    qps_relative_ci_half_width: float | None = None
    k: int = 10
    threads: int = 0
    cache_base: str = "cached"
    index_base: str = "cached/baselines"
    log_base: str = "logs"
    exp_name: str
    ef_search: list[int] | None = None
    itopk_size: list[int] | None = None
    metric: Metric
    normalize: bool = True
    validate_predicates: bool = False
    m: int | None = None
    ef_construction: int | None = None
    gamma: int | None = None
    m_beta: int | None = None
    metadata_mode: MetadataMode | None = None
    index_name: str | None = None
    graph_degree: int | None = None
    intermediate_graph_degree: int | None = None
    device: int | None = None

    @field_validator("ef_search", "itopk_size")
    @classmethod
    def _validate_search_sweep(cls, values: list[int] | None) -> list[int] | None:
        if values is None:
            return None
        if not values or any(value <= 0 for value in values):
            raise ValueError("search sweeps must contain positive values")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("search sweep values must be strictly increasing")
        return values

    @model_validator(mode="after")
    def _validate_fields(self) -> BaselinePerfConfig:
        self.batch_size = batch_size_for(self.baseline == "cagra", self.batch_size)
        validate_split(self.query_start, self.num_queries, 512, calibration=False)
        positive = (self.num_queries, self.batch_size, self.iterations, self.k)
        if any(value <= 0 for value in positive):
            raise ValueError(
                "query, batching, iteration, k, and build values must be positive"
            )
        if self.warmup < 0 or self.threads < 0:
            raise ValueError("warmup and threads must be non-negative")
        adaptive = (
            self.min_iterations is not None
            or self.qps_relative_ci_half_width is not None
        )
        if adaptive:
            if (
                self.min_iterations is None
                or self.qps_relative_ci_half_width is None
            ):
                raise ValueError(
                    "adaptive measurement requires min_iterations and "
                    "qps_relative_ci_half_width"
                )
            if not 3 <= self.min_iterations <= self.iterations:
                raise ValueError("min_iterations must be between 3 and iterations")
            if self.qps_relative_ci_half_width <= 0:
                raise ValueError("qps_relative_ci_half_width must be positive")
        if self.index_name is not None and not self.index_name.strip():
            raise ValueError("index_name must be non-empty when provided")
        acorn_fields = (self.gamma, self.m_beta, self.metadata_mode)
        graph_fields = (
            self.graph_degree,
            self.intermediate_graph_degree,
            self.device,
        )
        if self.baseline in {"acorn", "navix"}:
            if self.ef_search is None or self.itopk_size is not None:
                raise ValueError(f"{self.baseline.upper()} requires only ef_search")
            if self.m is None or (self.baseline == "navix" and self.ef_construction is None):
                raise ValueError(
                    f"{self.baseline.upper()} requires m and ef_construction"
                )
            if self.m <= 0 or (self.ef_construction is not None and self.ef_construction <= 0):
                raise ValueError("m and ef_construction must be positive")
            if any(value is not None for value in graph_fields):
                raise ValueError("CAGRA construction fields are CAGRA-only")
        else:
            if self.itopk_size is None or self.ef_search is not None:
                raise ValueError("CAGRA requires only itopk_size")
            if self.m is not None or self.ef_construction is not None:
                raise ValueError("m and ef_construction do not apply to CAGRA")
            if any(value is None for value in graph_fields):
                raise ValueError("CAGRA requires all CAGRA construction fields")
            assert self.graph_degree is not None
            assert self.intermediate_graph_degree is not None
            assert self.device is not None
            if (
                self.graph_degree <= 0
                or self.intermediate_graph_degree <= 0
                or self.device < 0
            ):
                raise ValueError("CAGRA construction values must be positive")
            if self.intermediate_graph_degree < self.graph_degree:
                raise ValueError(
                    "CAGRA intermediate_graph_degree must be at least graph_degree"
                )

        if self.baseline == "acorn":
            if any(value is None for value in acorn_fields):
                raise ValueError("ACORN requires gamma, m_beta, and metadata_mode")
            assert self.gamma is not None and self.m_beta is not None
            if self.gamma <= 0 or self.m_beta <= 0:
                raise ValueError("ACORN gamma and m_beta must be positive")
        elif any(value is not None for value in acorn_fields):
            raise ValueError("gamma, m_beta, and metadata_mode are ACORN-only")
        return self


def _extract_args(payload: Any, dataset_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Baseline perf config must be a mapping")
    if payload.get("kind") != "baseline_perf":
        raise ValueError(f"Expected kind 'baseline_perf', got {payload.get('kind')!r}")
    raw_args = payload.get("args")
    if not isinstance(raw_args, dict):
        raise TypeError("Baseline perf config must include an 'args' mapping")

    per_dataset = raw_args.get("per_dataset")
    if not isinstance(per_dataset, dict):
        raise TypeError("Baseline perf config must include a 'per_dataset' mapping")
    unknown = set(per_dataset) - set(KNOWN_DATASETS)
    if unknown:
        raise ValueError(f"Unknown baseline datasets: {', '.join(sorted(unknown))}")
    if dataset_name not in per_dataset:
        raise ValueError(f"No baseline settings for dataset {dataset_name!r}")

    overrides = BaselineDatasetOverrides.model_validate(
        per_dataset[dataset_name]
    ).model_dump(exclude_none=True)
    shared = {key: value for key, value in raw_args.items() if key != "per_dataset"}
    return {**shared, **overrides}


KNOWN_DATASETS = (
    "laion_all",
    "laion_neg",
    "laion_pos",
    "sift12",
    "sift24",
    "sift48",
    "sift_r10",
    "sift_r20",
    "sift_r40",
    "yfcc",
    "yfcc_single",
)
SCALAR_METADATA_DATASETS = frozenset({"sift12", "sift24", "sift48"})


def load_baseline_perf_config(
    path: str | Path,
    *,
    dataset_name: str,
    baseline: str | None = None,
) -> BaselinePerfConfig:
    import yaml

    payload = yaml.safe_load(Path(path).read_text())
    config = BaselinePerfConfig.model_validate(_extract_args(payload, dataset_name))
    if (
        config.metadata_mode == "workload_scalar"
        and dataset_name not in SCALAR_METADATA_DATASETS
    ):
        raise ValueError(
            "workload_scalar metadata is only valid for SIFT equality workloads"
        )
    if baseline is not None and config.baseline != baseline:
        raise ValueError(
            f"Config selects {config.baseline!r}, but the runner is {baseline!r}"
        )
    return config


__all__ = [
    "BaselinePerfConfig",
    "KNOWN_DATASETS",
    "SCALAR_METADATA_DATASETS",
    "load_baseline_perf_config",
]
