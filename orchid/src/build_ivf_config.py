from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator
from query_settings import validate_split, batch_size_for


class BuildIvfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_queries: int = 10000
    query_start: int = 1024
    beta: float = 0.0
    iterations: int = 10
    min_iterations: int | None = None
    qps_relative_ci_half_width: float | None = None
    warmup: int = 3
    low_nprobe: int = 1
    high_nprobe: int = 1024
    step_nprobe: int = 10
    cache_base: str = "cached"
    batch_size: int | None = None
    sample_queries: int = 512
    sample_docs: int = 512
    alpha: float = 0.0
    k: int = 10
    gpu: bool = False
    gpu_device: int = 0
    gpu_build: bool = False
    log_base: str = "logs"
    sig_dims: int = 64
    exp_name: str = ""
    metric: str = "ip"
    simd: bool = False
    val_flat: bool = False
    validate_predicates: bool = False

    @model_validator(mode="after")
    def _validate_queries(self):
        validate_split(
            self.query_start, self.num_queries, self.sample_queries, calibration=False
        )
        if self.sample_docs <= 0:
            raise ValueError("sample_docs must be positive")
        batch_size_for(self.gpu, self.batch_size)
        return self


class PerfDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float | None = None
    beta: float | None = None
    low_nprobe: int | None = None
    step_nprobe: int | None = None
    iterations: int | None = None
    min_iterations: int | None = None
    qps_relative_ci_half_width: float | None = None
    warmup: int | None = None
    high_nprobe: int | None = None


def _extract_args(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise ValueError("Config payload is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a mapping, got {type(payload).__name__}")
    if "kind" in payload:
        kind = payload["kind"]
        if kind != "build_ivf":
            raise ValueError(f"Expected kind 'build_ivf', got {kind!r}")
        if "args" not in payload:
            raise ValueError("Config payload with 'kind' must include 'args'")
    if "args" in payload:
        args = payload["args"]
        if not isinstance(args, dict):
            raise TypeError(f"Config args must be a mapping, got {type(args).__name__}")
        return args
    return payload


def _validate_config(args: dict[str, Any]) -> BuildIvfConfig:
    return BuildIvfConfig.model_validate(args)


def _extract_perf_args(payload: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
    raw_args = payload.get("args")
    if not isinstance(raw_args, dict):
        raise TypeError("Perf config must include an 'args' mapping")
    per_dataset = raw_args.get("per_dataset")
    if per_dataset is None:
        return dict(raw_args)
    if not isinstance(per_dataset, dict):
        raise TypeError("Perf config 'per_dataset' must be a mapping")

    shared_args = {k: v for k, v in raw_args.items() if k != "per_dataset"}

    per_dataset_entry = None
    if dataset_name is not None:
        per_dataset_entry = per_dataset.get(dataset_name)
        if per_dataset_entry is not None and not isinstance(per_dataset_entry, dict):
            raise TypeError("Perf config per-dataset entry must be a mapping")

    if per_dataset_entry is None:
        return shared_args

    per_dataset_args = PerfDatasetConfig.model_validate(per_dataset_entry).model_dump(
        exclude_none=True
    )
    return {**shared_args, **per_dataset_args}


def load_build_ivf_config(
    path: str | Path, *, dataset_name: str | None = None
) -> BuildIvfConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    if isinstance(payload, dict) and payload.get("kind") == "perf":
        args = _extract_perf_args(payload, dataset_name)
    else:
        args = _extract_args(payload)
    return _validate_config(args)
