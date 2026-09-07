from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class ConstructIvfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vec_zarr: str = "../vector/laion.zarr/"
    pred_zarr: str = "../predicate/laion_neg.zarr/"
    nlists: int = 1024
    alpha: float = 0.10722672220103233
    sig_dims: int = 32
    sample_docs: int = 500
    sample_queries: int = 500
    metric: str = "ip"
    runs: int = 1
    csv: str = "logs/construct.csv"
    index_path: str | None = None
    print_yaml: bool = False
    kmeans_verbose: bool = False
    bf16_clustering: bool = False

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        metric = value.lower()
        if metric not in {"ip", "l2"}:
            raise ValueError("metric must be 'ip' or 'l2'")
        return metric

    @field_validator("runs")
    @classmethod
    def _validate_runs(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("runs must be positive")
        return value


def _extract_args(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise ValueError("Config payload is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a mapping, got {type(payload).__name__}")
    if "kind" in payload:
        kind = payload["kind"]
        if kind != "construct_ivf":
            raise ValueError(f"Expected kind 'construct_ivf', got {kind!r}")
        if "args" not in payload:
            raise ValueError("Config payload with 'kind' must include 'args'")
    if "args" in payload:
        args = payload["args"]
        if not isinstance(args, dict):
            raise TypeError(f"Config args must be a mapping, got {type(args).__name__}")
        return args
    return payload


def _validate_config(args: dict[str, Any]) -> ConstructIvfConfig:
    return ConstructIvfConfig.model_validate(args)


def load_construct_ivf_config(path: str | Path) -> ConstructIvfConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    args = _extract_args(payload)
    return _validate_config(args)
