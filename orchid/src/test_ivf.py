from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from query_settings import validate_split


class TestIvfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nlist: int = 512
    b_construct: list[float] = Field(default_factory=lambda: [0.1])
    prob_dims: int = 1
    pred_zarr: str = "../predicate/laion_neg.zarr"
    vec_zarr: str = "../vector/laion.zarr"
    cache_dir: str | None = None
    csv: str = "logs/out.csv"
    y_min: float = 0.0001
    y_max: float = 1.0
    x_min: float = 0.01
    x_max: float = 1.0
    alpha: float | None = None
    num_queries: int = 512
    query_start: int = 512
    sample_queries: int = 512
    sample_docs: int = 512
    grid_size: int = 100
    k: int = 10
    ablation: bool = False
    ablation_queries: int = 1
    ablation_ood: bool = False

    use_gpu: bool = True
    predict_pl: bool = False

    @model_validator(mode="after")
    def _validate_queries(self):
        validate_split(
            self.query_start, self.num_queries, self.sample_queries, calibration=True
        )
        if self.sample_docs <= 0:
            raise ValueError("sample_docs must be positive")
        return self

    @field_validator("b_construct", mode="before")
    @classmethod
    def _coerce_b_construct(cls, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return [float(value)]
        return value


def _extract_args(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise ValueError("Config payload is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a mapping, got {type(payload).__name__}")
    if "kind" in payload:
        kind = payload["kind"]
        if kind != "test_ivf":
            raise ValueError(f"Expected kind 'test_ivf', got {kind!r}")
        if "args" not in payload:
            raise ValueError("Config payload with 'kind' must include 'args'")
    if "args" in payload:
        args = payload["args"]
        if not isinstance(args, dict):
            raise TypeError(f"Config args must be a mapping, got {type(args).__name__}")
        return args
    return payload


def _validate_config(args: dict[str, Any]) -> TestIvfConfig:
    return TestIvfConfig.model_validate(args)


def load_test_ivf_config(path: str | Path) -> TestIvfConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    args = _extract_args(payload)
    return _validate_config(args)
