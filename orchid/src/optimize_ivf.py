from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator
from query_settings import validate_split


class OptimizeIvfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Shared IVF evaluation args (mirrors test_ivf defaults where relevant).
    nlist: int
    prob_dims: int
    pred_zarr: str
    vec_zarr: str
    cache_dir: str | None = None
    csv: str = "logs/out.csv"
    y_min: float = 0.0001
    y_max: float = 1.0
    x_min: float = 0.01
    x_max: float = 1.0
    num_queries: int = 512
    query_start: int = 512
    sample_queries: int = 512
    sample_docs: int = 512
    k: int = 10
    ablation: bool = False
    ablation_queries: int = 1
    ablation_ood: bool = False
    use_gpu: bool = True
    predict_pl: bool = False

    # Optimization args.
    alpha0: float | None = None
    beta0: float | None = None
    step_log10: float = 1.5
    min_step_log10: float = 0.10
    max_iters: int = 50
    tol: float = 1e-2
    step_decay: float = 0.5
    patience: int = 1

    @model_validator(mode="after")
    def _validate_queries(self):
        validate_split(
            self.query_start, self.num_queries, self.sample_queries, calibration=True
        )
        if self.sample_docs <= 0:
            raise ValueError("sample_docs must be positive")
        return self


def _extract_args(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise ValueError("Config payload is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a mapping, got {type(payload).__name__}")
    if "kind" in payload:
        kind = payload["kind"]
        if kind != "optimize_ivf":
            raise ValueError(f"Expected kind 'optimize_ivf', got {kind!r}")
        if "args" not in payload:
            raise ValueError("Config payload with 'kind' must include 'args'")
    if "args" in payload:
        args = payload["args"]
        if not isinstance(args, dict):
            raise TypeError(f"Config args must be a mapping, got {type(args).__name__}")
        return args
    return payload


def _validate_config(args: dict[str, Any]) -> OptimizeIvfConfig:
    return OptimizeIvfConfig.model_validate(args)


def load_optimize_ivf_config(path: str | Path) -> OptimizeIvfConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    args = _extract_args(payload)
    return _validate_config(args)
