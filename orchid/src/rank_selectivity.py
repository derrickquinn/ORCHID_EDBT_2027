from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class RankSelectivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nlist: int = 512
    prob_dims: int = 64
    pred_zarr: str | None = None
    vec_zarr: str | None = None
    npy_base: str = "logs"
    npy: str | None = None
    num_queries: int = 1000
    sample_queries: int = 500
    sample_docs: int = 500
    use_gpu: bool = True


def _extract_args(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise ValueError("Config payload is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a mapping, got {type(payload).__name__}")
    if "kind" in payload:
        kind = payload["kind"]
        if kind != "rank_sel":
            raise ValueError(f"Expected kind 'rank_sel', got {kind!r}")
        if "args" not in payload:
            raise ValueError("Config payload with 'kind' must include 'args'")
    if "args" in payload:
        args = payload["args"]
        if not isinstance(args, dict):
            raise TypeError(f"Config args must be a mapping, got {type(args).__name__}")
        return args
    return payload


def _validate_config(args: dict[str, Any]) -> RankSelectivityConfig:
    return RankSelectivityConfig.model_validate(args)


def load_rank_selectivity_config(path: str | Path) -> RankSelectivityConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    args = _extract_args(payload)
    return _validate_config(args)
