from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    vec_zarr: str
    pred_zarr: str
    nlists: int


def _extract_args(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise ValueError("Config payload is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a mapping, got {type(payload).__name__}")
    if "kind" in payload:
        kind = payload["kind"]
        if kind != "dataset":
            raise ValueError(f"Expected kind 'dataset', got {kind!r}")
        if "args" not in payload:
            raise ValueError("Config payload with 'kind' must include 'args'")
    if "args" in payload:
        args = payload["args"]
        if not isinstance(args, dict):
            raise TypeError(f"Config args must be a mapping, got {type(args).__name__}")
        return args
    return payload


def _validate_config(args: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig.model_validate(args)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    args = _extract_args(payload)
    return _validate_config(args)
