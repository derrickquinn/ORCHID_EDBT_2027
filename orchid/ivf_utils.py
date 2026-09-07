from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from torch import from_numpy as torch_from_numpy
import zarr
from query_settings import query_slice


@contextmanager
def _null_task(*_args, **_kwargs):
    yield


def _task_cm(task):
    return task if task is not None else _null_task


def _is_zarr_group(path: str | Path) -> bool:
    meta = Path(path) / "zarr.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text())
        except json.JSONDecodeError:
            return False
        return payload.get("node_type") == "group"
    return (Path(path) / ".zgroup").exists()


def load_train_c(path: str | Path, *, task=None) -> np.ndarray | None:
    try:
        return zarr.load(path)
    except (FileNotFoundError, NotImplementedError, OSError, ValueError):
        if task is not None and _is_zarr_group(path):
            with _task_cm(task)("Invalid train_c cache (group), rebuilding", verbose=True):
                pass
        return None


def load_svd_mask(
    pred_zarr: str | Path,
    *,
    sample_queries: int,
    task=None,
) -> np.ndarray:
    if sample_queries <= 0:
        raise ValueError("sample_queries must be positive")
    mask = zarr.open_group(pred_zarr, mode="r")["pred_mask"]
    rows = query_slice(mask.shape[0], 0, sample_queries, name="SVD predicate mask")
    with _task_cm(task)("Load SVD mask", verbose=True):
        return np.array(mask[rows, :])


def build_train_c(
    docs: np.ndarray,
    svd_mask: np.ndarray | None,
    *,
    alpha: float,
    sig_dims: int,
    sample_docs: int,
    task=None,
) -> np.ndarray:
    if sig_dims < 0:
        raise ValueError("sig_dims must be non-negative")
    if sig_dims == 0 or alpha == 0.0:
        train_c = np.zeros((docs.shape[0], docs.shape[1] + sig_dims), dtype="float32")
        train_c[:, : docs.shape[1]] = docs
        return train_c

    if svd_mask is None:
        raise ValueError("svd_mask is required when alpha and sig_dims are non-zero")
    if sample_docs <= 0:
        raise ValueError("sample_docs must be positive")
    query_slice(svd_mask.shape[1], 0, sample_docs, name="SVD documents")

    with _task_cm(task)("SVD", verbose=True):
        m_small = svd_mask[:, :sample_docs].astype(float)
        u, sing, _ = np.linalg.svd(m_small, full_matrices=False)
        u = u[:, :sig_dims]
        sing = sing[:sig_dims]

    with _task_cm(task)("Construct base document signatures", verbose=True):
        d_trunc = torch_from_numpy(svd_mask).T.float()
        rhs = torch_from_numpy(u @ np.diag(1 / np.sqrt(sing))).float()
        d_sig_base = (d_trunc @ rhs).float()

    with _task_cm(task)("Compute signature norms", verbose=True):
        d_sig_base = np.ascontiguousarray(d_sig_base.numpy())
        d_sig_norms = np.linalg.norm(d_sig_base, axis=-1)

    with _task_cm(task)("Build train_c", verbose=True):
        train_c = np.empty(
            (docs.shape[0], docs.shape[1] + sig_dims), dtype="float32"
        )
        train_c[:, : docs.shape[1]] = docs
        norm_t = torch_from_numpy(d_sig_norms)
        base_t = torch_from_numpy(d_sig_base)
        d_sig_t = base_t / (norm_t + 1e-9)[:, None] * alpha
        train_c_t = torch_from_numpy(train_c)
        train_c_t[:, -sig_dims:] = d_sig_t
        return train_c_t.contiguous().float().numpy()


def load_or_build_train_c(
    cache_dir: str | Path | None,
    docs: np.ndarray,
    svd_mask: np.ndarray | None,
    *,
    alpha: float,
    sig_dims: int,
    sample_docs: int,
    task=None,
) -> np.ndarray:
    cache_dir = None if cache_dir is None else Path(cache_dir)
    invalid_cache = False
    if cache_dir is not None:
        train_c = load_train_c(cache_dir, task=task)
        if train_c is not None:
            return train_c
        invalid_cache = _is_zarr_group(cache_dir)

    train_c = build_train_c(
        docs, svd_mask, alpha=alpha, sig_dims=sig_dims, sample_docs=sample_docs, task=task
    )

    if cache_dir is not None and not invalid_cache:
        with _task_cm(task)("Save train_c", verbose=True):
            zarr.create_array(cache_dir, data=train_c, overwrite=True)
    return train_c


def load_or_build_centroids(
    cache_dir: str | Path | None,
    train_c: np.ndarray,
    *,
    nlist: int,
    use_gpu: bool,
    task=None,
) -> np.ndarray:
    cache_dir = None if cache_dir is None else Path(cache_dir)
    if cache_dir is not None:
        path = cache_dir / "idx_centroids.npy"
        if path.exists():
            with _task_cm(task)("Load centroids", verbose=True):
                return np.load(path)

    with _task_cm(task)("Build centroids", verbose=True):
        idx = faiss.Kmeans(
            train_c.shape[1],
            nlist,
            spherical=True,
            gpu=use_gpu,
            niter=10,
            verbose=False,
            seed=0,
        )
        num_samples = max(100 * nlist, 100_000)
        idx.train(train_c[:num_samples])
        centroids = idx.centroids

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_dir / "idx_centroids.npy", centroids)
    return centroids


def load_or_build_clusters(
    cache_dir: str | Path | None,
    train_c: np.ndarray,
    centroids: np.ndarray,
    *,
    use_gpu: bool,
    task=None,
) -> np.ndarray:
    cache_dir = None if cache_dir is None else Path(cache_dir)
    if cache_dir is not None:
        path = cache_dir / "clusters.npy"
        if path.exists():
            with _task_cm(task)("Load clusters", verbose=True):
                return np.load(path)

    if use_gpu:
        with _task_cm(task)("Build clusters GPU", verbose=True):
            res = faiss.StandardGpuResources()
            centroids_idx = faiss.GpuIndexFlatIP(res, centroids.shape[-1])
            centroids_idx.add(centroids)
            clusters = centroids_idx.search(train_c, 1)[1][:, 0]
    else:
        with _task_cm(task)("Build clusters", verbose=True):
            centroids_t = torch_from_numpy(centroids).T.contiguous()
            train_t = torch_from_numpy(train_c).contiguous()
            scores = train_t @ centroids_t
            clusters = scores.argmax(dim=-1).numpy()

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_dir / "clusters.npy", clusters)
    return clusters
