#!/usr/bin/env python

from progress import Progress
import zarr
from query_settings import query_slice, validate_split

import torch
from torch.nn.functional import normalize
import numpy as np
import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.test_ivf import TestIvfConfig, load_test_ivf_config
from kmeans import TorchKmeans

import code  # noqa: F401

tracker = Progress()
task = tracker.task

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


@dataclass(frozen=True)
class IvfEvalResult:
    alpha: float
    beta: float
    nprobe: int
    avg_cost: float


class TestIvfArgs(Protocol):
    nlist: int
    alpha: float | None
    prob_dims: int
    pred_zarr: str
    vec_zarr: str
    cache_dir: str | None
    csv: str
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    num_queries: int
    query_start: int
    sample_queries: int
    sample_docs: int
    grid_size: int
    k: int
    ablation: bool
    ablation_queries: int
    ablation_ood: bool
    use_gpu: bool
    predict_pl: bool


@dataclass
class TestIvfState:
    args: TestIvfArgs
    train: np.ndarray
    queries: np.ndarray
    mask: np.ndarray
    mask_T: np.ndarray
    neighbors: np.ndarray
    d_sig_base: np.ndarray
    d_sig_norms: np.ndarray
    v: np.ndarray
    sing: np.ndarray
    train_c: np.ndarray
    idx: Any
    base_dim: int
    centroids_cache: dict[float, Any] = field(default_factory=dict)
    clusters_cache: dict[float, np.ndarray] = field(default_factory=dict)
    cluster_legals_cache: dict[tuple[float, bool], np.ndarray] = field(
        default_factory=dict
    )
    cluster_contribs_cache: dict[tuple[float, int], np.ndarray] = field(
        default_factory=dict
    )


@dataclass
class IvfLoadedData:
    train: np.ndarray
    queries: np.ndarray
    mask: np.ndarray
    mask_T: np.ndarray | None
    neighbors: np.ndarray
    svd_mask: np.ndarray


def get_train_c(train_c, d_sig_base, d_sig_norms, B, cache_dir=None):
    if cache_dir is not None and os.path.exists(cache_dir):
        with task("Load train_c", verbose=False):
            train_c = zarr.load(cache_dir)

    else:
        with task("Build d_sig"):
            norm_t = torch.from_numpy(d_sig_norms)
            base_t = torch.from_numpy(d_sig_base)

            denom = norm_t + 1e-9
            d_sig_t = base_t / denom[:, None] * B

        with task("Build train_c"):
            train_c_t = torch.from_numpy(train_c)
            sig_dims = d_sig_t.shape[1]
            train_c_t[:, -sig_dims:] = d_sig_t
            train_c_t = train_c_t.contiguous().float()
            train_c = train_c_t.numpy()

        if cache_dir is not None:
            with task("Save train_c"):
                zarr.create_array(cache_dir, data=train_c, overwrite=True)

    return train_c


def get_centroids(idx, train_c, n_list, cache_dir=None):
    try:
        if cache_dir is not None and os.path.exists(f"{cache_dir}/idx_centroids.npy"):
            with task("Load centroids"):
                centroids_np = np.load(f"{cache_dir}/idx_centroids.npy")
                if isinstance(idx, TorchKmeans):
                    idx.centroids = torch.from_numpy(
                        np.ascontiguousarray(centroids_np)
                    ).contiguous().float()
                else:
                    idx.centroids = centroids_np
        else:
            raise FileNotFoundError()

    except FileNotFoundError as _:
        with task("Build centroids"):
            num_samples = max(50 * n_list, 100_000)
            if isinstance(idx, TorchKmeans):
                train_t = torch.from_numpy(
                    np.ascontiguousarray(train_c[:num_samples])
                ).contiguous()
                idx.train(train_t)
            else:
                idx.train(train_c[:num_samples])
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                if isinstance(idx, TorchKmeans):
                    centroids_np = (
                        idx.centroids.detach().cpu().numpy().astype("float32", copy=False)
                    )
                    np.save(f"{cache_dir}/idx_centroids.npy", centroids_np)
                else:
                    np.save(f"{cache_dir}/idx_centroids.npy", idx.centroids)

    return idx.centroids


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise argparse.ArgumentTypeError("Boolean value is required")
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _assign_clusters_torch(
    train_c: np.ndarray,
    centroids: np.ndarray,
    *,
    use_gpu: bool,
) -> np.ndarray:
    train_t = torch.from_numpy(train_c).contiguous()
    centroids_t = torch.from_numpy(centroids).contiguous()
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("use_gpu=True but CUDA is not available")
        train_t = train_t.cuda()
        centroids_t = centroids_t.cuda()
    centroids_t = centroids_t.t().contiguous()
    block_size = 128_000
    n = train_t.shape[0]
    if use_gpu:
        clusters_t = torch.empty((n,), device=train_t.device, dtype=torch.int64)
    else:
        clusters = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            scores = train_t[start:end] @ centroids_t
            labels = scores.argmax(dim=-1)
            if use_gpu:
                clusters_t[start:end] = labels
            else:
                clusters[start:end] = labels.numpy()
    if use_gpu:
        return clusters_t.cpu().numpy()
    return clusters


def get_clusters(train_c, idx, use_gpu=False, cache_dir=None):
    try:
        if cache_dir is not None:
            with task("Load clusters"):
                clusters = np.load(f"{cache_dir}/clusters.npy")
        else:
            raise FileNotFoundError()

    except FileNotFoundError as _:
        with task("Build clusters (torch)"):
            assert isinstance(idx, TorchKmeans)
            train_t = torch.from_numpy(np.ascontiguousarray(train_c)).contiguous()
            clusters_t = idx.assign(train_t, use_gpu=use_gpu)
            clusters = clusters_t.cpu().numpy()
        if cache_dir is not None:
            with task("Save clusters"):
                np.save(f"{cache_dir}/clusters.npy", clusters)
    return clusters


def get_cluster_legals(clusters, mask_T, nlist, cache_dir=None, use_gpu=False):
    try:
        if cache_dir is not None:
            with task("Load cluster legals"):
                cluster_legals = np.load(f"{cache_dir}/cluster_legals.npy")
        else:
            raise FileNotFoundError()

    except FileNotFoundError as _:

        with task("Build cluster legals"):
            n_rows, n_cols = mask_T.shape
            chunk_size = int(1e9 // n_cols)
            chunk_size = max(1, min(chunk_size, n_rows))

            clusters_all_t = torch.from_numpy(clusters)
            if use_gpu:
                clusters_all_t = clusters_all_t.cuda()
            cluster_counts_t = torch.bincount(clusters_all_t, minlength=nlist).float()
            cluster_counts_t = torch.clamp(cluster_counts_t, min=1.0)
            cluster_sums_t = torch.zeros(
                (nlist, n_cols),
                dtype=cluster_counts_t.dtype,
                device=cluster_counts_t.device,
            )

            # Chunk over docs to avoid materializing the full mask on GPU.
            with task("Build cluster legals inner loop"):
                for start in range(0, n_rows, chunk_size):
                    end = min(n_rows, start + chunk_size)
                    clusters_t = torch.from_numpy(clusters[start:end])
                    mask_t = torch.from_numpy(mask_T[start:end])
                    if use_gpu:
                        clusters_t = clusters_t.cuda()
                        mask_t = mask_t.cuda()
                    mask_t = mask_t.float()
                    cluster_sums_t.index_add_(0, clusters_t, mask_t)

            cluster_legals_t = cluster_sums_t / cluster_counts_t[:, None]
            cluster_legals = cluster_legals_t.float().cpu().numpy().T


        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(f"{cache_dir}/cluster_legals.npy", cluster_legals)

    return cluster_legals

def get_cluster_contribs(
    clusters: np.ndarray,
    neighbors: np.ndarray,
    nlist: int,
    cache_dir: str | None = None,
    k: int = 10,
):
    try:
        if cache_dir is not None:
            cluster_contribs = np.load(f"{cache_dir}/cluster_contribs.npy")
        else:
            raise FileNotFoundError()
    except FileNotFoundError as _:
        with task("Cluster isin"):
            neighbors_contig = np.ascontiguousarray(neighbors[:, :k])  # (NQ,K)

            cluster_contribs = (
                np.array(
                    [
                        np.bincount(clusters[nbr], minlength=nlist)
                        for nbr in neighbors_contig
                    ]
                )
                / neighbors_contig.shape[1]
            )

            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(f"{cache_dir}/cluster_contribs.npy", cluster_contribs)

    return cluster_contribs


def predict_cluster_legals(mask, v, sing, d_sig_base, clusters, nlist, sample_docs):
    with task("Predict cluster legals"):
        sig_dims = sing.shape[0]
        if sig_dims == 0:
            raise ValueError("sig_dims must be > 0 for prediction")

        with task("Q_sig"):
            sing_safe = np.where(sing > 1e-9, sing, 1e9)
            t_q = v[:sig_dims, :].T @ np.diag(1 / np.sqrt(sing_safe))
            q_trunc = mask[:, :sample_docs].astype(np.float32, copy=False)
            q_sig = q_trunc @ t_q

        with task("Cluster sig"):
            cluster_t = torch.from_numpy(clusters)
            if cluster_t.dtype != torch.int64:
                cluster_t = cluster_t.to(torch.int64)
            base_t = torch.from_numpy(d_sig_base)
            cluster_sums_t = torch.zeros((nlist, sig_dims), dtype=base_t.dtype)
            cluster_sums_t.index_add_(0, cluster_t, base_t)
            cluster_counts_t = torch.bincount(cluster_t, minlength=nlist)
            cluster_sizes_t = cluster_counts_t.to(dtype=base_t.dtype)
            cluster_sizes_t = torch.clamp(cluster_sizes_t, min=1.0)
            cluster_means_t = cluster_sums_t / cluster_sizes_t[:, None]
            cluster_sig_t = cluster_means_t / cluster_sizes_t[:, None]
            empty_mask = cluster_counts_t == 0
            if torch.any(empty_mask):
                cluster_sig_t[empty_mask] = float("nan")
            cluster_sig = cluster_sig_t.numpy()
            floor = 1.0 / float(cluster_sizes_t.max().item())

        with task("P_l final matmul"):
            pred_legals = q_sig @ cluster_sig.T
            pred_legals = np.clip(pred_legals + floor, floor, 1.0)
    return pred_legals


def interesting_sort(arr):
    taken = []
    base = arr.tolist()

    while len(base) > 0:
        if len(taken) == 0:
            choice = base.pop(0)
        else:
            min_distances = [
                np.min(
                    [
                        (0.99 + random.random() / 50) * np.abs(np.log(b) - np.log(t))
                        for t in taken
                    ]
                )
                for b in base
            ]
            c_idx = np.argmax(min_distances)
            choice = base.pop(c_idx)
        taken.append(choice)
    return np.array(taken)


def ax_vals(x_min, x_max, n, scramble=False, include_zero=True):
    rest = np.logspace(np.log10(x_min), np.log10(x_max), n)
    if include_zero:
        zero = np.array([1e-10])
        all = np.concatenate([zero, rest])
    else:
        all = rest
    if scramble:
        all = interesting_sort(all)

    if include_zero:
        all[0] = 0.0
    return all


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default=None)
    parser.add_argument("--nlist", type=int, default=512)
    parser.add_argument("--b_construct", type=float, default=0.1, nargs="+")
    parser.add_argument("--prob_dims", type=int, default=1)
    parser.add_argument("--pred_zarr", type=str, default="../predicate/laion_neg.zarr")
    parser.add_argument("--vec_zarr", type=str, default="../vector/laion.zarr")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default="logs/out.csv")
    parser.add_argument("--y_min", type=float, default=0.0001)
    parser.add_argument("--y_max", type=float, default=1.0)
    parser.add_argument("--x_min", type=float, default=0.01)
    parser.add_argument("--x_max", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--num_queries", "-nq", type=int, default=512)
    parser.add_argument("--query_start", type=int, default=512)
    parser.add_argument("--sample_queries", type=int, default=512)
    parser.add_argument("--sample_docs", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation_queries", type=int, default=1)
    parser.add_argument("--ablation_ood", action="store_true")
    parser.add_argument(
        "--use_gpu",
        "-gpu",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
    )
    parser.add_argument("--no_use_gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--predict_pl", "-ppl", action="store_true")
    return parser


def _cli_override_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    overrides: set[str] = set()
    for token in argv:
        if token == "--":
            break
        opt = token.split("=", 1)[0]
        action = parser._option_string_actions.get(opt)
        if action is not None and action.dest != "config":
            overrides.add(action.dest)
    return overrides


def parse_args() -> TestIvfConfig:
    parser = build_arg_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args_dict = vars(args)
    config_path = args_dict.pop("config", None)
    if config_path:
        base = load_test_ivf_config(config_path).model_dump()
        overrides = _cli_override_dests(parser, argv)
        for dest in overrides:
            base[dest] = args_dict[dest]
        return TestIvfConfig.model_validate(base)
    return TestIvfConfig.model_validate(args_dict)


def _init_csv(path: str) -> None:
    with open(path, "w") as f:
        f.write("nlist,b_construct,prob_dims,b_probe,n_probe,avg_cost\n")


def _resolve_alphas(args: TestIvfArgs, *, include_zero: bool = True) -> np.ndarray:
    if args.alpha is not None:
        alpha = float(args.alpha)
        if include_zero and alpha != 0.0:
            return np.array([0.0, alpha])
        return np.array([alpha])
    return ax_vals(
        args.x_min,
        args.x_max,
        args.grid_size,
        scramble=True,
        include_zero=include_zero,
    )


def _resolve_betas(args: TestIvfArgs) -> np.ndarray:
    return ax_vals(args.y_min, args.y_max, args.grid_size, include_zero=True)


def reset_tracker() -> None:
    global tracker, task
    tracker = Progress()
    task = tracker.task


def load_ivf_arrays(
    args: TestIvfArgs,
    *,
    load_pred_mt: bool = True,
    write_pred_mt: bool = True,
) -> IvfLoadedData:
    """Read disjoint fitting/validation slices without modifying input stores.

    write_pred_mt is retained for callers of the old API; transposes are now
    local to the selected query window and are never written into the dataset.
    """
    validate_split(
        args.query_start, args.num_queries, args.sample_queries, calibration=True
    )
    gv = zarr.open_group(args.vec_zarr, mode="r")
    g = zarr.open_group(args.pred_zarr, mode="r")
    rows = query_slice(
        gv["test"].shape[0],
        args.query_start,
        args.num_queries,
        name="validation vectors",
    )
    query_slice(
        g["neighbors"].shape[0],
        args.query_start,
        args.num_queries,
        name="validation ground truth",
    )
    query_slice(
        g["pred_mask"].shape[0],
        args.query_start,
        args.num_queries,
        name="validation masks",
    )
    fit = query_slice(g["pred_mask"].shape[0], 0, args.sample_queries, name="SVD masks")
    query_slice(g["pred_mask"].shape[1], 0, args.sample_docs, name="SVD documents")
    with task("Load Vectors", verbose=True):
        train = np.asarray(gv["train"][:], dtype=np.float32)
        queries = np.asarray(gv["test"][rows], dtype=np.float32)
    with task("Load Predicate Data", verbose=True):
        mask = np.asarray(g["pred_mask"][rows, :])
        svd_mask = np.asarray(g["pred_mask"][fit, :])
        if load_pred_mt and "pred_mt" in g and g["pred_mt"].shape[1] >= rows.stop:
            mask_T = np.asarray(g["pred_mt"][:, rows])
        else:
            mask_T = np.ascontiguousarray(mask.T)
        neighbors = np.asarray(g["neighbors"][rows, : args.k])
        if mask.shape[1] != len(train) or neighbors.shape != (args.num_queries, args.k):
            raise ValueError(
                "validation masks/ground truth do not align with the vectors"
            )
    return IvfLoadedData(
        train=train,
        queries=queries,
        mask=mask,
        mask_T=mask_T,
        neighbors=neighbors,
        svd_mask=svd_mask,
    )


def prepare_state_from_loaded(
    config: TestIvfArgs, loaded: IvfLoadedData
) -> TestIvfState:
    args = config
    train = loaded.train
    queries = loaded.queries
    mask = loaded.mask
    mask_T = loaded.mask_T
    neighbors = loaded.neighbors
    svd_mask = loaded.svd_mask

    with task("Norm Vectors", verbose=True):
        train_t = torch.from_numpy(train)
        queries_t = torch.from_numpy(queries)

        train = normalize(train_t).numpy()
        queries = normalize(queries_t).numpy()

        queries = queries[: args.num_queries]

    if mask_T is None:
        with task("Transpose Mask", verbose=True):
            mask_t = torch.from_numpy(mask)
            mask_T = mask_t.T.contiguous().numpy()

    if args.ablation:
        with task(
            f"Zeroing queries matching any of the first {args.ablation_queries}",
            verbose=True,
        ):
            qn_mask = mask_T[:, : args.ablation_queries]
            qn_mask_t = torch.from_numpy(qn_mask)
            mask_T_t = torch.from_numpy(mask_T)

            matches_qn = (
                (qn_mask_t[:, None, :] == mask_T_t[:, :, None]).any(dim=-1).all(dim=0)
            ).numpy()

            excl_ids = np.flatnonzero(matches_qn)
            if args.ablation_ood:
                eval_ids = excl_ids
            else:
                eval_ids = np.arange(queries.shape[0])

            with task("Counting unique queries"):
                q_unique = qn_mask_t.unique(dim=1).shape[1]

            with tracker.paused():
                print(f"Excluding {len(excl_ids)} queries across {q_unique} classes which match qn from SVD")
                print(f"Evaluating {len(eval_ids)} queries")
            # Withhold matching predicates from the independent fitting pool.
            held_out = {np.packbits(row).tobytes() for row in mask[excl_ids]}
            keep = [np.packbits(row).tobytes() not in held_out for row in svd_mask]
            svd_mask = svd_mask[keep]
            if not len(svd_mask):
                raise ValueError("ablation excludes every SVD fitting predicate")
            queries = queries[eval_ids, :]
            neighbors = neighbors[eval_ids, :]

            # Zero out for clustering.
            mask_T = mask_T[:, eval_ids]

            mask = np.ascontiguousarray(mask)
            #queries = np.ascontiguousarray(queries)
            #mask_T = np.ascontiguousarray(mask_T)

    with task("SVD", verbose=True):
        sample_docs = args.sample_docs
        sample_quer = args.sample_queries

        sig_dims = args.prob_dims

        svd_mask = svd_mask[:sample_quer, :]
        m_small = svd_mask[:, :sample_docs]
        m_small = m_small.astype(float)

        u, sing, v = np.linalg.svd(m_small, full_matrices=False)
        u = u[:, :sig_dims]
        sing = sing[:sig_dims]

    with task("Construct base document signatures", verbose=True):
        d_trunc = torch.from_numpy(svd_mask).T.float()
        rhs = torch.from_numpy(u @ np.diag(1 / np.sqrt(sing))).float()
        d_sig_base = (d_trunc @ rhs).float()

    with task("Compute signature norms", verbose=True):
        d_sig_base = np.ascontiguousarray(d_sig_base.numpy())
        d_sig_norms = np.linalg.norm(d_sig_base, axis=-1)

    with task("Truncate mask", verbose=True):
        mask = mask[: args.num_queries]

    with task("Train C", verbose=True):
        train_c = np.empty(
            (train.shape[0], train.shape[1] + args.prob_dims), dtype="float32"
        )
        train_c[:, : train.shape[1]] = train
        base_dim = queries.shape[1]
        idx = TorchKmeans(
            base_dim + args.prob_dims,
            args.nlist,
            niter=10,
            verbose=True,
            seed=0,
            spherical=True,
            k_block=1024,
            bf16_clustering=True,
        )

    return TestIvfState(
        args=args,
        train=train,
        queries=queries,
        mask=mask,
        mask_T=mask_T,
        neighbors=neighbors,
        d_sig_base=d_sig_base,
        d_sig_norms=d_sig_norms,
        v=v,
        sing=sing,
        train_c=train_c,
        idx=idx,
        base_dim=base_dim,
    )


def prepare_state(config: TestIvfArgs) -> TestIvfState:
    reset_tracker()
    args = config
    loaded = load_ivf_arrays(args, load_pred_mt=True, write_pred_mt=True)
    return prepare_state_from_loaded(args, loaded)


def evaluate_alphas(
    state: TestIvfState,
    alphas: list[float] | np.ndarray,
    betas: list[float] | np.ndarray,
    *,
    write_csv: bool = True,
    write_summary: bool = True,
    csv_path: str | None = None,
) -> list[IvfEvalResult]:
    args = state.args
    results: list[IvfEvalResult] = []
    csv_target = csv_path or args.csv
    beta = np.asarray(betas, dtype=float)
    centroids_cache = state.centroids_cache
    clusters_cache = state.clusters_cache
    cluster_legals_cache = state.cluster_legals_cache
    cluster_contribs_cache = state.cluster_contribs_cache

    for B in tracker.progress(
        f"Running {len(alphas)} alpha settings", alphas, verbose=True
    ):
        with task("Setup IVF"):
            csv_prefix = f"{args.cache_dir}/ivf_amplitude_{args.nlist},{B:0.5f},{args.prob_dims}_fit0-{args.sample_queries}_docs{args.sample_docs}_q{args.query_start}-{args.num_queries}_ab{args.ablation}-{args.ablation_queries}-{args.ablation_ood}"
            if args.cache_dir is None:
                save_dir = None
                train_c_dir = None
            else:
                save_dir = csv_prefix
                train_c_dir = f"{csv_prefix}_train_c"

        alpha_key = round(float(B), 12)
        cached_centroids = centroids_cache.get(alpha_key)
        cached_clusters = clusters_cache.get(alpha_key)

        if cached_centroids is not None and cached_clusters is not None:
            state.idx.centroids = cached_centroids
            clusters = cached_clusters
        else:
            train_c = get_train_c(
                state.train_c, state.d_sig_base, state.d_sig_norms, B, train_c_dir
            )
            state.idx.centroids = get_centroids(
                state.idx, train_c, args.nlist, cache_dir=save_dir
            )
            clusters = get_clusters(
                train_c, state.idx, use_gpu=args.use_gpu, cache_dir=save_dir
            )
            centroids_to_cache = state.idx.centroids
            if torch.is_tensor(centroids_to_cache):
                centroids_to_cache = centroids_to_cache.detach().cpu().contiguous()
            centroids_cache[alpha_key] = centroids_to_cache
            clusters_cache[alpha_key] = clusters

        legals_key = (alpha_key, bool(args.predict_pl))
        cached_legals = cluster_legals_cache.get(legals_key)
        if cached_legals is not None:
            cluster_legals = cached_legals
        else:
            if args.predict_pl:
                if args.ablation:
                    raise Exception("args.predict_pl code path not updated for ablation")
                pred_cache = None
                if save_dir is not None:
                    pred_cache = f"{save_dir}/cluster_legals_pred.npy"
                if pred_cache is not None and os.path.exists(pred_cache):
                    cluster_legals = np.load(pred_cache)
                else:
                    cluster_legals = predict_cluster_legals(
                        state.mask,
                        state.v,
                        state.sing,
                        state.d_sig_base,
                        clusters,
                        args.nlist,
                        args.sample_docs,
                    )
                    if pred_cache is not None and save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(pred_cache, cluster_legals)
            else:
                cluster_legals = get_cluster_legals(
                    clusters, state.mask_T, args.nlist, cache_dir=save_dir
                )
            cluster_legals_cache[legals_key] = cluster_legals

        contribs_key = (alpha_key, int(args.k))
        cached_contribs = cluster_contribs_cache.get(contribs_key)
        if cached_contribs is not None:
            cluster_contribs = cached_contribs
        else:
            cluster_contribs = get_cluster_contribs(
                clusters, state.neighbors, args.nlist, save_dir, k=args.k
            )
            cluster_contribs_cache[contribs_key] = cluster_contribs
        cluster_contribs_t = torch.from_numpy(cluster_contribs)

        with task("Cluster Sizes"):
            cluster_sizes = np.bincount(clusters, minlength=args.nlist)
        with task("Base Scores"):
            centroids = state.idx.centroids
            if torch.is_tensor(centroids):
                centroids = centroids.detach().cpu().numpy()
            c_l = centroids[:, : state.base_dim]
            c_l_normed = c_l / np.linalg.norm(c_l, axis=1, keepdims=True)
            base = state.queries @ c_l_normed.T

        with task("Log Prob Setup"):
            eligible_clusters = cluster_legals > 0
            log_pl = np.log(np.where(eligible_clusters, cluster_legals, 1.0))
            beta_t = torch.from_numpy(beta).to(torch.float32)
            log_pl_t = torch.from_numpy(log_pl).to(torch.float32)
            base_t = torch.from_numpy(base).to(torch.float32)

        with task("Adj Scores"):
            if args.use_gpu:
                beta_t = beta_t.cuda()
                log_pl_t = log_pl_t.cuda()
                base_t = base_t.cuda()

        with task("Reindex Sizes"):
            cluster_size_t = torch.from_numpy(cluster_sizes)
            if args.use_gpu:
                cluster_size_t = cluster_size_t.cuda()
            else:
                cluster_size_t = cluster_size_t.cpu()

        with task("Sorted Margins"):
            if args.use_gpu:
                cluster_contribs_t = cluster_contribs_t.cuda()
            else:
                cluster_contribs_t = cluster_contribs_t.cpu()

        with task("Sort/Accumulate"):
            beta_count = beta_t.shape[0]
            query_count = log_pl_t.shape[0]
            nlist = log_pl_t.shape[1]
            target_elems = 100_000_000
            q_chunk = max(
                1, min(query_count, target_elems // max(1, beta_count * nlist))
            )

            sum_csr_t = torch.zeros(
                (beta_count, nlist),
                device=cluster_size_t.device,
                dtype=torch.float32,
            )
            sum_recall_t = torch.zeros(
                (beta_count, nlist),
                device=cluster_contribs_t.device,
                dtype=cluster_contribs_t.dtype,
            )

            for q_start in range(0, query_count, q_chunk):
                q_end = min(query_count, q_start + q_chunk)
                log_pl_chunk = log_pl_t[q_start:q_end]
                base_chunk = base_t[q_start:q_end]

                cluster_logprobs = torch.einsum("p,qc->pqc", beta_t, log_pl_chunk)
                eligible_chunk = torch.as_tensor(
                    eligible_clusters[q_start:q_end], device=cluster_logprobs.device
                )
                # Avoid 0 * -inf for beta=0, then exclude empty clusters.
                if B != 0.0:
                    cluster_logprobs.masked_fill_(
                        ~eligible_chunk.unsqueeze(0), -torch.inf
                    )
                else:
                    cluster_logprobs.masked_fill_(
                        (~eligible_chunk.unsqueeze(0)) & (beta_t[:, None, None] != 0),
                        -torch.inf,
                    )
                adj_dis_chunk = -(cluster_logprobs + base_chunk)

                if args.use_gpu:
                    if adj_dis_chunk.device.type != "cuda":
                        adj_dis_chunk = adj_dis_chunk.cuda()

                # Looping to prevent GPU OOM.
                rankings_chunk = torch.empty_like(adj_dis_chunk, dtype=torch.int32)
                for b, row in enumerate(adj_dis_chunk):
                    rankings_chunk[b] = torch.argsort(row, dim=-1)

                rankings_size = rankings_chunk
                if rankings_size.device != cluster_size_t.device:
                    rankings_size = rankings_size.to(cluster_size_t.device)
                csr_chunk = cluster_size_t[rankings_size]
                if B != 0.0 or np.any(beta != 0):
                    finite_scores = torch.isfinite(-adj_dis_chunk)
                    scan = torch.gather(finite_scores, -1, rankings_chunk.long())
                    csr_chunk = csr_chunk * scan.to(csr_chunk.device)
                sum_csr_t += csr_chunk.cumsum(dim=-1).float().sum(dim=1)

                rankings_margin = rankings_chunk
                if rankings_margin.device != cluster_contribs_t.device:
                    rankings_margin = rankings_margin.to(cluster_contribs_t.device)
                cc_chunk = cluster_contribs_t[q_start:q_end]
                cc_broadcast = cc_chunk.unsqueeze(0).expand(
                    rankings_margin.shape[0], -1, -1
                )
                sorted_margins_chunk = torch.gather(
                    cc_broadcast, index=rankings_margin, dim=-1
                )
                sum_recall_t += sorted_margins_chunk.cumsum(dim=-1).sum(dim=1)

            all_costs = (sum_csr_t / query_count).cpu().numpy()
            all_cumulative_recall: np.ndarray = (
                (sum_recall_t / query_count).cpu().numpy()
            )

        with task("Passes 95"):
            all_passes_95: np.ndarray = all_cumulative_recall >= 0.95

        with task("Nprobe 95"):
            assert len(all_passes_95.shape) > 1, "Must hold to safely disable pyright"
            all_nprobe_95 = [
                np.min(np.flatnonzero(passes_95))
                for passes_95 in all_passes_95  # type: ignore[reportAttributeAccessIssue]
            ]

        with task("Costs 95"):
            all_costs_95 = [
                costs[nprobe] for costs, nprobe in zip(all_costs, all_nprobe_95)
            ]

        with task("Log Results"):
            for beta_i, nprobe, cost in zip(beta, all_nprobe_95, all_costs_95):
                results.append(
                    IvfEvalResult(
                        alpha=float(B),
                        beta=float(beta_i),
                        nprobe=int(nprobe + 1),
                        avg_cost=float(cost),
                    )
                )
                if write_csv:
                    with open(csv_target, "a") as f:
                        f.write(f"{csv_prefix},{beta_i},{nprobe + 1},{cost:.1f}\n")

            if write_summary:
                analytics_dir = "analytics"
                os.makedirs(analytics_dir, exist_ok=True)
                csv_name = os.path.basename(csv_target)
                summary_path = os.path.join(analytics_dir, f"{csv_name}.log")
                with open(summary_path, "w") as summary_file:
                    tracker.print_summary(file=summary_file)

    return results


def run(config: TestIvfConfig) -> None:
    state = prepare_state(config)
    _init_csv(state.args.csv)

    with task("Setup loop", verbose=True):
        alphas = _resolve_alphas(state.args, include_zero=True)
        betas = _resolve_betas(state.args)

    evaluate_alphas(
        state,
        alphas,
        betas,
        write_csv=True,
        write_summary=True,
        csv_path=state.args.csv,
    )


if __name__ == "__main__":
    run(parse_args())
