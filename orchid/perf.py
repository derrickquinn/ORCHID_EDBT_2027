#!/usr/bin/env python
# ruff: noqa: E402
import argparse
from functools import cache
import os
from pathlib import Path
import sys
from time import perf_counter

_INITIAL_CPU_AFFINITY = (
    os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
)

import faiss
import numpy as np
import torch
import yaml
import zarr
from query_settings import query_slice, batch_size_for, batch_slices
from predicate_routing import predicate_log_scores, exclude_empty_probes

from ivf_utils import (
    load_or_build_centroids,
    load_or_build_clusters,
    load_or_build_train_c,
    load_svd_mask,
)
from predicate_index import materialize_results
from predicate_workload import load_or_build_predicate_workload
from perf_results import (
    CLUSTER_TIMING_STAGES,
    SweepResult,
    recall_at_k,
    require_iteration_agreement,
    require_measurement_convergence,
    run_measured_iterations,
    summarize_timings,
    write_timing_results,
    write_sweep_results,
)
from src.build_ivf_config import BuildIvfConfig, load_build_ivf_config
from src.dataset_config import DatasetConfig, load_dataset_config
from progress import Progress
from utils_cpp import (
    cluster_legals,
    prepare_cluster_order,
    topk_small,
)

torch.set_num_threads(32)
torch.set_num_interop_threads(1)
if _INITIAL_CPU_AFFINITY is not None:
    os.sched_setaffinity(0, _INITIAL_CPU_AFFINITY)
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

tracker = Progress()
task = tracker.task

def load_docs(vec_zarr):
    gv = zarr.open(vec_zarr, mode="r")
    return gv["train"][:]


def load_quer(vec_zarr, num_queries=10000, query_start=1024):
    gv = zarr.open(vec_zarr, mode="r")
    return gv["test"][
        query_slice(gv["test"].shape[0], query_start, num_queries, name="test vectors")
    ]


def load_gt(pred_zarr, k=10, nq=10000, query_start=1024):
    gv = zarr.open(pred_zarr, mode="r")
    return gv["neighbors"][
        query_slice(gv["neighbors"].shape[0], query_start, nq, name="ground truth"), :k
    ]


def load_mask(pred_zarr):
    gv = zarr.open(pred_zarr, mode="r")
    return gv["pred_mask"]


def load_mask_T(pred_zarr):
    gv = zarr.open(pred_zarr, mode="r")
    return gv["pred_mt"][:]


@cache
def search_elig(queries, idx, *args):
    distances, indices = idx.search(queries, *args)
    return indices, distances


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--dset_cfg", "--dataset_config", dest="dset_cfg", type=str, default=None
    )
    parser.add_argument("--num_queries", "-nq", type=int, default=10000)
    parser.add_argument("--beta", "-b", type=float, default=0.00)
    parser.add_argument("--iterations", "-i", type=int, default=10)
    parser.add_argument("--min-iterations", type=int, default=None)
    parser.add_argument("--qps-relative-ci-half-width", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--low_nprobe", "-ln", type=int, default=1)
    parser.add_argument("--high_nprobe", "-hn", type=int, default=1024)
    parser.add_argument("--step_nprobe", "-sn", type=int, default=10)
    parser.add_argument("--vec_zarr", "-v", type=str, default=None)
    parser.add_argument(
        "--pred_zarr", "-p", type=str, default=None
    )
    parser.add_argument("--cache_base", "-cb", type=str, default="cached")
    parser.add_argument("--nlists", "-nl", type=int, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=None)
    parser.add_argument("--query_start", type=int, default=1024)
    parser.add_argument("--sample_queries", type=int, default=512)
    parser.add_argument("--sample_docs", type=int, default=512)
    parser.add_argument("--alpha", "-a", type=float, default=0.10722672220103233)
    parser.add_argument("--k", "--k_final", "-k", dest="k", type=int, default=10)
    parser.add_argument(
        "--gpu",
        "-g",
        action="store_true",
        help="Use GPU for vector search",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device ordinal used for vector search",
    )
    parser.add_argument(
        "--gpu_build",
        action="store_true",
        help="Use GPU for centroids/clusters construction",
    )
    parser.add_argument("--log_base", "-lb", type=str, default="logs")
    parser.add_argument("--sig_dims", "-sd", type=int, default=64)
    parser.add_argument("--exp_name", "--log", "-l", dest="exp_name", type=str, default="")
    parser.add_argument("--metric", "-m", type=str, default="ip")
    parser.add_argument("--simd", "-s", action="store_true")
    parser.add_argument("--val_flat", "-vf", action="store_true")
    parser.add_argument(
        "--validate_predicates",
        action="store_true",
        help="Compare generated predicate masks with the legacy oracle offline",
    )
    return parser


def _cli_override_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    overrides: set[str] = set()
    for token in argv:
        if token == "--":
            break
        opt = token.split("=", 1)[0]
        action = parser._option_string_actions.get(opt)
        if action is not None and action.dest not in {
            "config",
            "dset_cfg",
        }:
            overrides.add(action.dest)
    return overrides


def _clone_index_to_gpu(index, device: int):
    if device < 0 or device >= faiss.get_num_gpus():
        raise ValueError(
            f"GPU device {device} is unavailable; found {faiss.get_num_gpus()} GPUs"
        )

    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.use_cuvs = False
    gpu_index = faiss.index_cpu_to_gpu(resources, device, index, options)
    return resources, gpu_index


def parse_args() -> tuple[BuildIvfConfig, DatasetConfig]:
    parser = build_arg_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args_dict = vars(args)
    config_path = args_dict.pop("config", None)
    dset_cfg = args_dict.pop("dset_cfg", None)
    dataset_fields = ("vec_zarr", "pred_zarr", "nlists")
    dataset_overrides = {field: args_dict.pop(field) for field in dataset_fields}
    if dset_cfg:
        dataset_payload = load_dataset_config(dset_cfg).model_dump()
    else:
        dataset_payload = {}

    dataset_name = dataset_payload.get("name")

    if config_path:
        base = load_build_ivf_config(
            config_path, dataset_name=dataset_name
        ).model_dump()
    else:
        base = args_dict.copy()

    overrides = _cli_override_dests(parser, argv)
    for dest in overrides:
        if dest in dataset_fields:
            dataset_payload[dest] = dataset_overrides[dest]
            continue
        base[dest] = args_dict[dest]

    missing = [
        field
        for field in dataset_fields
        if field not in dataset_payload or dataset_payload[field] is None
    ]
    if missing:
        raise ValueError(
            "Missing dataset config fields: "
            f"{', '.join(missing)}. Provide --dset_cfg or explicit overrides."
        )

    if not dataset_payload.get("name"):
        dataset_payload["name"] = "cli"

    return (
        BuildIvfConfig.model_validate(base),
        DatasetConfig.model_validate(dataset_payload),
    )


def main():
    args, dataset = parse_args()
    timing_stages = CLUSTER_TIMING_STAGES
    debug = bool(os.getenv("PERF_DEBUG"))

    use_gpu = args.gpu
    use_gpu_build = args.gpu_build
    flat = args.val_flat

    iters = args.iterations
    warmup = args.warmup
    if use_gpu and flat:
        raise ValueError("GPU pre-filter/flat search is not supported")

    B = batch_size_for(use_gpu, args.batch_size)
    args.batch_size = B
    if iters <= 0:
        raise ValueError("iterations must be positive")
    adaptive = (
        args.min_iterations is not None
        or args.qps_relative_ci_half_width is not None
    )
    if adaptive:
        if (
            args.min_iterations is None
            or args.qps_relative_ci_half_width is None
        ):
            raise ValueError(
                "adaptive measurement requires min_iterations and "
                "qps_relative_ci_half_width"
            )
        if not 3 <= args.min_iterations <= iters:
            raise ValueError("min_iterations must be between 3 and iterations")
        if args.qps_relative_ci_half_width <= 0:
            raise ValueError("qps_relative_ci_half_width must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    beta = args.beta
    vec_zarr = dataset.vec_zarr
    pred_zarr = dataset.pred_zarr
    cache_dir = Path(args.cache_base) / dataset.name / args.exp_name
    log_dataset_name = dataset.name + ("-g" if use_gpu else "")
    log_dir = Path(args.log_base) / log_dataset_name
    nlists = dataset.nlists
    merged_args = args.model_dump()
    merged_args.update(dataset.model_dump())
    resolved_config = {
        "kind": "build_ivf_resolved",
        "args": merged_args,
        "cache_dir": str(cache_dir),
        "log_dir": str(log_dir),
    }
    print(yaml.safe_dump(resolved_config, sort_keys=False))

    k = args.k
    rk = args.k
    num_queries = args.num_queries
    n = -1

    low_nprobe = args.low_nprobe
    high_nprobe = args.high_nprobe if not use_gpu else min(2048, args.high_nprobe)

    nprobes = np.logspace(
        np.log2(low_nprobe), np.log2(high_nprobe), num=args.step_nprobe, base=2
    )
    nprobes = np.round(nprobes).astype(int)
    nprobes = np.unique(nprobes).tolist()

    # Validate small query/metadata reads before loading document vectors.
    queries = load_quer(vec_zarr, num_queries, args.query_start).astype(
        "float32", copy=False
    )
    gt_all = load_gt(pred_zarr, k=k, nq=num_queries, query_start=args.query_start)
    if gt_all.shape != (num_queries, k):
        raise ValueError("ground truth does not contain the requested k")
    with task("Load vectors", verbose=True):
        clustering_stem = (
            cache_dir
            / f"ivf_amplitude_{nlists},{args.alpha:.5f},{args.sig_dims}_fit0-{args.sample_queries}_docs{args.sample_docs}"
        )
        train_c_stem = (
            cache_dir
            / f"train_c_amplitude_{args.alpha:.5f},{args.sig_dims}_fit0-{args.sample_queries}_docs{args.sample_docs}"
        )

        docs = load_docs(vec_zarr).astype("float32", copy=False)

    with task("Normalize vectors", verbose=True):
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)

    with task("Prepare clustering inputs", verbose=True):
        svd_mask = None
        if args.sig_dims > 0 and args.alpha != 0.0:
            svd_mask = load_svd_mask(
                pred_zarr, sample_queries=args.sample_queries, task=task
            )
        train_c = load_or_build_train_c(
            train_c_stem,
            docs,
            svd_mask,
            alpha=args.alpha,
            sig_dims=args.sig_dims,
            sample_docs=args.sample_docs,
            task=task,
        )
        centroids = load_or_build_centroids(
            clustering_stem,
            train_c,
            nlist=nlists,
            use_gpu=use_gpu_build,
            task=task,
        )
        clusters = load_or_build_clusters(
            clustering_stem, train_c, centroids, use_gpu=use_gpu_build, task=task
        ).astype("int64")

        d = queries.shape[1]

    with task("Process centroids", verbose=True):
        n_lists, _ = centroids.shape
        centroids = centroids[:, :d].astype("float32")
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        n = docs.shape[0]
        d = queries.shape[1]

    with task("Prepare cluster-major internal IDs", verbose=True):
        clusters_i32 = np.ascontiguousarray(clusters, dtype=np.int32)
        internal_to_external, offsets, counts = prepare_cluster_order(
            clusters_i32, n_lists
        )
        internal_to_external = np.ascontiguousarray(internal_to_external)
        offsets = np.ascontiguousarray(offsets)
        counts = np.ascontiguousarray(counts)
        external_to_internal = np.empty(n, dtype=np.int64)
        external_to_internal[internal_to_external] = np.arange(
            n, dtype=np.int64
        )

    with task("Load or build predicate index", verbose=True):
        predicate_cache = Path(f"{clustering_stem}.predicates")
        predicate_workload = load_or_build_predicate_workload(
            dataset,
            predicate_cache,
            external_to_internal,
            n,
            num_queries,
            query_start=args.query_start,
        )

    with task("Build IVF", verbose=True):
        if not args.val_flat:
            metric = (
                faiss.METRIC_L2 if args.metric == "L2" else faiss.METRIC_INNER_PRODUCT
            )
            if_str = f"IVF{n_lists},Flat"
            index = faiss.index_factory(d, if_str, metric)
            index.quantizer.add(centroids)
            index.is_trained = True
            index.train_encoder(n, faiss.swig_ptr(docs), faiss.swig_ptr(clusters))
            index.add_core(
                n,
                faiss.swig_ptr(docs),
                faiss.swig_ptr(external_to_internal),
                faiss.swig_ptr(clusters),
            )
            index.parallel_mode = 3

            # Check here due to index compatibility
            if args.simd:
                index.use_simd = True
            # os.makedirs("IVF_SIZE", exist_ok=True)
            # faiss.write_index(index, f"IVF_SIZE/{log_dir}-{if_str}.faiss")
        else:
            index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

            docs_internal = np.ascontiguousarray(docs[internal_to_external])
            index.add(docs_internal)

    if use_gpu:
        with task("Upload to GPU", verbose=True):
            _gpu_resources, index = _clone_index_to_gpu(index, args.gpu_device)

    with task("Compute rankings", verbose=True):
        queries = np.ascontiguousarray(queries)
        centroids = np.ascontiguousarray(centroids)
        queries_t = torch.from_numpy(queries)
        centroids_t = torch.from_numpy(centroids[:, :d])
    with task("Prepare predicate evaluation", verbose=True):
        bytes_per_query = (n + 7) // 8
        predicate_bitmap = np.empty((B, bytes_per_query), dtype=np.uint8)

    if args.validate_predicates:
        with task("Validate predicate index", verbose=True):
            mask_oracle = load_mask(pred_zarr)
            for start in range(0, num_queries, B):
                stop = min(start + B, num_queries)
                validation_results = predicate_workload.evaluate_batch(
                    start, stop, threads=0
                )
                actual = materialize_results(
                    validation_results, n, predicate_bitmap[: stop - start], threads=0
                )
                expected_external = np.asarray(
                    mask_oracle[args.query_start + start : args.query_start + stop, :n]
                )
                expected = np.packbits(
                    expected_external[:, internal_to_external],
                    axis=1,
                    bitorder="little",
                )
                if not np.array_equal(actual, expected):
                    mismatched = np.flatnonzero(np.any(actual != expected, axis=1))
                    query = start + int(mismatched[0])
                    raise AssertionError(
                        f"predicate index differs from legacy mask oracle at query {query}"
                    )
            del expected, expected_external, validation_results

    out_legals_batch = np.empty((B, n_lists), dtype=np.float32)
    out_pred_batch = np.empty((B, n_lists), dtype=np.float32)

    sweep_results: list[SweepResult] = []
    timing_metric_fields = [
        f"{stage}_{stat}_ms"
        for stage in timing_stages
        for stat in ("median", "p95")
    ]

    print("n_probe,recall,qps")
    print(
        "timings_batch_header,n_probe,samples,qps,"
        + ",".join(timing_metric_fields)
    )

    for n_probe in nprobes:
        if not args.val_flat:
            index.nprobe = n_probe
        params = (
            faiss.SearchParametersSelector2D()
            if args.val_flat
            else faiss.SearchParametersIVF(nprobe=n_probe)
        )
        batches = list(batch_slices(num_queries, B))
        returned = np.full((num_queries, rk), -1, dtype=np.int64)
        timing_samples: list[list[float]] = []
        out_rankings_batch = np.empty((B, n_probe), dtype=np.int64)

        def select_topk(scores: torch.Tensor) -> torch.Tensor:
            rankings = topk_small(
                scores.numpy(),
                n_probe,
                out=out_rankings_batch[: scores.shape[0]],
                threads=0,
            )
            return torch.from_numpy(rankings)

        def execute_trace(*, record: bool) -> None:
            for i, rows in enumerate(batches):
                qi = queries[rows]
                size = len(qi)
                batch_start = perf_counter()

                stage_start = perf_counter()
                predicate_results = predicate_workload.evaluate_batch(
                    rows.start, rows.stop, threads=0
                )
                predicate_eval_time = perf_counter() - stage_start

                stage_start = perf_counter()
                mi = materialize_results(
                    predicate_results, n, predicate_bitmap[:size], threads=0
                )
                mask_materialize_time = perf_counter() - stage_start

                if not args.val_flat:
                    stage_start = perf_counter()
                    qi_t = queries_t[rows]
                    cluster_dis = qi_t @ centroids_t.T
                    if args.alpha != 0.0 or beta != 0.0:
                        cluster_legals(
                            mi,
                            offsets,
                            counts,
                            out=out_legals_batch[:size],
                            threads=0,
                        )
                    if args.alpha != 0.0 or beta != 0.0:
                        pred_term = predicate_log_scores(
                            out_legals_batch[:size],
                            beta,
                            out=out_pred_batch[:size],
                            threads=0,
                        )
                        pred_term = torch.from_numpy(pred_term)
                        cluster_scores = cluster_dis + pred_term
                    else:
                        cluster_scores = cluster_dis
                    cluster_scoring_time = perf_counter() - stage_start

                    stage_start = perf_counter()
                    rankings_adj = select_topk(cluster_scores)
                    cluster_rank_time = perf_counter() - stage_start

                    stage_start = perf_counter()
                    Dq_t = torch.gather(cluster_dis, 1, rankings_adj)
                    Iq = np.ascontiguousarray(rankings_adj.cpu().numpy())
                    if args.alpha != 0.0 or beta != 0.0:
                        Iq = exclude_empty_probes(Iq, out_legals_batch[:size])
                    Dq = np.ascontiguousarray(Dq_t.cpu().numpy())
                else:
                    cluster_scoring_time = cluster_rank_time = 0.0
                    stage_start = perf_counter()
                sel = faiss.IDSelector2DBitmap(mi)
                params.sel2d = sel
                search_prep_time = perf_counter() - stage_start

                stage_start = perf_counter()
                if debug:
                    bytes_per_bitmap = mi.shape[1] if mi.ndim == 2 else None
                    print(
                        "debug: search_preassigned start "
                        f"batch={i} B={mi.shape[0]} bytes_per_bitmap={bytes_per_bitmap} "
                        f"nprobe={n_probe} ntotal={getattr(index, 'ntotal', 'n/a')}",
                        flush=True,
                    )

                if not args.val_flat:
                    _, indices = index.search_preassigned(
                        qi, rk, Iq, Dq, params=params
                    )
                else:
                    _, indices = index.search(qi, rk, params=params)
                if use_gpu:
                    _gpu_resources.syncDefaultStreamCurrentDevice()
                if debug:
                    print("debug: search_preassigned done", flush=True)
                vector_search_time = perf_counter() - stage_start

                valid = indices >= 0
                returned[rows].fill(-1)
                returned[rows][valid] = internal_to_external[indices[valid]]

                batch_total_time = perf_counter() - batch_start
                if record:
                    accounted_components = [
                        predicate_eval_time,
                        mask_materialize_time,
                        cluster_scoring_time,
                        cluster_rank_time,
                        search_prep_time,
                    ]
                    accounted_components.append(vector_search_time)
                    other_time = max(
                        batch_total_time - sum(accounted_components), 0.0
                    )
                    timing_samples.append(
                        accounted_components + [other_time, batch_total_time]
                    )

        measurement = run_measured_iterations(
            lambda record: execute_trace(record=record),
            warmup=warmup,
            iterations=iters,
            num_queries=num_queries,
            min_iterations=args.min_iterations,
            qps_relative_ci_half_width=args.qps_relative_ci_half_width,
        )
        measurement_context = f"dataset={dataset.name}, n_probe={n_probe}"
        if adaptive:
            assert measurement.qps_ci_lower is not None
            assert measurement.qps_ci_upper is not None
            assert measurement.qps_relative_ci_half_width is not None
            print(
                f"qps_confidence,{n_probe},"
                f"{len(measurement.iteration_qps)},{measurement.qps:.2f},"
                f"{measurement.qps_ci_lower:.2f},"
                f"{measurement.qps_ci_upper:.2f},"
                f"{measurement.qps_relative_ci_half_width:.4%}",
                flush=True,
            )
            require_measurement_convergence(
                measurement,
                context=measurement_context,
            )
        else:
            require_iteration_agreement(
                measurement,
                context=measurement_context,
            )

        timing_array = np.asarray(timing_samples, dtype=np.float64)
        expected_samples = len(measurement.iteration_qps) * len(batches)
        timing_summary = summarize_timings(
            timing_array,
            timing_stages,
            expected_samples=expected_samples,
        )
        timing_record: dict[str, float | int] = {
            "n_probe": n_probe,
            "samples": timing_array.shape[0],
            **timing_summary,
        }
        measured_qps = measurement.qps
        timing_record["qps"] = measured_qps
        timing_values = [
            f"{float(timing_record[field]):.6f}"
            for field in timing_metric_fields
        ]
        print(
            f"timings_batch,{n_probe},{timing_record['samples']},"
            f"{measured_qps:.2f}," + ",".join(timing_values)
        )

        returned = returned.reshape(-1, rk)
        result = SweepResult(
            parameter=n_probe,
            recall=recall_at_k(returned, gt_all, k),
            qps=measured_qps,
            timing_samples=timing_array.shape[0],
            timing_summary=timing_summary,
        )
        sweep_results.append(result)
        result_str = f"{n_probe},{result.recall:.3f},{result.qps:.2f}"

        print(result_str)

    method = (
        "pre-filter" if flat else "orchid" if args.alpha != 0.0
        else "orchid-p" if beta != 0.0 else "ivf"
    )
    experiment = args.exp_name or method
    write_timing_results(
        sweep_results, log_dir, experiment, parameter_name="n_probe"
    )
    output = write_sweep_results(
        sweep_results, log_dir, experiment, parameter_name="n_probe"
    )
    print(f"Wrote sweep: {output}")


if __name__ == "__main__":
    main()
