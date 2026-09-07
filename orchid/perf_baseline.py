"""Shared end-to-end harness for the ACORN, NaviX, and CAGRA baselines."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Protocol

import numpy as np
import zarr
from query_settings import query_slice, batch_size_for, validate_split

from predicate_index import materialize_byte_masks, materialize_results
from predicate_workload import (
    acorn_construction_metadata,
    load_or_build_predicate_workload,
)
from perf_results import (
    BASELINE_TIMING_STAGES,
    SweepResult,
    recall_at_k,
    require_iteration_agreement,
    require_measurement_convergence,
    run_measured_iterations,
    summarize_timings,
    write_timing_results,
    write_sweep_results,
)
from src.baseline_perf_config import load_baseline_perf_config
from src.dataset_config import DatasetConfig, load_dataset_config


zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


TIMING_STAGES = BASELINE_TIMING_STAGES
ADAPTER_INSTALL_DIR = Path(__file__).resolve().parent / "baseline_adapters/_install"


class PredicateBatchSource(Protocol):
    def evaluate_batch(self, start: int, stop: int, *, threads: int = 0) -> list: ...


class BaselineIndex(Protocol):
    ntotal: int
    d: int

    def search(
        self,
        queries: np.ndarray,
        masks: np.ndarray,
        k: int,
        ef_search: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(slots=True)
class OnlinePass:
    labels: np.ndarray
    timings: np.ndarray


def _import_adapter(baseline: str):
    if ADAPTER_INSTALL_DIR.exists() and str(ADAPTER_INSTALL_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPTER_INSTALL_DIR))
    try:
        return importlib.import_module(f"{baseline}_adapter")
    except ModuleNotFoundError as error:
        expected_modules = {f"{baseline}_adapter"}
        if baseline == "cagra":
            expected_modules.add("_cagra_native")
        if error.name not in expected_modules:
            raise
        build_command = (
            "pixi run --manifest-path cagra_env/pixi.toml build-cagra"
            if baseline == "cagra"
            else "baseline_adapters/build_adapters.py"
        )
        raise ModuleNotFoundError(
            f"{baseline.upper()} adapter is not installed; run {build_command}"
        ) from error


def run_online_pass(
    index: BaselineIndex,
    predicate_workload: PredicateBatchSource,
    queries: np.ndarray,
    mask_buffer: np.ndarray,
    *,
    k: int,
    ef_search: int,
    batch_size: int,
    threads: int = 0,
    record_timings: bool = True,
    packed_masks: bool = False,
) -> OnlinePass:
    """Run one pass through predicate evaluation, mask construction, and search."""
    if (
        queries.ndim != 2
        or queries.dtype != np.float32
        or not queries.flags.c_contiguous
    ):
        raise ValueError("queries must be a C-contiguous float32 matrix")
    if queries.shape[1] != index.d:
        raise ValueError("query dimension does not match the index")
    if batch_size <= 0 or k <= 0 or ef_search <= 0:
        raise ValueError("batch_size, k, and ef_search must be positive")
    mask_width = (index.ntotal + 7) // 8 if packed_masks else index.ntotal
    if (
        mask_buffer.dtype != np.uint8
        or not mask_buffer.flags.c_contiguous
        or mask_buffer.ndim != 2
        or mask_buffer.shape[0] < min(batch_size, len(queries))
        or mask_buffer.shape[1] != mask_width
    ):
        raise ValueError(
            "mask_buffer has the wrong shape for the selected mask representation"
        )

    labels = np.full((len(queries), k), -1, dtype=np.int64)
    timing_rows: list[list[float]] = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        batch_start = perf_counter()

        stage_start = perf_counter()
        predicate_results = predicate_workload.evaluate_batch(
            start, stop, threads=threads
        )
        predicate_eval_time = perf_counter() - stage_start
        if len(predicate_results) != stop - start:
            raise RuntimeError("predicate workload returned the wrong batch size")

        stage_start = perf_counter()
        materialize = materialize_results if packed_masks else materialize_byte_masks
        masks = materialize(
            predicate_results,
            index.ntotal,
            mask_buffer,
            threads=threads,
        )
        mask_materialize_time = perf_counter() - stage_start

        stage_start = perf_counter()
        _, batch_labels = index.search(
            queries[start:stop], masks, k=k, ef_search=ef_search
        )
        vector_search_time = perf_counter() - stage_start
        if batch_labels.shape != (stop - start, k):
            raise RuntimeError("baseline adapter returned labels with the wrong shape")
        labels[start:stop] = batch_labels

        if record_timings:
            total_time = perf_counter() - batch_start
            components = (
                predicate_eval_time + mask_materialize_time + vector_search_time
            )
            timing_rows.append(
                [
                    predicate_eval_time,
                    mask_materialize_time,
                    vector_search_time,
                    max(total_time - components, 0.0),
                    total_time,
                ]
            )

    timings = np.asarray(timing_rows, dtype=np.float64).reshape(-1, len(TIMING_STAGES))
    return OnlinePass(labels, timings)


def run_precomputed_mask_pass(
    index: BaselineIndex,
    queries: np.ndarray,
    masks: np.ndarray,
    *,
    k: int,
    ef_search: int,
    batch_size: int,
    record_timings: bool = True,
    packed_masks: bool = False,
) -> OnlinePass:
    """Run filtered search with predicate masks materialized before timing."""
    if (
        queries.ndim != 2
        or queries.dtype != np.float32
        or not queries.flags.c_contiguous
    ):
        raise ValueError("queries must be a C-contiguous float32 matrix")
    if queries.shape[1] != index.d:
        raise ValueError("query dimension does not match the index")
    if batch_size <= 0 or k <= 0 or ef_search <= 0:
        raise ValueError("batch_size, k, and ef_search must be positive")
    mask_width = (index.ntotal + 7) // 8 if packed_masks else index.ntotal
    if (
        masks.dtype != np.uint8
        or not masks.flags.c_contiguous
        or masks.shape != (len(queries), mask_width)
    ):
        raise ValueError("masks have the wrong shape for the selected representation")

    labels = np.full((len(queries), k), -1, dtype=np.int64)
    timing_rows: list[list[float]] = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        batch_start = perf_counter()
        search_start = perf_counter()
        _, batch_labels = index.search(
            queries[start:stop], masks[start:stop], k=k, ef_search=ef_search
        )
        vector_search_time = perf_counter() - search_start
        if batch_labels.shape != (stop - start, k):
            raise RuntimeError("baseline adapter returned labels with the wrong shape")
        labels[start:stop] = batch_labels

        if record_timings:
            total_time = perf_counter() - batch_start
            timing_rows.append(
                [
                    0.0,
                    0.0,
                    vector_search_time,
                    max(total_time - vector_search_time, 0.0),
                    total_time,
                ]
            )

    timings = np.asarray(timing_rows, dtype=np.float64).reshape(-1, len(TIMING_STAGES))
    return OnlinePass(labels, timings)


def build_arg_parser(baseline: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the {baseline.upper()} baseline with online predicates"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dset-cfg", "--dataset-config", required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--index-base", type=Path, default=Path("cached/baselines"))
    parser.add_argument("--index-name")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="construct and save the index without entering the benchmark path",
    )
    parser.add_argument("--cache-base", type=Path, default=Path("cached"))
    parser.add_argument("--predicate-cache", type=Path)
    parser.add_argument("--log-base", type=Path, default=Path("logs"))
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--query-start", type=int, default=1024)
    parser.add_argument("--num-queries", type=int, default=10000)
    parser.add_argument(
        "--batch-size", type=int, default=1 if baseline == "cagra" else 100
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-iterations", type=int)
    parser.add_argument("--qps-relative-ci-half-width", type=float)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--metric", choices=("l2", "ip"), default="ip")
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--validate-predicates", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--precompute-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="materialize every query mask before timing to measure native search only",
    )
    if baseline in {"acorn", "navix"}:
        parser.add_argument("--ef-search", type=int, nargs="+", default=[16])
        parser.add_argument("--m", type=int, default=32)
        parser.add_argument(
            "--ef-construction", type=int, default=None if baseline == "acorn" else 40,
            help="construction search width (ACORN defaults to M * gamma)",
        )
    if baseline == "acorn":
        parser.add_argument("--gamma", type=int, default=12)
        parser.add_argument("--m-beta", type=int, default=64)
        metadata = parser.add_mutually_exclusive_group()
        metadata.add_argument("--acorn-metadata", type=Path)
        metadata.add_argument("--acorn-dummy-metadata", action="store_true")
        metadata.add_argument("--acorn-workload-metadata", action="store_true")
    elif baseline == "cagra":
        parser.add_argument("--itopk-size", type=int, nargs="+", default=[64])
        parser.add_argument("--graph-degree", type=int, default=32)
        parser.add_argument("--intermediate-graph-degree", type=int, default=64)
        parser.add_argument("--device", type=int, default=0)
    return parser


def _cli_override_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    overrides: set[str] = set()
    for token in argv:
        if token == "--":
            break
        action = parser._option_string_actions.get(token.split("=", 1)[0])
        if action is not None and action.dest not in {
            "config",
            "dset_cfg",
        }:
            overrides.add(action.dest)
    return overrides


def parse_args(
    baseline: str,
) -> tuple[argparse.Namespace, DatasetConfig, dict[str, object] | None]:
    parser = build_arg_parser(baseline)
    argv = __import__("sys").argv[1:]
    args = parser.parse_args(argv)
    dataset = load_dataset_config(args.dset_cfg)
    resolved_config: dict[str, object] | None = None
    config = None

    if args.config is not None:
        cli_values = vars(args).copy()
        overrides = _cli_override_dests(parser, argv)
        config = load_baseline_perf_config(
            args.config, dataset_name=dataset.name, baseline=baseline
        )
        config_values = config.model_dump()
        resolved_config = config_values
        for field, value in config_values.items():
            if field == "baseline":
                continue
            if hasattr(args, field):
                setattr(args, field, value)
        for field in overrides:
            setattr(args, field, cli_values[field])
        for field in ("cache_base", "index_base", "log_base"):
            setattr(args, field, Path(getattr(args, field)))

        metadata_flags = {
            "acorn_metadata",
            "acorn_dummy_metadata",
            "acorn_workload_metadata",
        }
        if baseline == "acorn" and not overrides.intersection(metadata_flags):
            args.acorn_dummy_metadata = config.metadata_mode == "dummy"
            args.acorn_workload_metadata = config.metadata_mode == "workload_scalar"

    if baseline == "acorn" and args.ef_construction is None:
        args.ef_construction = args.m * args.gamma

    if args.index is None:
        if args.config is None:
            parser.error("--index is required unless --config supplies index_base")
        index_name = args.index_name or dataset.name
        if baseline == "acorn":
            index_name += f"_efc{args.ef_construction}"
        extension = ".cagra" if baseline == "cagra" else ".faiss"
        args.index = args.index_base / baseline / f"{index_name}{extension}"

    if config is not None:
        resolved_config = {
            field: getattr(args, field)
            for field in type(config).model_fields
            if field != "baseline" and hasattr(args, field)
        }
        resolved_config["baseline"] = baseline
        if baseline == "acorn":
            if args.acorn_metadata is not None:
                resolved_config["metadata_mode"] = "file"
                resolved_config["metadata_path"] = str(args.acorn_metadata)
            elif args.acorn_workload_metadata:
                resolved_config["metadata_mode"] = "workload_scalar"
            else:
                resolved_config["metadata_mode"] = "dummy"
        for field, value in tuple(resolved_config.items()):
            if isinstance(value, Path):
                resolved_config[field] = str(value)
        resolved_config["precompute_masks"] = args.precompute_masks

    return args, dataset, resolved_config


def _normalize_rows(vectors: np.ndarray) -> None:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector")
    vectors /= norms


def _load_vectors(
    dataset: DatasetConfig,
    num_queries: int,
    *,
    load_docs: bool,
    query_start: int = 1024,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    group = zarr.open_group(dataset.vec_zarr, mode="r")
    queries = np.ascontiguousarray(
        group["test"][
            query_slice(
                group["test"].shape[0], query_start, num_queries, name="test vectors"
            )
        ],
        dtype=np.float32,
    )
    n_docs = int(group["train"].shape[0])
    docs = (
        np.ascontiguousarray(group["train"][:], dtype=np.float32) if load_docs else None
    )
    return docs, queries, n_docs


def _load_or_build_index(
    baseline: str,
    adapter,
    args,
    dataset: DatasetConfig,
    docs: np.ndarray | None,
):
    if not args.build_index:
        if not args.index.exists():
            raise FileNotFoundError(
                f"baseline index does not exist: {args.index}; use --build-index"
            )
        if baseline == "cagra":
            return adapter.Index.load(str(args.index), device=args.device)
        return adapter.Index.load(str(args.index))

    if docs is None:
        raise AssertionError("documents were not loaded for index construction")
    metric = adapter.Metric.L2 if args.metric == "l2" else adapter.Metric.INNER_PRODUCT
    if baseline == "acorn":
        if args.acorn_metadata is not None:
            metadata = np.ascontiguousarray(
                np.load(args.acorn_metadata, allow_pickle=False), dtype=np.int32
            )
        elif args.acorn_dummy_metadata:
            metadata = np.zeros(len(docs), dtype=np.int32)
        elif args.acorn_workload_metadata:
            metadata = acorn_construction_metadata(
                dataset, len(docs), "workload_scalar"
            )
        else:
            raise ValueError("building ACORN requires an explicit metadata policy")
        if metadata.shape != (len(docs),):
            raise ValueError("ACORN metadata must contain one int32 value per document")
        build_start = perf_counter()
        index = adapter.Index.build(
            docs,
            metadata,
            m=args.m,
            gamma=args.gamma,
            m_beta=args.m_beta,
            ef_construction=args.ef_construction,
            metric=metric,
        )
    elif baseline == "navix":
        build_start = perf_counter()
        index = adapter.Index.build(
            docs,
            m=args.m,
            ef_construction=args.ef_construction,
            metric=metric,
        )
    else:
        build_start = perf_counter()
        index = adapter.Index.build(
            docs,
            graph_degree=args.graph_degree,
            intermediate_graph_degree=args.intermediate_graph_degree,
            device=args.device,
            metric=metric,
        )
    build_seconds = perf_counter() - build_start
    args.index.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = args.index.with_suffix(args.index.suffix + ".tmp")
    save_start = perf_counter()
    index.save(str(temporary_index))
    temporary_index.replace(args.index)
    save_seconds = perf_counter() - save_start
    print(f"index_build_seconds: {build_seconds:.6f}")
    print(f"index_save_seconds: {save_seconds:.6f}")
    return index


def _validate_predicates(
    dataset: DatasetConfig,
    predicate_workload: PredicateBatchSource,
    mask_buffer: np.ndarray,
    *,
    num_queries: int,
    batch_size: int,
    n_docs: int,
    threads: int,
    packed_masks: bool,
    query_start: int = 1024,
) -> None:
    oracle = zarr.open_group(dataset.pred_zarr, mode="r")["pred_mask"]
    for start in range(0, num_queries, batch_size):
        stop = min(start + batch_size, num_queries)
        results = predicate_workload.evaluate_batch(start, stop, threads=threads)
        if packed_masks:
            packed = materialize_results(
                results, n_docs, mask_buffer[: stop - start], threads=threads
            )
            actual = np.unpackbits(
                packed, axis=1, count=n_docs, bitorder="little"
            )
        else:
            actual = materialize_byte_masks(
                results, n_docs, mask_buffer[: stop - start], threads=threads
            )
        expected = np.asarray(
            oracle[query_start + start : query_start + stop, :n_docs], dtype=np.uint8
        )
        if expected.shape != actual.shape:
            raise ValueError("legacy predicate mask shape does not match the index")
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(np.any(actual != expected, axis=1))[0])
            raise AssertionError(
                f"predicate index differs from legacy mask oracle at query {start + mismatch}"
            )


def main(baseline: str) -> None:
    if baseline not in {"acorn", "navix", "cagra"}:
        raise ValueError(f"unsupported baseline: {baseline}")
    parser = build_arg_parser(baseline)
    args, dataset, resolved_config = parse_args(baseline)
    if any(
        value <= 0
        for value in (
            args.num_queries,
            args.batch_size,
            args.iterations,
            args.k,
        )
    ):
        parser.error(
            "num-queries, batch-size, iterations, and k must be positive"
        )
    if baseline in {"acorn", "navix"} and (
        args.m <= 0 or args.ef_construction <= 0
    ):
        parser.error("m and ef-construction must be positive")
    batch_size_for(baseline == "cagra", args.batch_size)
    validate_split(args.query_start, args.num_queries, 512, calibration=False)
    if args.build_only and not args.build_index:
        parser.error("--build-only requires --build-index")
    if args.warmup < 0 or args.threads < 0:
        parser.error("warmup and threads must be non-negative")
    adaptive = (
        args.min_iterations is not None
        or args.qps_relative_ci_half_width is not None
    )
    if adaptive:
        if (
            args.min_iterations is None
            or args.qps_relative_ci_half_width is None
        ):
            parser.error(
                "adaptive measurement requires min-iterations and "
                "qps-relative-ci-half-width"
            )
        if not 3 <= args.min_iterations <= args.iterations:
            parser.error("min-iterations must be between 3 and iterations")
        if args.qps_relative_ci_half_width <= 0:
            parser.error("qps-relative-ci-half-width must be positive")
    if baseline in {"acorn", "navix"} and any(
        value <= 0 for value in args.ef_search
    ):
        parser.error("ef-search must be positive")
    if baseline == "acorn" and (args.gamma <= 0 or args.m_beta <= 0):
        parser.error("gamma and m-beta must be positive")
    if baseline == "cagra" and (
        args.graph_degree <= 0
        or args.intermediate_graph_degree <= 0
        or args.device < 0
        or any(value <= 0 for value in args.itopk_size)
    ):
        parser.error("CAGRA construction and search values are invalid")

    if resolved_config is not None:
        import yaml

        print(
            yaml.safe_dump(
                {
                    "kind": "baseline_perf_resolved",
                    "args": resolved_config,
                    "dataset": dataset.model_dump(),
                    "index": str(args.index),
                },
                sort_keys=False,
            )
        )
    adapter = _import_adapter(baseline)
    ground_truth_store = zarr.open_group(dataset.pred_zarr, mode="r")["neighbors"]
    query_slice(
        ground_truth_store.shape[0],
        args.query_start,
        args.num_queries,
        name="ground truth",
    )
    docs, queries, dataset_n_docs = _load_vectors(
        dataset,
        args.num_queries,
        load_docs=args.build_index,
        query_start=args.query_start,
    )
    if len(queries) != args.num_queries:
        raise ValueError("dataset contains fewer queries than requested")
    if args.normalize:
        _normalize_rows(queries)
        if docs is not None:
            _normalize_rows(docs)

    index = _load_or_build_index(baseline, adapter, args, dataset, docs)
    if index.ntotal <= 0:
        raise ValueError("baseline index is empty")
    if index.ntotal != dataset_n_docs:
        raise ValueError("baseline index size does not match the dataset")
    if index.d != queries.shape[1]:
        raise ValueError("query dimension does not match the baseline index")
    if args.build_only:
        print(f"built_index: {args.index}")
        return

    predicate_cache = args.predicate_cache or (
        args.cache_base / dataset.name / "baseline_predicates_external"
    )
    document_ids = np.arange(index.ntotal, dtype=np.int64)
    packed_masks = baseline == "cagra"
    predicate_workload = load_or_build_predicate_workload(
        dataset,
        predicate_cache,
        document_ids,
        index.ntotal,
        args.num_queries,
        mask_format="packed" if packed_masks else "byte",
        query_start=args.query_start,
    )
    mask_width = (index.ntotal + 7) // 8 if packed_masks else index.ntotal
    mask_buffer = np.empty(
        (min(args.batch_size, args.num_queries), mask_width), dtype=np.uint8
    )

    if args.validate_predicates:
        _validate_predicates(
            dataset,
            predicate_workload,
            mask_buffer,
            num_queries=args.num_queries,
            batch_size=args.batch_size,
            n_docs=index.ntotal,
            threads=args.threads,
            packed_masks=packed_masks,
            query_start=args.query_start,
        )

    precomputed_masks: np.ndarray | None = None
    if args.precompute_masks:
        precomputed_masks = np.empty((args.num_queries, mask_width), dtype=np.uint8)
        for start in range(0, args.num_queries, args.batch_size):
            stop = min(start + args.batch_size, args.num_queries)
            predicate_results = predicate_workload.evaluate_batch(
                start, stop, threads=args.threads
            )
            materialize = materialize_results if packed_masks else materialize_byte_masks
            materialize(
                predicate_results,
                index.ntotal,
                precomputed_masks[start:stop],
                threads=args.threads,
            )

    ground_truth = np.asarray(
        zarr.open_group(dataset.pred_zarr, mode="r")["neighbors"][
            args.query_start : args.query_start + args.num_queries, : args.k
        ]
    )
    if ground_truth.shape != (args.num_queries, args.k):
        raise ValueError(
            "ground truth does not contain the requested query count and k"
        )
    timing_fields = [
        f"{stage}_{stat}_ms" for stage in TIMING_STAGES for stat in ("median", "p95")
    ]
    parameter_name = "itopk_size" if baseline == "cagra" else "ef_search"
    sweep_values = args.itopk_size if baseline == "cagra" else args.ef_search
    print(f"baseline,{parameter_name},recall,qps," + ",".join(timing_fields))

    sweep_results: list[SweepResult] = []
    for search_parameter in sweep_values:
        passes: list[OnlinePass] = []

        def execute_trace(record: bool) -> None:
            if precomputed_masks is None:
                result = run_online_pass(
                    index,
                    predicate_workload,
                    queries,
                    mask_buffer,
                    k=args.k,
                    ef_search=search_parameter,
                    batch_size=args.batch_size,
                    threads=args.threads,
                    record_timings=record,
                    packed_masks=packed_masks,
                )
            else:
                result = run_precomputed_mask_pass(
                    index,
                    queries,
                    precomputed_masks,
                    k=args.k,
                    ef_search=search_parameter,
                    batch_size=args.batch_size,
                    record_timings=record,
                    packed_masks=packed_masks,
                )
            if record:
                passes.append(result)

        measurement = run_measured_iterations(
            execute_trace,
            warmup=args.warmup,
            iterations=args.iterations,
            num_queries=args.num_queries,
            min_iterations=args.min_iterations,
            qps_relative_ci_half_width=args.qps_relative_ci_half_width,
        )
        measurement_context = (
            f"baseline={baseline}, dataset={dataset.name}, "
            f"{parameter_name}={search_parameter}"
        )
        if adaptive:
            assert measurement.qps_ci_lower is not None
            assert measurement.qps_ci_upper is not None
            assert measurement.qps_relative_ci_half_width is not None
            print(
                f"qps_confidence,{search_parameter},"
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
        timings = np.concatenate([result.timings for result in passes])
        summary = summarize_timings(
            timings,
            TIMING_STAGES,
            expected_samples=len(measurement.iteration_qps)
            * ((args.num_queries + args.batch_size - 1) // args.batch_size),
        )
        recall = recall_at_k(passes[-1].labels, ground_truth, args.k)
        result = SweepResult(
            parameter=search_parameter,
            recall=recall,
            qps=measurement.qps,
            timing_samples=timings.shape[0],
            timing_summary=summary,
        )
        sweep_results.append(result)
        values = ",".join(f"{summary[field]:.6f}" for field in timing_fields)
        print(
            f"{baseline},{result.parameter},{result.recall:.6f},"
            f"{result.qps:.2f},{values}"
        )

    experiment = args.exp_name or baseline
    write_timing_results(
        sweep_results, args.log_base / dataset.name, experiment,
        parameter_name=parameter_name,
    )
    output = write_sweep_results(
        sweep_results, args.log_base / dataset.name, experiment,
        parameter_name=parameter_name,
    )
    print(f"Wrote sweep: {output}")


__all__ = [
    "OnlinePass",
    "TIMING_STAGES",
    "build_arg_parser",
    "main",
    "parse_args",
    "run_online_pass",
    "run_precomputed_mask_pass",
]
