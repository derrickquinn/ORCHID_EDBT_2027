"""Exercise complete harnesses on temporary synthetic data; not a performance run."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("orchid", "orchid-p", "ivf", "pre-filter", "acorn", "navix", "cagra")


def make_inputs(directory: Path) -> Path:
    """Generate a small trace matching the harness's deterministic SIFT predicates."""
    rng = np.random.default_rng(7)
    docs = rng.standard_normal((2048, 32)).astype(np.float32)
    queries = rng.standard_normal((1124, 32)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    predicate_rng = np.random.default_rng(42)
    values = predicate_rng.integers(0, 12, size=len(docs))
    terms = predicate_rng.integers(0, 12, size=len(queries))
    masks = terms[:, None] == values
    scores = queries @ docs.T
    scores[~masks] = -np.inf
    neighbors = np.argsort(-scores, axis=1)[:, :10].astype(np.int64)
    vectors = directory / "vectors.zarr"
    predicates = directory / "predicates.zarr"
    vg = zarr.open_group(str(vectors), mode="w")
    vg.create_array("train", data=docs)
    vg.create_array("test", data=queries)
    pg = zarr.open_group(str(predicates), mode="w")
    pg.create_array("pred_mask", data=masks)
    pg.create_array("neighbors", data=neighbors)
    config = directory / "dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "sift12",
                "vec_zarr": str(vectors),
                "pred_zarr": str(predicates),
                "nlists": 16,
            }
        )
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS[:4])
    )
    args = parser.parse_args()
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    assert zarr.config.get("codec_pipeline.path") == "zarrs.ZarrsCodecPipeline"
    with tempfile.TemporaryDirectory(prefix="orchid-harness-") as name:
        directory = Path(name)
        config = make_inputs(directory)
        for method in args.methods:
            if method in METHODS[:4]:
                alpha = "0.4472135954999579" if method == "orchid" else "0"
                beta = "0.1" if method in ("orchid", "orchid-p") else "0"
                command = [
                    sys.executable,
                    "perf.py",
                    "--dset_cfg",
                    str(config),
                    "--num_queries",
                    "100",
                    "--iterations",
                    "1",
                    "--warmup",
                    "0",
                    "--low_nprobe",
                    "16",
                    "--high_nprobe",
                    "16",
                    "--step_nprobe",
                    "1",
                    "--alpha",
                    alpha,
                    "--beta",
                    beta,
                    "--sig_dims",
                    "4",
                    "--sample_queries",
                    "32",
                    "--sample_docs",
                    "128",
                    "--cache_base",
                    str(directory / "cache"),
                    "--log_base",
                    str(directory / "logs"),
                    "--exp_name",
                    method,
                    "--validate_predicates",
                ]
                if method == "pre-filter":
                    command.append("--val_flat")
                parameter = "n_probe"
            else:
                command = [
                    sys.executable,
                    f"perf_{method}.py",
                    "--dset-cfg",
                    str(config),
                    "--num-queries",
                    "100",
                    "--iterations",
                    "1",
                    "--warmup",
                    "0",
                    "--metric",
                    "ip",
                    "--normalize",
                    "--build-index",
                    "--index",
                    str(directory / f"{method}.index"),
                    "--cache-base",
                    str(directory / "cache"),
                    "--log-base",
                    str(directory / "logs"),
                    "--exp-name",
                    method,
                    "--validate-predicates",
                ]
                if method == "cagra":
                    command += ["--itopk-size", "128"]
                    parameter = "itopk_size"
                else:
                    command += [
                        "--ef-search",
                        "128",
                        "--m",
                        "16",
                        "--ef-construction",
                        "40",
                    ]
                    parameter = "ef_search"
                    if method == "acorn":
                        command += [
                            "--acorn-workload-metadata",
                            "--gamma",
                            "12",
                            "--m-beta",
                            "32",
                        ]
            env = dict(
                os.environ,
                OMP_NUM_THREADS="32",
                OMP_PROC_BIND="true",
                OMP_PLACES="cores",
            )
            subprocess.run(command, cwd=ROOT / "orchid", env=env, check=True)
            output = directory / "logs/sift12" / f"{method}.csv"
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1 and parameter in rows[0], rows
            recall = float(rows[0]["recall"])
            assert recall >= 0.999 if method in METHODS[:4] else recall >= 0.5, rows
            assert float(rows[0]["qps"]) > 0, rows
            with output.with_suffix(".timings.csv").open() as handle:
                timings = list(csv.DictReader(handle))
            assert len(timings) == 1 and int(timings[0]["samples"]) == 1, timings
            assert "predicate_eval_median_ms" in timings[0], timings
            print(f"{method}: complete harness and local CSV export passed", flush=True)


if __name__ == "__main__":
    main()
