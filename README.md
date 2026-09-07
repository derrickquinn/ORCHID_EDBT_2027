# ORCHID EDBT artifact

This artifact packages the modern ORCHID evaluation harness and the shared
ACORN, NaviX, and CAGRA baseline harness. All methods evaluate predicates online
and include predicate evaluation and mask materialization in measured latency.
ORCHID, ORCHID-P, IVF, and pre-filter use `orchid/perf.py`; graph baselines use
`orchid/perf_baseline.py` through thin method entry points.

## Environment and native builds

The supported environment is Linux x86-64 with AVX2. Install
[Pixi](https://pixi.sh), then run from this repository:

```bash
pixi install
pixi run build-faiss
pixi run build-baselines
pixi run check
pixi run smoke-baselines
```

The environment includes Python, C++ build tools, MKL, and Python dependencies.
`utils_cpp` is installed from bundled sources. FAISS is built
separately before installing its generated Python package; stock FAISS wheels
do not provide the required filtered-search extensions.

The ACORN/NaviX build fetches and verifies upstream revisions pinned in
`orchid/baseline_adapters/build_adapters.py`, then builds their Python adapters.
It needs network access on the first build. See [build details](docs/build.md)
for source overrides, CPU tuning, and optional GPU support.

## Local data

Datasets and metadata are **not distributed with this artifact**. Supply the
prepared stores and metadata described in [the input specification](docs/data.md).
For an existing asset directory:

```bash
ln -s /absolute/path/to/assets/data data
export ORCHID_ASSET_ROOT=/absolute/path/to/assets
```

The regression suite and native adapter smoke tests use synthetic inputs and
do not require the paper datasets.

## Run an evaluation

The launcher selects the checked-in paper configuration and runs from `orchid/`,
so input/config/cache paths have the same meaning for all methods.

```bash
pixi run benchmark orchid sift12
pixi run benchmark orchid-p sift12
pixi run benchmark ivf sift12
pixi run benchmark pre-filter sift12
pixi run benchmark acorn sift12 -- --build-index
pixi run benchmark navix sift12 -- --build-index
```

For ACORN/NaviX, `--build-index` builds and saves an index before measurement.
Omit it to load an existing index. Use `--build-only` to construct without
running a sweep. Index construction is outside the timed region.

Supported datasets are `sift12`, `sift24`, `sift48`, `sift_r10`, `sift_r20`,
`sift_r40`, `laion_all`, `laion_pos`, `laion_neg`, `yfcc`, and `yfcc_single`.
Use `pixi run benchmark orchid sift12 --dry-run` to inspect a command without
loading data. Pass additional harness arguments after `--`; for example:

```bash
pixi run benchmark orchid sift12 -- --validate_predicates --exp_name checked-orchid
pixi run benchmark navix sift12 -- --validate-predicates --exp-name checked-navix
```

The paper workload uses CPU B=100 and 32 threads, or GPU B=1. The launcher defaults to
`OMP_NUM_THREADS=32`, `OMP_PROC_BIND=true`, and `OMP_PLACES=cores`, respecting
explicit environment overrides. Preserve CPU affinity, OpenMP settings, and
NUMA placement across methods. For example, on a machine where
CPUs 4–35 and NUMA nodes 0–1 are available:

```bash
numactl -C 4-35 -i 0,1 env OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores \
  pixi run benchmark orchid sift12
```

Choose a placement appropriate for your machine and use it for every comparison.
The harness selects `zarrs.ZarrsCodecPipeline` for dataset I/O; preserve this
pipeline when evaluating performance.

## CAGRA

CAGRA uses its own CUDA/cuVS environment and native adapter:

```bash
pixi install --manifest-path orchid/cagra_env/pixi.toml
pixi run --manifest-path orchid/cagra_env/pixi.toml build-cagra
pixi run --manifest-path orchid/cagra_env/pixi.toml smoke-cagra
pixi run --manifest-path orchid/cagra_env/pixi.toml benchmark cagra sift12 -- --build-index
```

Use the same CPU/NUMA placement as the other methods. The adapter uses cuVS
25.08 and CUDA 12.8, with one query per predicate/transfer/search batch (B=1).
`--cuda-architectures` on the build task overrides GPU detection.

## Outputs and measurement

Each completed sweep writes ordinary local files under `orchid/logs/<dataset>/`:

- `<experiment>.csv`: search parameter, recall@k, and QPS.
- `<experiment>.timings.csv`: measured batch count and median/p95 stage latency.

Use a different `--exp_name` (ORCHID/IVF) or `--exp-name` (baselines) to retain
multiple runs. Reusing a name replaces its CSV files. Resolved configurations
and adaptive QPS confidence intervals are printed to the terminal.

Paper configs use one untimed full-trace warmup, at least three measured traces,
and up to 100 traces to meet a 5% relative QPS confidence half-width. A sweep
that does not converge fails before writing completed-sweep CSVs.
[Measurement details](docs/measurement.md) describe stage boundaries, predicate
validation, and search-only ablations.

Calibration entry points are retained with their YAML recipes:

```bash
pixi run calibrate --config experiments/grid/sift12.yaml
pixi run optimize --config experiments/spool/sift12.yaml
```

These tasks run from `orchid/`. They report offline scan-cost estimates, not
online QPS.
