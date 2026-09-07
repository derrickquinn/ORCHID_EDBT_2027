# ACORN, NaviX, and CAGRA adapters

This directory and `../cagra_native` form the narrow ABI boundary between the
ORCHID benchmark code and the native baseline implementations. Neither pinned
Faiss fork is modified.

The ACORN and NaviX adapters:

- owns a native index and supports build, save, load, and filtered search;
- accepts a C-contiguous `uint8[nquery, ntotal]` mask without copying it;
- releases the Python GIL around native build, I/O, and search;
- returns `float32` distances and `int64` IDs; and
- verifies the fork revision when configured from a Git checkout.

The pinned revisions are:

- ACORN: `3996dcf90ca2fb58abe46679a13d6a57792274c2`
- NaviX: `25d563a2dec9087891f19055b8407d4383a26342`

## Build

The reproducible build task fetches the two public forks, checks out the pinned
commits above, builds static CPU-only Faiss libraries, builds both adapters, and
installs the Python extensions in `_install`:

```bash
python build_adapters.py
```

Existing fork checkouts can be reused and their revisions are still verified
before configuration (ACORN's upstream CMake may fetch its own declared build
dependencies if they are not already cached):

```bash
python build_adapters.py --no-fetch \
  --acorn-source /path/to/ACORN \
  --navix-source /path/to/faiss-navix
```

After `pixi run build-baselines`, `pixi run smoke-baselines` runs both native
smoke tests. `pixi run check-baseline-sources` is the non-compiling revision check.

`real_data_smoke_test.py` adds a bounded end-to-end check using real vectors,
the published predicate pipeline, native filtered search, and save/load. Run it
with the repository's required Zarrs pipeline and CPU/NUMA placement; it is a
correctness check and does not write benchmark results.

For manual builds, build each fork's `faiss` target normally in its own build
directory, then configure this directory in a separate adapter build directory:

```bash
cmake -S . -B build-navix \
  -DBASELINE=navix \
  -DBASELINE_FAISS_SOURCE_DIR=/path/to/faiss-navix \
  -DBASELINE_FAISS_LIBRARY=/path/to/navix-build/faiss/libfaiss.a \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build-navix -j
```

Use the same command with `BASELINE=acorn` and the ACORN source/library paths.
The modules are intentionally built separately, so neither fork becomes a
source-level dependency of the other.

Run the contract test with the selected build directory on `PYTHONPATH`:

```bash
PYTHONPATH=build-navix python smoke_test.py navix_adapter
PYTHONPATH=build-acorn python smoke_test.py acorn_adapter
```

`Index.search(queries, masks, k, ef_search)` deliberately rejects implicit
dtype or layout conversion. The caller owns mask construction and its timing;
the adapter only passes the byte buffer into the baseline search implementation.

ACORN's pinned serializer omits its constructor metadata. The adapter repairs
the fork's dangling metadata pointer after load. The graph already contains the
metadata-dependent structure built at construction time; online filtering is
still driven entirely by the supplied byte mask.

## Shared runner

`../perf_acorn.py`, `../perf_navix.py`, and `../perf_cagra.py` are thin entry
points over `../perf_baseline.py`. All use the published `PredicateWorkload`
pipeline and direct `utils_cpp` mask materializers. Their measured online path
is:

```text
predicate evaluation -> uint8 mask materialization -> native baseline search
```

ACORN and NaviX consume one byte per document. CAGRA consumes the canonical
little-endian packed bitmap directly, so its mask-materialization stage uses
`materialize_results` and does not charge packing to search. Its adapter is a
standalone native cuVS extension derived from
`nicolelii/CAGRA_SIGMOD_Artifacts` at
`79c153ffc4ebf4363f522097958f83d7eec19ad4`; it does not depend on Faiss. As in
that artifact, it constructs a `bitset_filter` and calls
`cuvs::neighbors::cagra::search` once per query. The outer runner still batches
100 predicates at a time so predicate and mask processing remain directly
comparable with the other methods.

The ACORN/NaviX byte path preserves the same online predicate-result boundary
as ORCHID and CAGRA. Predicate evaluation first returns mixed posting-list or
representation-native dense-mask results, then a separate materialization
stage writes those results into the reusable byte-per-document output buffer
required by the baseline adapters. Dense masks stay packed for ORCHID/CAGRA
and byte-addressed for ACORN/NaviX. The two stages are timed independently as
`predicate_eval` and `mask_materialize`.

Build that module in the isolated CAGRA Pixi environment:

```bash
pixi run --manifest-path ../cagra_env/pixi.toml build-cagra
```

The task builds `../cagra_native` against the pinned cuVS 25.08 environment,
installs `_cagra_native` beside the CPU baseline adapters, and verifies the
upstream artifact provenance exported by the module.

For an existing index, a run is configured with a dataset config and one or
more `ef_search` values:

```bash
python ../perf_navix.py \
  --config ../config/perf/navix.yaml \
  --dset-cfg ../config/dataset/sift12.yaml \
  --index /path/to/navix.index
```

The checked configs cover every published dataset and fix the paper batch size,
normalization, graph parameters, search sweep, and metric. ACORN's config also
makes its one-scalar-per-document construction policy explicit: SIFT equality
uses the exact workload scalar, while range and multi-label workloads use dummy
constructor metadata. All online filtering still comes from the per-query byte
mask.

Index construction is opt-in through `--build-index`. Direct CLI use without a
config requires either an explicit `int32` NumPy metadata file,
`--acorn-workload-metadata`, or `--acorn-dummy-metadata`. Predicate validation
against the legacy Zarr mask is also opt-in through `--validate-predicates`.
`--precompute-masks` materializes every query mask before warmup and timing; it
provides a native filtered-search-only ablation while preserving identical
query masks and recall semantics.

Like `perf.py`, each baseline run writes a local recall/QPS CSV and a detailed
stage-timing CSV under `logs/<dataset>/`. See the repository README for the
complete environment, build, and run commands.
