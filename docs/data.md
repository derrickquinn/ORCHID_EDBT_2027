# Evaluation input specification

SIFT-12 has a download and preparation recipe below. Other YAML files describe
existing paper workloads and require externally prepared inputs.

Paths in `orchid/config/dataset/*.yaml` are resolved from `orchid/` by the
launcher. Edit a config or pass a custom dataset config to use other locations.

| Workload | Vector store under `data/` | Predicate/ground-truth store under `data/` |
| --- | --- | --- |
| SIFT equality | `sift1m.zarr` | `sift1m_c12.zarr`, `sift1m_c24.zarr`, `sift1m_c48.zarr` |
| SIFT ranges | `sift1m.zarr` | `sift1m_c1000_r10.zarr`, `sift1m_c1000_r20.zarr`, `sift1m_c1000_r40.zarr` |
| LAION | `laion1m.zarr` | `laion_all.zarr`, `laion_pos.zarr`, `laion_neg.zarr` |
| YFCC conjunction | `yfcc10m.zarr` | `yfcc10m.zarr` |
| YFCC single tag | `yfcc10m_single.zarr` | `yfcc10m_single.zarr` |

Vector stores contain `train` (documents × dimensions) and `test` (queries ×
dimensions). Predicate stores contain `neighbors` (queries × k) with external
document IDs. ORCHID's offline signature construction also reads `pred_mask`
(queries × documents). That mask is an offline oracle for predicate validation;
the measured path evaluates predicates through query-independent indexes.
Calibration uses `pred_mt` if it covers the validation window, or transposes in memory.

The default test trace uses rows 1024–11023 (10,000 queries), reserving rows
0–511 for SVD fitting and 512–1023 for validation. CPU B=100, GPU B=1, and k=10.
For a 10,000-row store, use `--num_queries 8976` (ORCHID/IVF) or
`--num-queries 8976` (baselines); the final partial CPU batch is retained.
Document order, query order, metadata, and ground-truth IDs
must correspond. Paper configs normalize vectors consistently across methods.

## Predicate metadata

`ORCHID_ASSET_ROOT` points to the directory containing these external assets:

```text
assets/
  data/raw/yfcc100M/
    base.metadata.10M.spmat
    query.metadata.private.2727415019.100K.spmat
  setup/laion/
    keywords.jsonl
    predicate_queries.txt
```

For LAION, each JSONL record supplies `top_30_keywords`; the query-term file
contains recovered atomic term IDs for the existing query trace. Neither file
is bundled. YFCC uses the sparse document and query metadata above. SIFT scalar
attributes and query predicates use the existing deterministic seed-42 workload;
SIFT-12 vectors and ground truth are prepared by the recipe below.

`ORCHID_ASSET_ROOT` changes metadata lookup paths, not vector-store paths in
dataset YAML. Both must point to corresponding inputs. Caches are created under
`orchid/cached/`; use fresh caches after changing data or metadata.

## Prepare SIFT-12

After `pixi install`, run from the repository root:

```bash
pixi run prepare-sift12
pixi run benchmark orchid sift12 -- --num_queries 8976
```

Build FAISS before benchmarking as described in the main README. Preparation
itself needs neither a GPU nor native FAISS/baseline builds. It explicitly uses
`zarrs.ZarrsCodecPipeline` and inherits the caller's CPU/NUMA placement. Allow
several minutes and conservatively reserve 25 GB of disk (the two boolean mask
layouts total 20 GB before compression), plus the 501 MiB source download.
Writes are streamed in blocks, without allocating either full mask in memory.

The output is Zarr v3, using the local stores' shapes, dtypes, chunks and Zstd
settings:

| Store | Array | Shape / dtype |
| --- | --- | --- |
| `sift1m.zarr` | `train` | 1,000,000 × 128 float32 |
| `sift1m.zarr` | `test` | 10,000 × 128 float32 |
| `sift1m.zarr` | `neighbors_unfiltered` | 10,000 × 10 int64 (L2) |
| `sift1m_c12.zarr` | `pred_mask` | 10,000 × 1,000,000 bool |
| `sift1m_c12.zarr` | `pred_mt` | 1,000,000 × 10,000 bool |
| `sift1m_c12.zarr` | `neighbors` | 10,000 × 10 int64 (filtered cosine) |

Document and query ordering is unchanged from ANN-Benchmarks. The source is
checksum-pinned, then multiplied by a bundled recovered rotation; vectors are
not normalized at preparation time. The maximum difference measured against
all local vectors is `1.52587890625e-5`. Predicate generation draws all 1,000,000
int64 document labels and then all 10,000 query labels from
`np.random.default_rng(42).integers(0, 12, ...)`, matching the online evaluator.
The two neighbor tables are copied from a small checksum-pinned reference
archive, preserving the local IDs and ordering exactly. They are not taken from
the unfiltered ANN-Benchmarks HDF5 neighbor table. See
[reference provenance](../scripts/data/README.md).

To reuse an existing download or select another output location:

```bash
pixi run prepare-sift12 --source /path/to/sift-128-euclidean.hdf5
pixi run prepare-sift12 --output-dir /path/to/assets/data
```

`--source` is still checksum-verified. By default the download is cached at
`data/raw/sift-128-euclidean.hdf5`. Failed downloads are not cached. For a custom
output directory, point the repository's `data` symlink there or change the
vector/predicate paths in a dataset YAML. `ORCHID_ASSET_ROOT` does not redirect
those paths. No external predicate metadata is needed for SIFT-12.

A completed run writes `sift12-preparation.json`. Repeating the command checks
that recipe marker and the array shapes/dtypes, then reuses the stores; it does
not audit every chunk's contents. Existing stores without a matching marker
are never overwritten. Generation happens in a temporary directory, with the
completion marker published last. After an interrupted final publication, use
a fresh output directory. If the process was forcibly killed, remove its
`.sift12-preparation.lock` only after confirming it is no longer running.
Use fresh benchmark caches when changing input stores.

Preparation regression tests (tiny synthetic files, no network or GPU):

```bash
pixi run python -m unittest discover -s orchid/tests -p test_prepare_sift12.py -v
```
