# Evaluation input specification

No datasets, precomputed masks, ground truth, or metadata are released here.
The YAML files describe existing paper workloads and require locally prepared
inputs; they are not download or generation recipes.

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
vectors and ground truth remain external.

`ORCHID_ASSET_ROOT` changes metadata lookup paths, not vector-store paths in
dataset YAML. Both must point to corresponding inputs. Caches are created under
`orchid/cached/`; use fresh caches after changing data or metadata.
