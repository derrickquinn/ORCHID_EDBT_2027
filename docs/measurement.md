# Online measurement protocol

The artifact uses the manuscript query split and CPU B=100 / GPU B=1 protocol.
Loading data, building indexes, clustering, and loading ground truth are outside
timing. Predicate evaluation, mask construction, and filtered search are inside.

ORCHID, ORCHID-P, and IVF share these non-overlapping timing stages:

1. `predicate_eval`: evaluate query predicates against document indexes.
2. `mask_materialize`: turn mixed posting/bitmap results into a packed bitmap.
3. `cluster_scoring`: centroid scores and, for ORCHID, per-cluster selectivity
   and log-scaled score adjustment.
4. `cluster_rank`: native top-k cluster selection.
5. `search_prep`: selected centroid distances/IDs and the bitmap selector.
6. `vector_search`: native filtered vector search.
7. `other`: remaining work, including internal-to-external ID mapping.
8. `total`: complete batch latency.

Cluster-major IDs let ORCHID use the same packed bitmap for selectivity and
filtered search. Returned IDs are mapped back before recall is computed.
Pre-filter uses the same predicate path with exact filtered flat search; its
launcher selects one sweep point.

ACORN/NaviX/CAGRA share `predicate_eval`, `mask_materialize`, `vector_search`,
`other`, and `total`. ACORN/NaviX require byte masks; CAGRA uses a packed bitmap.
Representation-specific materialization is charged to each method. CAGRA issues
one filtered cuVS search per query with predicate processing and transfer also B=1.

QPS uses the complete measured trace, including loop and timing overhead. Stage
CSVs report median and p95 batch latency in milliseconds. Recall@k uses external
document IDs and supplied ground truth.

Paper configs use one full-trace warmup and adaptive measurements from 3 to 100
traces with a target relative QPS confidence half-width of 5%. Raw CLI defaults
use three warmups; paper configs override this to one. Non-convergence raises
an error before final CSV export.

`--validate_predicates` (ORCHID/IVF) or `--validate-predicates` (baselines)
compares generated predicates with stored masks outside timing. Baseline
`--precompute-masks` is a separate search-only ablation; it removes predicate
evaluation and materialization from the measured path.
