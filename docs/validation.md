# Packaging validation

Validated on Linux x86-64 with an NVIDIA RTX 6000 Ada GPU:

- Installed both Pixi environments from the artifact manifests and lockfiles.
- Built bundled FAISS (generic and AVX2 Python modules) and `utils_cpp`.
- Fetched the pinned public ACORN/NaviX sources and built both adapters.
- Built the CAGRA adapter against the separate cuVS 25.08/CUDA 12.8 environment.
- Validated 45 YAML configs and passed 66 regression tests.
- Passed native filtered-search and save/load checks for ACORN, NaviX, and CAGRA.
- Ran complete synthetic traces for ORCHID, ORCHID-P, IVF, pre-filter, ACORN,
  NaviX, and CAGRA, including predicate-oracle checks and local CSV export.

The synthetic harness check uses 2,048 generated 32-dimensional vectors and
100 queries, with temporary masks and exact ground truth. Its inputs are
deleted afterward. These runs establish correctness and packaging integration;
they are not paper performance measurements. No paper datasets were included
or used for this validation. Optional FAISS GPU search was not built or tested.

The ORCHID measured trace and baseline online/search-only loops match the
research snapshot in `SOURCE_VERSIONS.json`. Artifact-specific changes handle
packaging, local output, and command-line entry points.
