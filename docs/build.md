# Native builds

`SOURCE_VERSIONS.json` identifies the research-code snapshot and bundled FAISS
revision. Native kernels and adapters are included as source. No build requires
a private repository checkout.

## CPU

`pixi run build-faiss` builds the bundled generic and AVX2 FAISS Python modules
and installs them into the active environment. To override compilation:

```bash
pixi run build-faiss --jobs 16 --opt-level avx512_spr
```

Only select an instruction set supported by the evaluation machine. Predicate
kernels require AVX2 even with FAISS `--opt-level generic`. The build uses MKL
from the environment instead of a fixed oneAPI installation.

ACORN and NaviX are independent native extensions, each linked against its own
pinned FAISS fork. To reuse source checkouts without fetching:

```bash
pixi run build-baselines --no-fetch \
  --acorn-source /path/to/ACORN --navix-source /path/to/faiss-navix
```

| Component | Revision |
| --- | --- |
| ACORN | `3996dcf90ca2fb58abe46679a13d6a57792274c2` |
| NaviX | `25d563a2dec9087891f19055b8407d4383a26342` |
| CAGRA adapter upstream | `79c153ffc4ebf4363f522097958f83d7eec19ad4` |

The builder rejects wrong revisions and tracked source modifications. ACORN's
upstream CMake may still fetch its declared build dependencies unless cached.
See [the adapter contract](../orchid/baseline_adapters/README.md) for manual builds.

## GPU

CAGRA's separate Pixi manifest pins cuVS/CUDA. Its default CUDA architecture is
detected from the build machine; cross-compilation can specify
`pixi run --manifest-path orchid/cagra_env/pixi.toml build-cagra --cuda-architectures 89`.

The harness retains optional FAISS GPU search (`--gpu`) and GPU clustering
(`--gpu_build`). To use them, add a compatible CUDA toolkit to the main
environment and rebuild with `pixi run build-faiss --gpu`. The default main
environment builds CPU FAISS; CAGRA is independently available through its
dedicated environment. GPU pre-filter is not supported by the harness.

## Checks

`pixi run check` validates configs and runs synthetic harness/kernel tests.
`pixi run smoke-baselines` checks native filtered search and index save/load.
CAGRA's smoke task requires an NVIDIA GPU.

`pixi run smoke-harness` runs complete ORCHID/ORCHID-P/IVF/pre-filter traces on
temporary synthetic Zarr inputs and checks recall plus local CSV output.
After building the baseline adapters, add `--methods acorn navix` to test their
complete traces. CAGRA's environment has its own `smoke-harness` task. These are
correctness checks, not performance measurements, and their inputs are deleted
on completion.

For real-data checks, use predicate validation and a bounded sweep over local
inputs. Keep Zarrs enabled and preserve CPU/NUMA placement. When running through
an agent sandbox, request elevated execution before any command that may access
a Zarr store.
