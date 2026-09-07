#!/bin/bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build_python}"
FAISS_ENABLE_MKL="${FAISS_ENABLE_MKL:-ON}"

# Pixi-based build (uses the local manifest in faiss-src).
PY=$(pixi run python -c "import sys; print(sys.executable)")
NPY=$(pixi run python -c "import numpy as np; print(np.get_include())")
PREFIX=$(pixi run python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))")
PY_INCLUDE=$(pixi run python -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIBDIR=$(pixi run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')")
PY_LDLIB=$(pixi run python -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY') or '')")
PY_LIBRARY=""
if [[ -n "$PY_LIBDIR" && -n "$PY_LDLIB" && -f "$PY_LIBDIR/$PY_LDLIB" ]]; then
	PY_LIBRARY="$PY_LIBDIR/$PY_LDLIB"
fi

MKL_ROOT="${MKL_ROOT:-/opt/intel/oneapi/mkl/latest}"
MKL_LIBDIR="${MKL_LIBDIR:-$MKL_ROOT/lib}"
MKL_CMAKE_PATH="${MKL_CMAKE_PATH:-$MKL_LIBDIR/cmake/mkl}"
ONEAPI_COMPILER_PATH="${ONEAPI_COMPILER_PATH:-/opt/intel/oneapi/compiler/latest}"
MKL_LIBRARIES="${MKL_LIBRARIES:--Wl,--start-group;${MKL_LIBDIR}/libmkl_intel_lp64.a;${MKL_LIBDIR}/libmkl_gnu_thread.a;${MKL_LIBDIR}/libmkl_core.a;-Wl,--end-group}"

cmake_args=(
	-S .
	-B "$BUILD_DIR"
	-DFAISS_ENABLE_PYTHON=ON
	-DFAISS_ENABLE_GPU=ON
	-DFAISS_ENABLE_MKL="$FAISS_ENABLE_MKL"
	-DCMAKE_BUILD_TYPE=Release
	-DFAISS_OPT_LEVEL=avx512_spr
	-DBUILD_TESTING=OFF
	-DMKL_ROOT="$MKL_ROOT"
	-DBLA_VENDOR=Intel10_64_dyn
	-DCMAKE_PREFIX_PATH="$MKL_CMAKE_PATH;$ONEAPI_COMPILER_PATH;$PREFIX"
	"-DMKL_LIBRARIES=$MKL_LIBRARIES"
	-DPython_EXECUTABLE="$PY"
	-DPython_ROOT_DIR="$PREFIX"
	-DPython_INCLUDE_DIRS="$PY_INCLUDE"
	-DPython_NumPy_INCLUDE_DIRS="$NPY"
)
if [[ -n "$PY_LIBRARY" ]]; then
	cmake_args+=("-DPython_LIBRARY=$PY_LIBRARY")
fi

cmake "${cmake_args[@]}"

cmake --build "$BUILD_DIR" --target swigfaiss -j
