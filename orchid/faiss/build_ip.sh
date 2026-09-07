#!/bin/bash
cmake -S . -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_MKL=ON \
		-DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=avx512_spr \
		-DBUILD_TESTING=OFF \
		-DMKL_ROOT=/opt/intel/oneapi/mkl/latest \

cmake --build build --target bench_inner_product_batch -j

